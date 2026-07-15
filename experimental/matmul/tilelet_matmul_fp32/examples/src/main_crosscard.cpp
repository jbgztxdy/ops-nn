/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file main_crosscard.cpp
 * \brief Two-card ("peer 直存") gate test driver for TileletMatmulFp32.
 *
 * Runs the same op cooperatively on two NPUs: the SOURCE rank (dev0) stages the
 * remote-tile A/B into a peer-visible arena on the COMPUTE card and computes its
 * local direct tiles into its own C; the COMPUTE rank (dev1) reads the arena,
 * computes the remote tiles and writes them straight back into the source card's
 * C over the link (peer_out). Both ranks run concurrently on their own stream;
 * the host zeroes the arena beforehand and syncs both streams, then verifies C.
 *
 * This is the gate that validates kernel-issued cross-card DataCopy + cross-card
 * AB-signal polling for the peer-store design.
 *
 * Usage (from examples/output, after building the op + this driver):
 *   ASCEND_RT_VISIBLE_DEVICES=0,1 \
 *   TILELET_DTYPE=bf16 TILELET_TRANSPOSE_X2=1 \
 *   TILELET_M=256 TILELET_K=64 TILELET_N=512 \
 *   ./execute_tilelet_matmul_fp32_crosscard
 * then: TILELET_DTYPE=bf16 python3 ../scripts/verify_result.py output/output_z.bin output/golden.bin
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_tilelet_matmul_fp32.h"

namespace {

constexpr int kSourceDevice = 0;
constexpr int kComputeDevice = 1;
// Generous peer arena; the op only touches the (small) staging/signal prefix.
constexpr size_t kArenaBytes = 512UL * 1024UL * 1024UL;

bool Check(aclError ret, const char* what) {
  if (ret != ACL_SUCCESS) {
    printf("[FAIL] %s -> %d\n", what, static_cast<int>(ret));
    return false;
  }
  return true;
}

int64_t EnvI64(const char* name, int64_t dflt) {
  const char* v = std::getenv(name);
  return v == nullptr ? dflt : std::strtoll(v, nullptr, 10);
}

bool EnvFlag(const char* name, bool dflt) {
  const char* v = std::getenv(name);
  if (v == nullptr) return dflt;
  return v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y';
}

bool IsBf16() {
  const char* v = std::getenv("TILELET_DTYPE");
  return v != nullptr && std::string(v) == "bf16";
}

bool ReadFile(const std::string& path, std::vector<uint8_t>& buf, size_t bytes) {
  FILE* f = fopen(path.c_str(), "rb");
  if (f == nullptr) {
    printf("[FAIL] open %s\n", path.c_str());
    return false;
  }
  buf.resize(bytes);
  size_t got = fread(buf.data(), 1, bytes, f);
  fclose(f);
  if (got != bytes) {
    printf("[FAIL] read %s: got %zu want %zu\n", path.c_str(), got, bytes);
    return false;
  }
  return true;
}

bool WriteFile(const std::string& path, const void* data, size_t bytes) {
  FILE* f = fopen(path.c_str(), "wb");
  if (f == nullptr) {
    printf("[FAIL] open %s for write\n", path.c_str());
    return false;
  }
  fwrite(data, 1, bytes, f);
  fclose(f);
  return true;
}

// Allocate a device buffer on the current device and copy host bytes into it.
void* DevFromHost(const void* host, size_t bytes) {
  void* dev = nullptr;
  if (aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return nullptr;
  if (host != nullptr && aclrtMemcpy(dev, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    aclrtFree(dev);
    return nullptr;
  }
  return dev;
}

aclTensor* MakeTensor(void* dev, const std::vector<int64_t>& shape, aclDataType dtype) {
  return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(), dev);
}

// One rank's cooperative op invocation. Runs GetWorkspaceSize + execute on the
// current device's stream; does not synchronize (caller syncs both streams).
struct RankPlan {
  aclTensor* x1;
  aclTensor* x2;
  aclTensor* arena;
  aclTensor* peerOut;  // null for source
  aclTensor* out;
  int64_t role;
  int64_t rankId;
};

bool LaunchRank(const RankPlan& p, bool transX2, int64_t remoteStart, int64_t remoteCount, int64_t wfM, int64_t wfN,
                int64_t commCores, int64_t commKTiles, aclrtStream stream, void** wsOut) {
  uint64_t wsSize = 0;
  aclOpExecutor* exec = nullptr;
  aclError ret = aclnnTileletMatmulFp32GetWorkspaceSize(
      p.x1, p.x2, /*bias=*/nullptr, p.arena, p.peerOut,
      /*transposeX1=*/false, transX2, remoteStart, remoteCount, wfM, wfN, commCores, commKTiles,
      /*enableDCopyback=*/false, p.role, p.rankId, /*worldSize=*/2, p.out, &wsSize, &exec);
  if (!Check(ret, "GetWorkspaceSize")) return false;
  void* ws = nullptr;
  if (wsSize != 0 && !Check(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws")) return false;
  *wsOut = ws;
  if (!Check(aclnnTileletMatmulFp32(ws, wsSize, exec, stream), "execute")) return false;
  return true;
}

}  // namespace

int main() {
  aclError init = aclInit(nullptr);
  if (init != ACL_SUCCESS) printf("[warn] aclInit -> %d (continuing)\n", static_cast<int>(init));

  const bool bf16 = IsBf16();
  const aclDataType dtype = bf16 ? ACL_BF16 : ACL_FLOAT;
  const size_t elem = bf16 ? 2 : 4;
  const bool transX2 = EnvFlag("TILELET_TRANSPOSE_X2", true);
  const int64_t M = EnvI64("TILELET_M", 256);
  const int64_t K = EnvI64("TILELET_K", 64);
  const int64_t N = EnvI64("TILELET_N", 512);
  const int64_t wfM = EnvI64("TILELET_WAVEFRONT_M", 16);
  const int64_t wfN = EnvI64("TILELET_WAVEFRONT_N", 8);
  const int64_t commCores = EnvI64("TILELET_COMM_CORE_NUM", EnvI64("TILELET_COMM_SMS", 8));
  const int64_t commKTiles = EnvI64("TILELET_COMM_K_TILES", 32);
  const int64_t remoteStart = EnvI64("TILELET_REMOTE_TILE_START", 2);
  const int64_t remoteCount = EnvI64("TILELET_REMOTE_TILE_COUNT", 2);

  const size_t aBytes = static_cast<size_t>(M) * K * elem;                    // x1 [M,K]
  const size_t bBytes = static_cast<size_t>(N) * K * elem;                    // x2 [N,K] (transpose_x2)
  const size_t cBytes = static_cast<size_t>(M) * N * elem;                    // C  [M,N]
  const std::vector<int64_t> shapeA = {M, K};
  const std::vector<int64_t> shapeB = transX2 ? std::vector<int64_t>{N, K} : std::vector<int64_t>{K, N};
  const std::vector<int64_t> shapeC = {M, N};
  const std::vector<int64_t> shapeArena = {static_cast<int64_t>(kArenaBytes / elem)};

  std::vector<uint8_t> hostA, hostB;
  if (!ReadFile("../input/input_a.bin", hostA, aBytes)) return 1;
  if (!ReadFile("../input/input_b.bin", hostB, bBytes)) return 1;

  // Enable bidirectional peer access between the two cards.
  if (!Check(aclrtSetDevice(kSourceDevice), "setDevice src")) return 1;
  aclError pa0 = aclrtDeviceEnablePeerAccess(kComputeDevice, 0);
  if (pa0 != ACL_SUCCESS) printf("[warn] enablePeer src->compute -> %d\n", static_cast<int>(pa0));
  if (!Check(aclrtSetDevice(kComputeDevice), "setDevice compute")) return 1;
  aclError pa1 = aclrtDeviceEnablePeerAccess(kSourceDevice, 0);
  if (pa1 != ACL_SUCCESS) printf("[warn] enablePeer compute->src -> %d\n", static_cast<int>(pa1));

  // Arena lives on the COMPUTE card (dev1), zeroed so all signals start at 0.
  void* arena = nullptr;
  if (!Check(aclrtMalloc(&arena, kArenaBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc arena")) return 1;
  if (!Check(aclrtMemset(arena, kArenaBytes, 0, kArenaBytes), "zero arena")) return 1;
  void* x1Compute = DevFromHost(hostA.data(), aBytes);
  void* x2Compute = DevFromHost(hostB.data(), bBytes);
  void* outDummy = DevFromHost(nullptr, cBytes);  // compute rank writes to peer_out, not this
  if (x1Compute == nullptr || x2Compute == nullptr || outDummy == nullptr) return 1;

  // Source card (dev0): its own x1/x2 and the real output C (zeroed).
  if (!Check(aclrtSetDevice(kSourceDevice), "setDevice src2")) return 1;
  void* x1Source = DevFromHost(hostA.data(), aBytes);
  void* x2Source = DevFromHost(hostB.data(), bBytes);
  void* cSource = nullptr;
  if (!Check(aclrtMalloc(&cSource, cBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc C")) return 1;
  if (!Check(aclrtMemset(cSource, cBytes, 0, cBytes), "zero C")) return 1;
  if (x1Source == nullptr || x2Source == nullptr) return 1;

  // Streams: one per device.
  aclrtStream streamSrc = nullptr;
  aclrtStream streamCompute = nullptr;
  if (!Check(aclrtCreateStream(&streamSrc), "stream src")) return 1;  // current device = src
  if (!Check(aclrtSetDevice(kComputeDevice), "setDevice compute3")) return 1;
  if (!Check(aclrtCreateStream(&streamCompute), "stream compute")) return 1;

  // Tensors (each wraps a device ptr; peer ptrs are usable cross-card).
  aclTensor* tArena = MakeTensor(arena, shapeArena, dtype);
  aclTensor* tX1c = MakeTensor(x1Compute, shapeA, dtype);
  aclTensor* tX2c = MakeTensor(x2Compute, shapeB, dtype);
  aclTensor* tOutDummy = MakeTensor(outDummy, shapeC, dtype);
  aclTensor* tPeerOut = MakeTensor(cSource, shapeC, dtype);  // source C, peer-written by compute
  aclTensor* tX1s = MakeTensor(x1Source, shapeA, dtype);
  aclTensor* tX2s = MakeTensor(x2Source, shapeB, dtype);
  aclTensor* tCSource = MakeTensor(cSource, shapeC, dtype);

  void* wsSrc = nullptr;
  void* wsCompute = nullptr;

  // Launch source (dev0) and compute (dev1) concurrently, then sync both.
  if (!Check(aclrtSetDevice(kSourceDevice), "setDevice src launch")) return 1;
  RankPlan src{tX1s, tX2s, tArena, /*peerOut=*/nullptr, tCSource, /*role=*/0, /*rankId=*/0};
  if (!LaunchRank(src, transX2, remoteStart, remoteCount, wfM, wfN, commCores, commKTiles, streamSrc, &wsSrc))
    return 1;

  if (!Check(aclrtSetDevice(kComputeDevice), "setDevice compute launch")) return 1;
  RankPlan cmp{tX1c, tX2c, tArena, tPeerOut, tOutDummy, /*role=*/1, /*rankId=*/1};
  if (!LaunchRank(cmp, transX2, remoteStart, remoteCount, wfM, wfN, commCores, commKTiles, streamCompute, &wsCompute))
    return 1;

  if (!Check(aclrtSetDevice(kSourceDevice), "setDevice src sync")) return 1;
  if (!Check(aclrtSynchronizeStreamWithTimeout(streamSrc, 600000), "sync src")) return 1;
  if (!Check(aclrtSetDevice(kComputeDevice), "setDevice compute sync")) return 1;
  if (!Check(aclrtSynchronizeStreamWithTimeout(streamCompute, 600000), "sync compute")) return 1;

  // Read the assembled C back from the source card and dump for verify_result.py.
  std::vector<uint8_t> hostC(cBytes, 0);
  if (!Check(aclrtSetDevice(kSourceDevice), "setDevice readback")) return 1;
  if (!Check(aclrtMemcpy(hostC.data(), cBytes, cSource, cBytes, ACL_MEMCPY_DEVICE_TO_HOST), "readback C")) return 1;
  if (!WriteFile("output_z.bin", hostC.data(), cBytes)) return 1;

  printf("cross-card run done: M=%ld N=%ld K=%ld remote=[%ld,%ld) comm=%ld -> output/output_z.bin\n", M, N, K,
         remoteStart, remoteStart + remoteCount, commCores);
  printf("verify: TILELET_DTYPE=%s python3 ../scripts/verify_result.py output/output_z.bin output/golden.bin\n",
         bf16 ? "bf16" : "float32");
  aclFinalize();
  return 0;
}
