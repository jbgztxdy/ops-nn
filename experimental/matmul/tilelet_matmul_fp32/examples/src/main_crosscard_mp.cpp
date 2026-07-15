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
 * \file main_crosscard_mp.cpp
 * \brief Two-PROCESS cross-card gate for TileletMatmulFp32 (framework de-risk).
 *
 * Same cooperative source/compute execution as main_crosscard.cpp, but the two
 * ranks run in SEPARATE PROCESSES that share device buffers via the ACL IPC-mem
 * key API (aclrtIpcMemGetExportKey / aclrtIpcMemImportByKey +
 * ENABLE_PEER_ACCESS) — the Ascend equivalent of CUDA's cross-process same-VA
 * allocator. This proves the peer-visible arena + kernel peer-store work across
 * processes, which is the foundation for the PD-disaggregated vLLM integration
 * (prefill process = source rank, decode process = compute rank).
 *
 * Rendezvous is a shared directory of key files (TILELET_MP_DIR). Buffers:
 *   - compute process owns the ARENA (staging+AB signals), exports its key.
 *   - source  process owns C (peer_out) and the D-done buffer (peer_dsignal),
 *     exports their keys.
 * Each imports the other's with peer access, then both launch the op (role 0/1)
 * concurrently. The source kernel waits in-kernel on the cross-card D-done, so
 * its stream completing implies C is fully assembled (no host cross-sync).
 *
 * Usage (run both, any order):
 *   TILELET_MP_DIR=/tmp/mp TILELET_DTYPE=bf16 TILELET_TRANSPOSE_X2=1 \
 *   TILELET_M=256 TILELET_K=64 TILELET_N=512 \
 *   ASCEND_RT_VISIBLE_DEVICES=0,1 ./execute_tilelet_matmul_fp32_crosscard_mp source
 *   ... (same env) ... ./execute_tilelet_matmul_fp32_crosscard_mp compute
 * then verify output/output_z.bin against golden.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

#include "acl/acl.h"
#include "aclnn_tilelet_matmul_fp32.h"
#include "tilelet_host_layout.h"

namespace {

constexpr size_t kKeyLen = 256;
constexpr size_t kArenaSlackBytes = 1UL * 1024UL * 1024UL;

bool Check(aclError ret, const char* what) {
  if (ret != ACL_SUCCESS) {
    printf("[FAIL] %s -> %d\n", what, static_cast<int>(ret));
    return false;
  }
  return true;
}

int64_t EnvI64(const char* n, int64_t d) { const char* v = std::getenv(n); return v ? std::strtoll(v, nullptr, 10) : d; }
bool EnvFlag(const char* n, bool d) { const char* v = std::getenv(n); return v ? (v[0]=='1'||v[0]=='t'||v[0]=='T'||v[0]=='y'||v[0]=='Y') : d; }
std::string EnvStr(const char* n, const char* d) { const char* v = std::getenv(n); return v ? std::string(v) : std::string(d); }
bool IsBf16() { const char* v = std::getenv("TILELET_DTYPE"); return v && std::string(v) == "bf16"; }

void WaitFile(const std::string& p) { while (access(p.c_str(), 0) != 0) usleep(20000); }

bool WriteKey(const std::string& path, const char* key) {
  FILE* f = fopen((path + ".tmp").c_str(), "wb");
  if (!f) return false;
  fwrite(key, 1, kKeyLen, f);
  fclose(f);
  return rename((path + ".tmp").c_str(), path.c_str()) == 0;  // atomic publish
}
bool ReadKey(const std::string& path, char* key) {
  WaitFile(path);
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  size_t got = fread(key, 1, kKeyLen, f);
  fclose(f);
  return got == kKeyLen;
}

bool ReadFileBytes(const std::string& path, std::vector<uint8_t>& buf, size_t bytes) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { printf("[FAIL] open %s\n", path.c_str()); return false; }
  buf.resize(bytes);
  size_t got = fread(buf.data(), 1, bytes, f);
  fclose(f);
  if (got != bytes) { printf("[FAIL] read %s got %zu want %zu\n", path.c_str(), got, bytes); return false; }
  return true;
}

void* DevFromHost(const void* host, size_t bytes) {
  void* dev = nullptr;
  if (aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return nullptr;
  if (host && aclrtMemcpy(dev, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) { aclrtFree(dev); return nullptr; }
  return dev;
}

aclTensor* MakeTensor(void* dev, const std::vector<int64_t>& shape, aclDataType dtype) {
  return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(), dev);
}

bool LaunchRank(aclTensor* x1, aclTensor* x2, aclTensor* arena, aclTensor* peerOut, aclTensor* peerDsignal,
                aclTensor* out, int64_t role, int64_t rankId, bool transX2, int64_t remoteStart, int64_t remoteCount,
                int64_t wfM, int64_t wfN, int64_t commCores, int64_t commKTiles, bool enDCopyback, aclrtStream stream,
                void** wsOut) {
  uint64_t wsSize = 0;
  aclOpExecutor* exec = nullptr;
  aclError ret = aclnnTileletMatmulFp32GetWorkspaceSize(
      x1, x2, nullptr, arena, peerOut, peerDsignal, false, transX2, remoteStart, remoteCount, wfM, wfN, commCores,
      commKTiles, enDCopyback, role, rankId, 2, out, &wsSize, &exec);
  if (!Check(ret, "GetWorkspaceSize")) return false;
  void* ws = nullptr;
  if (wsSize && !Check(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws")) return false;
  *wsOut = ws;
  return Check(aclnnTileletMatmulFp32(ws, wsSize, exec, stream), "execute");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: %s source|compute\n", argv[0]); return 2; }
  const std::string role = argv[1];
  const bool isSource = (role == "source");
  aclInit(nullptr);

  const bool bf16 = IsBf16();
  const aclDataType dtype = bf16 ? ACL_BF16 : ACL_FLOAT;
  const size_t elem = bf16 ? 2 : 4;
  const bool transX2 = EnvFlag("TILELET_TRANSPOSE_X2", true);
  const int64_t M = EnvI64("TILELET_M", 256), K = EnvI64("TILELET_K", 64), N = EnvI64("TILELET_N", 512);
  const int64_t wfM = EnvI64("TILELET_WAVEFRONT_M", 16), wfN = EnvI64("TILELET_WAVEFRONT_N", 8);
  const int64_t commCores = EnvI64("TILELET_COMM_CORE_NUM", 8), commKTiles = EnvI64("TILELET_COMM_K_TILES", 32);
  const int64_t remoteStart = EnvI64("TILELET_REMOTE_TILE_START", 2), remoteCount = EnvI64("TILELET_REMOTE_TILE_COUNT", 2);
  const bool enDCopyback = EnvFlag("TILELET_D_COPYBACK", false);
  const int srcDev = static_cast<int>(EnvI64("TILELET_SRC_DEV", 0));
  const int cmpDev = static_cast<int>(EnvI64("TILELET_CMP_DEV", 1));
  const std::string dir = EnvStr("TILELET_MP_DIR", "/home/zhipeng/tmp/mp_gate");
  const int myDev = isSource ? srcDev : cmpDev;

  const size_t aBytes = static_cast<size_t>(M) * K * elem;
  const size_t bBytes = static_cast<size_t>(N) * K * elem;
  const size_t cBytes = static_cast<size_t>(M) * N * elem;

  tilelet::ProblemParams pp;
  pp.m = M; pp.n = N; pp.k = K; pp.elemSize = elem; pp.transposeX2 = transX2; pp.enableDCopyback = enDCopyback;
  pp.wavefrontM = wfM; pp.wavefrontN = wfN; pp.commCoreNum = commCores; pp.commKTiles = commKTiles;
  const tilelet::Layout layout = tilelet::ComputeLayout(pp);
  const size_t arenaBytes = static_cast<size_t>(layout.arenaBytes) + kArenaSlackBytes;
  const size_t dsigBytes = static_cast<size_t>(layout.dSignalBytes) + kArenaSlackBytes;

  if (!Check(aclrtSetDevice(myDev), "setDevice")) return 1;
  aclrtStream stream = nullptr;
  if (!Check(aclrtCreateStream(&stream), "createStream")) return 1;

  std::vector<uint8_t> hostA, hostB;
  if (!ReadFileBytes("../input/input_a.bin", hostA, aBytes)) return 1;
  if (!ReadFileBytes("../input/input_b.bin", hostB, bBytes)) return 1;
  const std::vector<int64_t> shapeA = {M, K};
  const std::vector<int64_t> shapeB = transX2 ? std::vector<int64_t>{N, K} : std::vector<int64_t>{K, N};
  const std::vector<int64_t> shapeC = {M, N};
  char key[kKeyLen];
  void* ws = nullptr;

  if (isSource) {
    // Source owns x1/x2, the real C (peer_out target) and the D-done buffer.
    void* x1s = DevFromHost(hostA.data(), aBytes);
    void* x2s = DevFromHost(hostB.data(), bBytes);
    void* cSrc = nullptr; if (!Check(aclrtMalloc(&cSrc, cBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc C")) return 1;
    if (!Check(aclrtMemset(cSrc, cBytes, 0, cBytes), "zero C")) return 1;
    void* dsig = nullptr; if (!Check(aclrtMalloc(&dsig, dsigBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc dsig")) return 1;
    if (!Check(aclrtMemset(dsig, dsigBytes, 0, dsigBytes), "zero dsig")) return 1;
    // Export C + dsig keys for the compute process.
    memset(key, 0, kKeyLen);
    if (!Check(aclrtIpcMemGetExportKey(cSrc, cBytes, key, kKeyLen, ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION), "export C")) return 1;
    WriteKey(dir + "/c.key", key);
    memset(key, 0, kKeyLen);
    if (!Check(aclrtIpcMemGetExportKey(dsig, dsigBytes, key, kKeyLen, ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION), "export dsig")) return 1;
    WriteKey(dir + "/dsig.key", key);
    // Import the compute process's arena (peer access).
    char akey[kKeyLen]; if (!ReadKey(dir + "/arena.key", akey)) return 1;
    void* arena = nullptr;
    if (!Check(aclrtIpcMemImportByKey(&arena, akey, ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS), "import arena")) return 1;
    printf("[source] dev%d imported compute arena, launching role=0 M=%ld N=%ld K=%ld remote=[%ld,%ld) copyback=%d\n",
           srcDev, M, N, K, remoteStart, remoteStart + remoteCount, (int)enDCopyback);
    const std::vector<int64_t> shapeArena = {static_cast<int64_t>(arenaBytes / elem)};
    const std::vector<int64_t> shapeDsig = {static_cast<int64_t>(dsigBytes / elem)};
    aclTensor* tX1 = MakeTensor(x1s, shapeA, dtype);
    aclTensor* tX2 = MakeTensor(x2s, shapeB, dtype);
    aclTensor* tArena = MakeTensor(arena, shapeArena, dtype);
    aclTensor* tDsig = MakeTensor(dsig, shapeDsig, dtype);
    aclTensor* tOut = MakeTensor(cSrc, shapeC, dtype);
    if (!LaunchRank(tX1, tX2, tArena, /*peerOut=*/nullptr, tDsig, tOut, /*role=*/0, /*rankId=*/0, transX2, remoteStart,
                    remoteCount, wfM, wfN, commCores, commKTiles, enDCopyback, stream, &ws)) return 1;
    if (!Check(aclrtSynchronizeStreamWithTimeout(stream, 600000), "sync source")) return 1;
    // C is assembled (source waited in-kernel on D-done). Read back + dump.
    std::vector<uint8_t> hostC(cBytes, 0);
    if (!Check(aclrtMemcpy(hostC.data(), cBytes, cSrc, cBytes, ACL_MEMCPY_DEVICE_TO_HOST), "readback")) return 1;
    FILE* f = fopen("output/output_z.bin", "wb"); fwrite(hostC.data(), 1, cBytes, f); fclose(f);
    FILE* d = fopen((dir + "/source_done").c_str(), "w"); fprintf(d, "1"); fclose(d);
    printf("[source] done -> output/output_z.bin\n");
  } else {
    // Compute owns the arena; imports source C (peer_out) + dsig (peer_dsignal).
    void* arena = nullptr; if (!Check(aclrtMalloc(&arena, arenaBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc arena")) return 1;
    if (!Check(aclrtMemset(arena, arenaBytes, 0, arenaBytes), "zero arena")) return 1;
    memset(key, 0, kKeyLen);
    if (!Check(aclrtIpcMemGetExportKey(arena, arenaBytes, key, kKeyLen, ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION), "export arena")) return 1;
    WriteKey(dir + "/arena.key", key);
    void* x1c = DevFromHost(hostA.data(), aBytes);
    void* x2c = DevFromHost(hostB.data(), bBytes);
    void* outDummy = DevFromHost(nullptr, cBytes);
    char ckey[kKeyLen], dkey[kKeyLen];
    if (!ReadKey(dir + "/c.key", ckey)) return 1;
    if (!ReadKey(dir + "/dsig.key", dkey)) return 1;
    void* peerC = nullptr; void* peerDsig = nullptr;
    if (!Check(aclrtIpcMemImportByKey(&peerC, ckey, ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS), "import C")) return 1;
    if (!Check(aclrtIpcMemImportByKey(&peerDsig, dkey, ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS), "import dsig")) return 1;
    printf("[compute] dev%d owns arena, imported source C+dsig, launching role=1\n", cmpDev);
    const std::vector<int64_t> shapeArena = {static_cast<int64_t>(arenaBytes / elem)};
    const std::vector<int64_t> shapeDsig = {static_cast<int64_t>(dsigBytes / elem)};
    aclTensor* tX1 = MakeTensor(x1c, shapeA, dtype);
    aclTensor* tX2 = MakeTensor(x2c, shapeB, dtype);
    aclTensor* tArena = MakeTensor(arena, shapeArena, dtype);
    aclTensor* tPeerOut = MakeTensor(peerC, shapeC, dtype);
    aclTensor* tPeerDsig = MakeTensor(peerDsig, shapeDsig, dtype);
    aclTensor* tOut = MakeTensor(outDummy, shapeC, dtype);
    if (!LaunchRank(tX1, tX2, tArena, tPeerOut, tPeerDsig, tOut, /*role=*/1, /*rankId=*/1, transX2, remoteStart,
                    remoteCount, wfM, wfN, commCores, commKTiles, enDCopyback, stream, &ws)) return 1;
    if (!Check(aclrtSynchronizeStreamWithTimeout(stream, 600000), "sync compute")) return 1;
    printf("[compute] done\n");
    // Keep the arena alive until the source has read back C and imports are
    // released, so its imported peer mappings don't dangle during teardown.
    WaitFile(dir + "/source_done");
  }
  // NOTE: intentionally skip aclFinalize(). Tearing down while a peer process
  // still holds (or just released) an IPC-imported mapping crashes the ACL
  // runtime cleanup on this CANN version; the OS reclaims process resources on
  // exit. A real framework integration manages IPC lifetime explicitly instead.
  fflush(stdout);
  _exit(0);
}
