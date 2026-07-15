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
 * \file tilelet_host_layout.h
 * \brief Host-side, dependency-free replica of the tilelet_matmul_fp32 arena
 *        layout and the CUDA ratio->tile scheduling. Plain C++ (no ACL/GE deps)
 *        so both the two-card driver and a future framework layer can:
 *          (a) size the peer-visible arena to the op's exact `userWorkspaceBytes`
 *              instead of hardcoding 512 MB (doc task #2), and
 *          (b) map min/max_remote_cta_ratio -> remote_tile_start/count matching
 *              CUDA ClampRemoteCtasByRatio / ScheduleRemoteLinear (doc task #4).
 *
 * The layout math mirrors op_host/tilelet_matmul_fp32_tiling.cpp::InitTileletInfo
 * byte-for-byte (validated against the op's OP_LOGI "userWs" for several shapes).
 * The tiling here is deterministic because the op always takes the base path
 * (is_support_bl1 == false): singleCoreM = ceilAlign(min(M,128),16),
 * singleCoreN = ceilAlign(min(N,256),16), baseK = 128/sizeof(fp32) = 32.
 */

#ifndef TILELET_HOST_LAYOUT_H
#define TILELET_HOST_LAYOUT_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace tilelet {

// ---- constants mirrored from tilelet_matmul_fp32_tiling.cpp --------------------
constexpr uint64_t kBaseMCap = 128;      // BASIC_BLOCK_SIZE_128
constexpr uint64_t kBaseNCap = 256;      // BASIC_BLOCK_SIZE_256
constexpr uint64_t kAlign16 = 16;        // BASIC_ALIGN_16
constexpr uint64_t kAlign512 = 512;      // BASIC_ALIGN_512
constexpr uint64_t kSignalSlotBytes = 64;
constexpr uint64_t kBaseKFp32 = 128 / 4; // BASIC_BLOCK_K_128_BYTE / DATA_SIZE_FP32 = 32
constexpr uint64_t kDefaultWavefrontM = 16;
constexpr uint64_t kDefaultWavefrontN = 8;
constexpr uint64_t kDefaultCommCores = 8;
constexpr uint64_t kDefaultCommKTiles = 32;

inline uint64_t CeilDiv(uint64_t x, uint64_t align) { return align == 0 ? x : (x + align - 1) / align; }
inline uint64_t CeilAlign(uint64_t x, uint64_t align) { return CeilDiv(x, align) * align; }

struct ProblemParams {
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
    uint64_t elemSize = 2;              // bf16 = 2, fp32 = 4
    bool transposeX2 = true;
    bool enableDCopyback = false;
    uint64_t wavefrontM = kDefaultWavefrontM;
    uint64_t wavefrontN = kDefaultWavefrontN;
    uint64_t commCoreNum = kDefaultCommCores;
    uint64_t commKTiles = kDefaultCommKTiles;
    // AIC cores of the target die (910C die = 20). Only bounds commCoreNum when
    // the requested comm cores exceed it; for the benchmark (comm=8) it does not
    // change the layout. Over-estimating it is safe for arena sizing.
    uint64_t aicNum = 20;
};

struct Layout {
    uint64_t tileM = 0;
    uint64_t tileN = 0;
    uint64_t baseK = 0;
    uint64_t mTileCnt = 0;
    uint64_t nTileCnt = 0;
    uint64_t wavefrontRows = 0;
    uint64_t wavefrontCols = 0;
    uint64_t wavefrontCount = 0;
    uint64_t kGroupCount = 0;
    uint64_t compactTileSlots = 0;      // == total tilelet CTAs (padded space)
    uint64_t commCoreNum = 0;           // resolved (min of requested / computeCoreNum)
    uint64_t arenaBytes = 0;            // == op userWorkspaceBytes (staging+signals)
    uint64_t dSignalBytes = 0;          // source-side peer_dsignal buffer size (in-kernel D-done)
};

// Compute the full tilelet layout + arena size for a problem, assuming a remote
// range is active (the cross-card case always is). Mirrors InitTileletInfo.
inline Layout ComputeLayout(const ProblemParams& p) {
    Layout l;
    l.tileM = CeilAlign(std::min<uint64_t>(p.m, kBaseMCap), kAlign16);
    l.tileN = CeilAlign(std::min<uint64_t>(p.n, kBaseNCap), kAlign16);
    l.baseK = kBaseKFp32;
    if (l.tileM == 0 || l.tileN == 0) {
        return l;
    }
    const uint64_t wfM = p.wavefrontM == 0 ? kDefaultWavefrontM : p.wavefrontM;
    const uint64_t wfN = p.wavefrontN == 0 ? kDefaultWavefrontN : p.wavefrontN;
    l.mTileCnt = CeilDiv(p.m, l.tileM);
    l.nTileCnt = CeilDiv(p.n, l.tileN);
    l.wavefrontRows = CeilDiv(l.mTileCnt, wfM);
    l.wavefrontCols = CeilDiv(l.nTileCnt, wfN);
    l.wavefrontCount = l.wavefrontRows * l.wavefrontCols;
    const uint64_t abSignalCount = l.wavefrontRows + l.wavefrontCols - 1;
    const uint64_t commKTiles = p.commKTiles == 0 ? kDefaultCommKTiles : p.commKTiles;
    const uint64_t kTileCount = CeilDiv(p.k, std::max<uint64_t>(1, l.baseK));
    l.kGroupCount = std::max<uint64_t>(1, CeilDiv(kTileCount, commKTiles));
    const uint64_t abSignalGroupCount = abSignalCount * l.kGroupCount;
    l.compactTileSlots = l.wavefrontCount * wfM * wfN;

    // computeCoreNum = min(compactTileSlots, aicNum); commCoreNum = min(requested, computeCoreNum).
    const uint64_t computeCoreNum = std::max<uint64_t>(1, std::min(l.compactTileSlots, p.aicNum));
    const uint64_t requestedComm = p.commCoreNum == 0 ? kDefaultCommCores : p.commCoreNum;
    l.commCoreNum = std::min(requestedComm, computeCoreNum);

    const uint64_t wavefrontNStride = wfN * l.tileN;
    const uint64_t aSlabElements = wfM * l.tileM * p.k;
    uint64_t bSlabElements;
    if (p.transposeX2) {
        bSlabElements = wavefrontNStride * p.k;
    } else {
        const uint64_t bStageStride = p.enableDCopyback ? wavefrontNStride : p.n;
        bSlabElements = p.k * bStageStride;
    }
    const uint64_t dSlabElements = wfM * l.tileM * wavefrontNStride;

    const uint64_t abSignalSlots = abSignalGroupCount + 1 + abSignalGroupCount * l.commCoreNum;
    const uint64_t dSignalSlots = l.compactTileSlots + l.wavefrontCount * l.commCoreNum;
    l.dSignalBytes = CeilAlign(dSignalSlots * kSignalSlotBytes, kAlign512);
    const uint64_t abSignalByteOffset = 0;
    const uint64_t dSignalByteOffset = CeilAlign(abSignalByteOffset + abSignalSlots * kSignalSlotBytes, 64);
    const uint64_t aSlabByteOffset = CeilAlign(dSignalByteOffset + dSignalSlots * kSignalSlotBytes, 64);
    const uint64_t bSlabByteOffset =
        CeilAlign(aSlabByteOffset + l.wavefrontRows * aSlabElements * p.elemSize, 64);
    const uint64_t dSlabByteOffset =
        CeilAlign(bSlabByteOffset + l.wavefrontCols * bSlabElements * p.elemSize, 64);
    l.arenaBytes = CeilAlign(dSlabByteOffset + l.wavefrontCount * dSlabElements * p.elemSize, kAlign512);
    return l;
}

// Convenience: arena bytes only.
inline uint64_t ComputeArenaBytes(const ProblemParams& p) { return ComputeLayout(p).arenaBytes; }

// ---- ratio -> tile scheduling (mirrors CUDA torch_bindings.cpp) -----------------

// CUDA ClampRemoteCtasByRatio: clamp desired remote CTA count into
// [min(ceil(total*min), floor(total*max)), floor(total*max)].
inline int64_t ClampRemoteCtasByRatio(int64_t computeCtas, int64_t totalCtas, double minRatio, double maxRatio) {
    if (totalCtas <= 0) {
        return 0;
    }
    const int64_t total = std::max<int64_t>(0, totalCtas);
    const int64_t minComputeCtas =
        std::min<int64_t>(total, static_cast<int64_t>(std::ceil(static_cast<double>(total) * minRatio)));
    const int64_t maxComputeCtas =
        std::min<int64_t>(total, static_cast<int64_t>(std::floor(static_cast<double>(total) * maxRatio)));
    const int64_t lowerBound = std::min(minComputeCtas, maxComputeCtas);
    return std::max(lowerBound, std::min(std::max<int64_t>(0, computeCtas), maxComputeCtas));
}

struct RemoteRange {
    int64_t remoteTileStart = 0;
    int64_t remoteTileCount = 0;
    int64_t totalWavefronts = 0;
    int64_t wavefrontCtas = 0;
    int64_t computeWavefronts = 0;
};

// Mirror of ScheduleRemoteLinear: balance remote vs local wavefronts by the two
// cards' SM (AIC) counts, convert to CTAs, clamp by [min,max] ratio. Remote tiles
// are a prefix of the compact-slot linearization (remote_tile_start = 0), matching
// the Ascend gate convention and the benchmark (offload a prefix of wavefronts).
//
// sourceSm/computeSm are the AIC cores available to the source and compute ranks
// (compute side already net of comm cores, as CUDA does). commSms is informational.
inline RemoteRange MapRatioToRemoteTiles(const ProblemParams& p, double minRatio, double maxRatio,
                                         uint64_t sourceSm, uint64_t computeSm) {
    const Layout l = ComputeLayout(p);
    RemoteRange r;
    r.totalWavefronts = static_cast<int64_t>(l.wavefrontCount);
    const uint64_t wfM = p.wavefrontM == 0 ? kDefaultWavefrontM : p.wavefrontM;
    const uint64_t wfN = p.wavefrontN == 0 ? kDefaultWavefrontN : p.wavefrontN;
    r.wavefrontCtas = static_cast<int64_t>(wfM * wfN);
    const int64_t totalCtas = static_cast<int64_t>(l.compactTileSlots);

    if (r.totalWavefronts <= 1 || r.wavefrontCtas <= 0 || computeSm == 0 || sourceSm == 0) {
        r.computeWavefronts = 0;
        r.remoteTileCount = ClampRemoteCtasByRatio(0, totalCtas, minRatio, maxRatio);
        r.remoteTileStart = 0;
        return r;
    }

    double bestScore = std::numeric_limits<double>::infinity();
    int64_t bestComputeWavefronts = 0;
    for (int64_t remoteWavefronts = 1; remoteWavefronts < r.totalWavefronts; ++remoteWavefronts) {
        const int64_t localWavefronts = r.totalWavefronts - remoteWavefronts;
        const double remoteWork = static_cast<double>(remoteWavefronts) / static_cast<double>(computeSm);
        const double localWork = static_cast<double>(localWavefronts) / static_cast<double>(sourceSm);
        const double score = std::abs(remoteWork - localWork);
        if (score < bestScore) {
            bestScore = score;
            bestComputeWavefronts = remoteWavefronts;
        }
    }
    r.computeWavefronts = bestComputeWavefronts;
    const int64_t unclampedComputeCtas = std::min<int64_t>(totalCtas, r.computeWavefronts * r.wavefrontCtas);
    r.remoteTileCount = ClampRemoteCtasByRatio(unclampedComputeCtas, totalCtas, minRatio, maxRatio);
    r.remoteTileStart = 0;
    return r;
}

}  // namespace tilelet

#endif  // TILELET_HOST_LAYOUT_H
