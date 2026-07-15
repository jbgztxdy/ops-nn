/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file tilelet_matmul_fp32_base_kernel.h
 * \brief
 */
#ifndef __TILELET_MATMUL_FP32_BASE_KERNEL_H__
#define __TILELET_MATMUL_FP32_BASE_KERNEL_H__

#include "tilelet_matmul_fp32_base_block.h"

using namespace AscendC;
using namespace matmul;

constexpr uint32_t TILELET_MATMUL_FP32_COPY_UB_BYTES = 16 * 1024;
constexpr uint64_t TILELET_MATMUL_FP32_SIGNAL_WORD_STRIDE = 64 / sizeof(uint32_t);

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = TileletMatmulFp32BaseBlock,
          const MatmulConfig& MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>>
class TileletMatmulFp32BaseKernel {
public:
    __aicore__ inline TileletMatmulFp32BaseKernel(){};

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM,
                                GM_ADDR arenaGM, GM_ADDR peerOutGM, const void* tilingData, TPipe* pipe);

    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM,
                                      GM_ADDR arenaGM, GM_ADDR peerOutGM);
    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);
    // Cross-card ("peer 直存") drivers: source stages A/B into the peer arena and
    // computes local direct tiles; compute reads the arena and writes remote
    // tiles straight into the source card's C over the link.
    __aicore__ inline void ProcessCrossCardSource(uint8_t enAtomic);
    __aicore__ inline void ProcessCrossCardCompute();

private:
    __aicore__ inline uint64_t Min(uint64_t a, uint64_t b) const;
    __aicore__ inline bool DecodeCompactSlot(uint64_t slot, uint64_t& wfIndex, uint64_t& localM, uint64_t& localN,
                                             uint64_t& mTile, uint64_t& nTile) const;
    __aicore__ inline bool IsRemoteSlot(uint64_t slot) const;
    __aicore__ inline bool WavefrontHasRemoteTile(uint64_t wfIndex) const;
    __aicore__ inline bool IsDirectDWavefront(uint64_t wfIndex) const;
    __aicore__ inline uint64_t SignalCount() const;
    __aicore__ inline uint64_t AbSignalCount() const;
    __aicore__ inline uint64_t AbSignalIndex(uint64_t sigIndex, uint64_t kGroup) const;
    __aicore__ inline uint64_t InitSignalIndex() const;
    __aicore__ inline uint64_t AbCompletionIndex(uint64_t sigIndex, uint64_t kGroup, uint64_t coreIdx) const;
    __aicore__ inline uint64_t DCopyDoneIndex(uint64_t wfIndex, uint64_t coreIdx) const;
    __aicore__ inline uint64_t SignalForA(uint64_t wfRow) const;
    __aicore__ inline uint64_t SignalForB(uint64_t wfCol) const;
    __aicore__ inline void SignalOrigin(uint64_t sigIndex, uint64_t& wfRow, uint64_t& wfCol) const;
    __aicore__ inline bool SignalNeededForRemoteTiles(uint64_t sigIndex) const;
    __aicore__ inline void WaitAbSignalForWavefront(uint64_t wfIndex);
    __aicore__ inline void WaitRemoteWavefrontDone(uint64_t wfIndex);
    __aicore__ inline void WaitCopyBackWavefrontDone(uint64_t wfIndex);
    __aicore__ inline void SignalBarrier();
    __aicore__ inline uint32_t ReadSignal(GlobalTensor<uint32_t>& signal, uint64_t index);
    __aicore__ inline void WriteSignal(GlobalTensor<uint32_t>& signal, uint64_t index, uint32_t value);
    __aicore__ inline void ResetSignals();
    __aicore__ inline void ProcessCommCore(uint64_t coreIdx);
    __aicore__ inline void ProcessCommCopyIn(uint64_t coreIdx);
    __aicore__ inline void ProcessCommCopyBack(uint64_t coreIdx);
    __aicore__ inline void ProcessSerialTilelet(uint8_t enAtomic);
    __aicore__ inline void ProcessComputeCore(uint64_t computeIdx, uint8_t enAtomic);
    __aicore__ inline void ProcessComputeCoreDirect(uint64_t computeIdx, uint8_t enAtomic);
    __aicore__ inline void ProcessComputeCoreRemote(uint64_t computeIdx);
    __aicore__ inline void CopySignalAB(uint64_t sigIndex, uint64_t kGroup, uint64_t coreIdx);
    __aicore__ inline void ComputeDirectTile(uint64_t mTile, uint64_t nTile, uint8_t enAtomic);
    __aicore__ inline void ComputeRemoteTile(uint64_t wfIndex, uint64_t localM, uint64_t localN, uint64_t mTile,
                                             uint64_t nTile);
    __aicore__ inline void ComputeRemoteTileDirectD(uint64_t wfIndex, uint64_t localM, uint64_t localN,
                                                    uint64_t mTile, uint64_t nTile);
    __aicore__ inline void CopyBackRemoteTile(uint64_t wfIndex, uint64_t localM, uint64_t localN, uint64_t mTile,
                                              uint64_t nTile);
    __aicore__ inline void CopyBackRemoteWavefront(uint64_t wfIndex, uint64_t coreIdx);
    template <typename T>
    __aicore__ inline void CopyGmRange(GlobalTensor<T>& dst, uint64_t dstOffset, GlobalTensor<T>& src,
                                       uint64_t srcOffset, uint64_t elemCount);
    template <typename T>
    __aicore__ inline void CopyGmRangePart(GlobalTensor<T>& dst, uint64_t dstOffset, GlobalTensor<T>& src,
                                           uint64_t srcOffset, uint64_t elemCount, uint64_t worker, uint64_t workers);
    template <typename T>
    __aicore__ inline void CopyGm2D(GlobalTensor<T>& dst, uint64_t dstOffset, uint64_t dstRowStride,
                                    GlobalTensor<T>& src, uint64_t srcOffset, uint64_t srcRowStride,
                                    uint64_t rows, uint64_t rowElems);
    template <typename T>
    __aicore__ inline void CopyGm2DPart(GlobalTensor<T>& dst, uint64_t dstOffset, uint64_t dstRowStride,
                                        GlobalTensor<T>& src, uint64_t srcOffset, uint64_t srcRowStride,
                                        uint64_t rows, uint64_t rowElems, uint64_t worker, uint64_t workers);

protected:
    BLOCK_TYPE block_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB> mm_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    GlobalTensor<A_T> aStageGlobal_;
    GlobalTensor<B_T> bStageGlobal_;
    GlobalTensor<C_T> dStageGlobal_;
    GlobalTensor<uint32_t> abSignalGlobal_;
    GlobalTensor<uint32_t> dSignalGlobal_;
    TBuf<QuePosition::VECCALC> copyBuf_;
    TPipe* pipe_;
    bool crossCard_ = false;  // arena provided => cooperative two-card execution
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM, GM_ADDR arenaGM, GM_ADDR peerOutGM,
    const void* tilingData, TPipe* pipe)
{
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    pipe_ = pipe;
    InitInputs(aGM, bGM, cGM, biasGM, workspaceGM, arenaGM, peerOutGM);
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const uint64_t coreIdx = GetCurrentBlockIdx();
    if ASCEND_IS_AIV {
        if (tilelet.remoteTileEnd > tilelet.remoteTileStart && tilelet.commCoreNum > 0 &&
            coreIdx < tilelet.commCoreNum && GetSubBlockIdx() == 0) {
            pipe_->InitBuffer(copyBuf_, TILELET_MATMUL_FP32_COPY_UB_BYTES);
        }
    }
    if ASCEND_IS_AIC {
        mm_.SetSubBlockIdx(0);
        mm_.Init(&block_.matmulFp32TilingData_->tCubeTiling, pipe_);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::InitInputs(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM, GM_ADDR arenaGM, GM_ADDR peerOutGM)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    crossCard_ = (arenaGM != nullptr);
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(aGM),
                             static_cast<uint64_t>(block_.matmulFp32TilingData_->tCubeTiling.M) *
                                 block_.matmulFp32TilingData_->tCubeTiling.Ka);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(bGM),
                             static_cast<uint64_t>(block_.matmulFp32TilingData_->tCubeTiling.Kb) *
                                 block_.matmulFp32TilingData_->tCubeTiling.N);
    // Compute rank writes its remote tiles straight into the source card's C via
    // peer_out; source (and single-card) writes into its own local C.
    __gm__ C_T* outAddr = reinterpret_cast<__gm__ C_T*>(cGM);
    if (crossCard_ && tilelet.role == 1 && peerOutGM != nullptr) {
        outAddr = reinterpret_cast<__gm__ C_T*>(peerOutGM);
    }
    cGlobal_.SetGlobalBuffer(outAddr, static_cast<uint64_t>(block_.matmulFp32TilingData_->tCubeTiling.M) *
                                          block_.matmulFp32TilingData_->tCubeTiling.N);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(biasGM), block_.matmulFp32TilingData_->tCubeTiling.N);
    // Signals and A/B/D staging live in the peer-visible arena when running
    // cross-card, otherwise in this rank's own workspace. Offsets are identical
    // either way, so both ranks agree byte-for-byte on the layout.
    __gm__ uint8_t* userWorkspace =
        crossCard_ ? reinterpret_cast<__gm__ uint8_t*>(arenaGM) : reinterpret_cast<__gm__ uint8_t*>(workspaceGM);
    const uint64_t abSignalCount = AbSignalCount();
    const uint64_t abSignalSlots = abSignalCount + 1 + abSignalCount * tilelet.commCoreNum;
    const uint64_t dSignalSlots = tilelet.compactTileSlots +
                                  static_cast<uint64_t>(tilelet.wavefrontCount) * tilelet.commCoreNum;
    abSignalGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(userWorkspace + tilelet.abSignalByteOffset),
                                    abSignalSlots * TILELET_MATMUL_FP32_SIGNAL_WORD_STRIDE);
    dSignalGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(userWorkspace + tilelet.dSignalByteOffset),
                                   dSignalSlots * TILELET_MATMUL_FP32_SIGNAL_WORD_STRIDE);
    aStageGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(userWorkspace + tilelet.aSlabByteOffset),
                                  tilelet.wavefrontRows * tilelet.aSlabElements);
    bStageGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(userWorkspace + tilelet.bSlabByteOffset),
                                  tilelet.wavefrontCols * tilelet.bSlabElements);
    dStageGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(userWorkspace + tilelet.dSlabByteOffset),
                                  tilelet.wavefrontCount * tilelet.dSlabElements);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Process(
    uint64_t index, uint8_t enAtomic)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const uint64_t coreIdx = GetCurrentBlockIdx();
    if (coreIdx >= tilelet.totalCoreNum) {
        return;
    }
    if (crossCard_) {
        if (tilelet.role == 1) {
            ProcessCrossCardCompute();
        } else {
            ProcessCrossCardSource(enAtomic);
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }
    const bool remoteActive = tilelet.remoteTileEnd > tilelet.remoteTileStart;
    if ASCEND_IS_AIV {
        if (!remoteActive) {
            return;
        }

        if (tilelet.commCoreNum == 0) {
            return;
        }

        if (coreIdx == 0 && GetSubBlockIdx() == 0) {
            ResetSignals();
        }

        if (GetSubBlockIdx() != 0) {
            return;
        }
        while (ReadSignal(abSignalGlobal_, InitSignalIndex()) == 0) {
        }

        const bool isCommCore = coreIdx < tilelet.commCoreNum;
        const uint64_t localCoreIdx = isCommCore ? coreIdx : 0;
        if (isCommCore) {
            ProcessCommCore(localCoreIdx);
        }

        PipeBarrier<PIPE_ALL>();
        return;
    }

    if (tilelet.commCoreNum == 0) {
        if (tilelet.remoteTileEnd > tilelet.remoteTileStart) {
            ProcessSerialTilelet(enAtomic);
        } else {
            ProcessComputeCore(coreIdx, enAtomic);
        }
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();
        return;
    }

    const bool isComputeCore = coreIdx < tilelet.computeCoreNum;
    const uint64_t localCoreIdx = isComputeCore ? coreIdx : 0;
    if (remoteActive) {
        while (ReadSignal(abSignalGlobal_, InitSignalIndex()) == 0) {
        }
    }

    if (isComputeCore) {
        ProcessComputeCoreDirect(localCoreIdx, enAtomic);
    }
    PipeBarrier<PIPE_ALL>();

    if (isComputeCore) {
        ProcessComputeCoreRemote(localCoreIdx);
    }
    PipeBarrier<PIPE_ALL>();

    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    return;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline uint64_t
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Min(uint64_t a,
                                                                                              uint64_t b) const
{
    return a < b ? a : b;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline bool TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::DecodeCompactSlot(uint64_t slot, uint64_t& wfIndex,
                                                                             uint64_t& localM, uint64_t& localN,
                                                                             uint64_t& mTile, uint64_t& nTile) const
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const uint64_t slotsPerWavefront = static_cast<uint64_t>(tilelet.wavefrontM) * tilelet.wavefrontN;
    wfIndex = slot / slotsPerWavefront;
    const uint64_t slotInWavefront = slot - wfIndex * slotsPerWavefront;
    const uint64_t wfRow = wfIndex / tilelet.wavefrontCols;
    const uint64_t wfCol = wfIndex - wfRow * tilelet.wavefrontCols;
    const uint64_t groupM = tilelet.wavefrontM < 2 ? tilelet.wavefrontM : 2;
    const uint64_t groupN = tilelet.wavefrontN < 2 ? tilelet.wavefrontN : 2;
    if (groupM == 0 || groupN == 0) {
        localM = slotInWavefront;
        localN = 0;
    } else if (tilelet.wavefrontM % groupM == 0 && tilelet.wavefrontN % groupN == 0) {
        const uint64_t groupCols = tilelet.wavefrontN / groupN;
        const uint64_t groupSize = groupM * groupN;
        const uint64_t group = slotInWavefront / groupSize;
        const uint64_t local = slotInWavefront - group * groupSize;
        const uint64_t groupRow = group / groupCols;
        const uint64_t groupCol = group - groupRow * groupCols;
        const uint64_t innerN = local / groupM;
        const uint64_t innerM = local - innerN * groupM;
        localM = groupRow * groupM + innerM;
        localN = groupCol * groupN + innerN;
    } else {
        uint64_t remaining = slotInWavefront;
        localM = tilelet.wavefrontM;
        localN = 0;
        for (uint64_t groupMBase = 0; groupMBase < tilelet.wavefrontM; groupMBase += groupM) {
            const uint64_t currentGroupM = groupMBase + groupM <= tilelet.wavefrontM ?
                                               groupM :
                                               (tilelet.wavefrontM - groupMBase);
            for (uint64_t groupNBase = 0; groupNBase < tilelet.wavefrontN; groupNBase += groupN) {
                const uint64_t currentGroupN = groupNBase + groupN <= tilelet.wavefrontN ?
                                                   groupN :
                                                   (tilelet.wavefrontN - groupNBase);
                const uint64_t groupSize = currentGroupM * currentGroupN;
                if (remaining < groupSize) {
                    const uint64_t innerN = remaining / currentGroupM;
                    const uint64_t innerM = remaining - innerN * currentGroupM;
                    localM = groupMBase + innerM;
                    localN = groupNBase + innerN;
                    groupMBase = tilelet.wavefrontM;
                    break;
                }
                remaining -= groupSize;
            }
        }
    }
    mTile = wfRow * tilelet.wavefrontM + localM;
    nTile = wfCol * tilelet.wavefrontN + localN;
    return mTile < tilelet.mTileCnt && nTile < tilelet.nTileCnt;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline bool
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::IsRemoteSlot(uint64_t slot) const
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    return slot >= tilelet.remoteTileStart && slot < tilelet.remoteTileEnd;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline bool TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::WavefrontHasRemoteTile(uint64_t wfIndex) const
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const uint64_t slotsPerWavefront = static_cast<uint64_t>(tilelet.wavefrontM) * tilelet.wavefrontN;
    const uint64_t wfStart = wfIndex * slotsPerWavefront;
    const uint64_t wfEnd = wfStart + slotsPerWavefront;
    return wfEnd > tilelet.remoteTileStart && wfStart < tilelet.remoteTileEnd;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline bool
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::IsDirectDWavefront(
    uint64_t wfIndex) const
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    if (tilelet.commCoreNum == 0 || tilelet.remoteTileEnd <= tilelet.remoteTileStart) {
        return false;
    }
    if (tilelet.enableDCopyback == 0) {
        return true;
    }
    const uint64_t slotsPerWavefront = static_cast<uint64_t>(tilelet.wavefrontM) * tilelet.wavefrontN;
    return wfIndex == (static_cast<uint64_t>(tilelet.remoteTileEnd) - 1) / slotsPerWavefront;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline uint64_t
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::SignalCount() const
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    return static_cast<uint64_t>(tilelet.wavefrontRows) + tilelet.wavefrontCols - 1;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline uint64_t
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::AbSignalCount() const
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const uint64_t kGroupCount = tilelet.kGroupCount == 0 ? 1 : static_cast<uint64_t>(tilelet.kGroupCount);
    return SignalCount() * kGroupCount;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline uint64_t
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::AbSignalIndex(
    uint64_t sigIndex, uint64_t kGroup) const
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const uint64_t kGroupCount = tilelet.kGroupCount == 0 ? 1 : static_cast<uint64_t>(tilelet.kGroupCount);
    return sigIndex * kGroupCount + kGroup;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline uint64_t
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::InitSignalIndex() const
{
    return AbSignalCount();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline uint64_t TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                       MM_CB>::AbCompletionIndex(uint64_t sigIndex, uint64_t kGroup,
                                                                                 uint64_t coreIdx) const
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    return AbSignalCount() + 1 + AbSignalIndex(sigIndex, kGroup) * tilelet.commCoreNum + coreIdx;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline uint64_t TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                       MM_CB>::DCopyDoneIndex(uint64_t wfIndex,
                                                                              uint64_t coreIdx) const
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    return tilelet.compactTileSlots + wfIndex * tilelet.commCoreNum + coreIdx;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline uint64_t
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::SignalForA(
    uint64_t wfRow) const
{
    return wfRow == 0 ? 0 : wfRow;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline uint64_t
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::SignalForB(
    uint64_t wfCol) const
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    return wfCol == 0 ? 0 : static_cast<uint64_t>(tilelet.wavefrontRows) - 1 + wfCol;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::SignalOrigin(
    uint64_t sigIndex, uint64_t& wfRow, uint64_t& wfCol) const
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    if (sigIndex < tilelet.wavefrontRows) {
        wfRow = sigIndex;
        wfCol = 0;
    } else {
        wfRow = 0;
        wfCol = sigIndex - tilelet.wavefrontRows + 1;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline bool
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::SignalNeededForRemoteTiles(
    uint64_t sigIndex) const
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    for (uint64_t slot = tilelet.remoteTileStart; slot < tilelet.remoteTileEnd; ++slot) {
        uint64_t wfIndex = 0;
        uint64_t localM = 0;
        uint64_t localN = 0;
        uint64_t mTile = 0;
        uint64_t nTile = 0;
        if (!DecodeCompactSlot(slot, wfIndex, localM, localN, mTile, nTile)) {
            continue;
        }
        const uint64_t wfRow = wfIndex / tilelet.wavefrontCols;
        const uint64_t wfCol = wfIndex - wfRow * tilelet.wavefrontCols;
        if (sigIndex == SignalForA(wfRow) || sigIndex == SignalForB(wfCol)) {
            return true;
        }
    }
    return false;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::WaitAbSignalForWavefront(
    uint64_t wfIndex)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const uint64_t wfRow = wfIndex / tilelet.wavefrontCols;
    const uint64_t wfCol = wfIndex - wfRow * tilelet.wavefrontCols;
    const uint64_t sigA = SignalForA(wfRow);
    const uint64_t sigB = SignalForB(wfCol);
    const uint64_t waitSig = sigA > sigB ? sigA : sigB;
    const uint64_t kGroupCount = tilelet.kGroupCount == 0 ? 1 : static_cast<uint64_t>(tilelet.kGroupCount);
    for (uint64_t group = 0; group < kGroupCount; ++group) {
        while (ReadSignal(abSignalGlobal_, AbSignalIndex(waitSig, group)) == 0) {
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::WaitRemoteWavefrontDone(
    uint64_t wfIndex)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const uint64_t slotsPerWavefront = static_cast<uint64_t>(tilelet.wavefrontM) * tilelet.wavefrontN;
    const uint64_t wfStart = wfIndex * slotsPerWavefront;
    const uint64_t wfEnd = Min(wfStart + slotsPerWavefront, static_cast<uint64_t>(tilelet.compactTileSlots));
    for (uint64_t slot = wfStart; slot < wfEnd; ++slot) {
        uint64_t slotWf = 0;
        uint64_t localM = 0;
        uint64_t localN = 0;
        uint64_t mTile = 0;
        uint64_t nTile = 0;
        if (!IsRemoteSlot(slot) || !DecodeCompactSlot(slot, slotWf, localM, localN, mTile, nTile)) {
            continue;
        }
        while (ReadSignal(dSignalGlobal_, slot) == 0) {
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::WaitCopyBackWavefrontDone(
    uint64_t wfIndex)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    for (uint64_t core = 0; core < tilelet.commCoreNum; ++core) {
        while (ReadSignal(dSignalGlobal_, DCopyDoneIndex(wfIndex, core)) == 0) {
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::SignalBarrier()
{
    __asm__ __volatile__("" ::: "memory");
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline uint32_t
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::ReadSignal(
    GlobalTensor<uint32_t>& signal, uint64_t index)
{
    const uint64_t offset = index * TILELET_MATMUL_FP32_SIGNAL_WORD_STRIDE;
    SignalBarrier();
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(signal[offset]);
    SignalBarrier();
    return signal.GetValue(offset);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::WriteSignal(
    GlobalTensor<uint32_t>& signal, uint64_t index, uint32_t value)
{
    const uint64_t offset = index * TILELET_MATMUL_FP32_SIGNAL_WORD_STRIDE;
    signal.SetValue(offset, value);
    SignalBarrier();
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(signal[offset]);
    SignalBarrier();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::ResetSignals()
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const uint64_t abSignalCount = SignalCount();
    const uint64_t kGroupCount = tilelet.kGroupCount == 0 ? 1 : static_cast<uint64_t>(tilelet.kGroupCount);
    WriteSignal(abSignalGlobal_, InitSignalIndex(), 0);
    for (uint64_t sig = 0; sig < abSignalCount; ++sig) {
        for (uint64_t group = 0; group < kGroupCount; ++group) {
            WriteSignal(abSignalGlobal_, AbSignalIndex(sig, group), 0);
            for (uint64_t core = 0; core < tilelet.commCoreNum; ++core) {
                WriteSignal(abSignalGlobal_, AbCompletionIndex(sig, group, core), 0);
            }
        }
    }
    for (uint64_t slot = 0; slot < tilelet.compactTileSlots; ++slot) {
        WriteSignal(dSignalGlobal_, slot, 0);
    }
    for (uint64_t wf = 0; wf < tilelet.wavefrontCount; ++wf) {
        for (uint64_t core = 0; core < tilelet.commCoreNum; ++core) {
            WriteSignal(dSignalGlobal_, DCopyDoneIndex(wf, core), 0);
        }
    }
    WriteSignal(abSignalGlobal_, InitSignalIndex(), 1);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::ProcessCommCore(uint64_t coreIdx)
{
    ProcessCommCopyIn(coreIdx);
    ProcessCommCopyBack(coreIdx);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::ProcessCommCopyIn(uint64_t coreIdx)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const uint64_t abSignalCount = SignalCount();
    const uint64_t kGroupCount = tilelet.kGroupCount == 0 ? 1 : static_cast<uint64_t>(tilelet.kGroupCount);
    for (uint64_t sig = 0; sig < abSignalCount; ++sig) {
        if (!SignalNeededForRemoteTiles(sig)) {
            continue;
        }
        for (uint64_t group = 0; group < kGroupCount; ++group) {
            CopySignalAB(sig, group, coreIdx);
            PipeBarrier<PIPE_ALL>();
            WriteSignal(abSignalGlobal_, AbCompletionIndex(sig, group, coreIdx), 1);
            if (coreIdx == 0) {
                for (uint64_t core = 0; core < tilelet.commCoreNum; ++core) {
                    while (ReadSignal(abSignalGlobal_, AbCompletionIndex(sig, group, core)) == 0) {
                    }
                }
                WriteSignal(abSignalGlobal_, AbSignalIndex(sig, group), 1);
            }
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::ProcessCommCopyBack(uint64_t coreIdx)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    if (tilelet.enableDCopyback == 0) {
        return;
    }
    for (uint64_t wf = 0; wf < tilelet.wavefrontCount; ++wf) {
        if (!WavefrontHasRemoteTile(wf) || IsDirectDWavefront(wf)) {
            continue;
        }
        WaitRemoteWavefrontDone(wf);
        CopyBackRemoteWavefront(wf, coreIdx);
        PipeBarrier<PIPE_ALL>();
        WriteSignal(dSignalGlobal_, DCopyDoneIndex(wf, coreIdx), 1);
        WaitCopyBackWavefrontDone(wf);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::ProcessCrossCardSource(uint8_t enAtomic)
{
    // Source rank: AIV comm cores stage this problem's remote-tile A/B into the
    // peer arena (and raise AB signals across the link); AIC cores compute the
    // local direct tiles into the source card's own C. The arena is zeroed by
    // the host before launch, so no in-kernel signal reset is needed.
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const uint64_t coreIdx = GetCurrentBlockIdx();
    if ASCEND_IS_AIV {
        if (tilelet.commCoreNum == 0 || tilelet.remoteTileEnd <= tilelet.remoteTileStart) {
            return;
        }
        if (GetSubBlockIdx() != 0 || coreIdx >= tilelet.commCoreNum) {
            return;
        }
        ProcessCommCopyIn(coreIdx);
        return;
    }
    if ASCEND_IS_AIC {
        ProcessComputeCoreDirect(coreIdx, enAtomic);
        SetAtomicNone();
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::ProcessCrossCardCompute()
{
    // Compute rank: AIC cores wait for each remote wavefront's A/B to arrive in
    // the arena (AB signals raised by the source), read the staged A/B locally,
    // and write the remote tiles directly into the source card's C (cGlobal_ was
    // retargeted to peer_out). Host-level stream sync of both ranks gates
    // completion, so no D-done signalling is required here.
    if ASCEND_IS_AIV {
        return;
    }
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const uint64_t computeIdx = GetCurrentBlockIdx();
    for (uint64_t slot = computeIdx; slot < tilelet.compactTileSlots; slot += tilelet.computeCoreNum) {
        uint64_t wfIndex = 0;
        uint64_t localM = 0;
        uint64_t localN = 0;
        uint64_t mTile = 0;
        uint64_t nTile = 0;
        if (!DecodeCompactSlot(slot, wfIndex, localM, localN, mTile, nTile) || !IsRemoteSlot(slot)) {
            continue;
        }
        WaitAbSignalForWavefront(wfIndex);
        ComputeRemoteTileDirectD(wfIndex, localM, localN, mTile, nTile);
    }
    SetAtomicNone();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::ProcessSerialTilelet(uint8_t enAtomic)
{
    ProcessComputeCoreDirect(0, enAtomic);
    PipeBarrier<PIPE_ALL>();
    ProcessComputeCoreRemote(0);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::ProcessComputeCore(uint64_t computeIdx, uint8_t enAtomic)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    for (uint64_t slot = computeIdx; slot < tilelet.compactTileSlots; slot += tilelet.computeCoreNum) {
        uint64_t wfIndex = 0;
        uint64_t localM = 0;
        uint64_t localN = 0;
        uint64_t mTile = 0;
        uint64_t nTile = 0;
        if (!DecodeCompactSlot(slot, wfIndex, localM, localN, mTile, nTile)) {
            continue;
        }
        if (IsRemoteSlot(slot)) {
            WaitAbSignalForWavefront(wfIndex);
            if (IsDirectDWavefront(wfIndex)) {
                ComputeRemoteTileDirectD(wfIndex, localM, localN, mTile, nTile);
            } else {
                ComputeRemoteTile(wfIndex, localM, localN, mTile, nTile);
                PipeBarrier<PIPE_ALL>();
                WriteSignal(dSignalGlobal_, slot, 1);
            }
        } else {
            ComputeDirectTile(mTile, nTile, enAtomic);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::ProcessComputeCoreDirect(uint64_t computeIdx,
                                                                                    uint8_t enAtomic)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    for (uint64_t slot = computeIdx; slot < tilelet.compactTileSlots; slot += tilelet.computeCoreNum) {
        uint64_t wfIndex = 0;
        uint64_t localM = 0;
        uint64_t localN = 0;
        uint64_t mTile = 0;
        uint64_t nTile = 0;
        if (!DecodeCompactSlot(slot, wfIndex, localM, localN, mTile, nTile) || IsRemoteSlot(slot)) {
            continue;
        }
        ComputeDirectTile(mTile, nTile, enAtomic);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::ProcessComputeCoreRemote(uint64_t computeIdx)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    for (uint64_t slot = computeIdx; slot < tilelet.compactTileSlots; slot += tilelet.computeCoreNum) {
        uint64_t wfIndex = 0;
        uint64_t localM = 0;
        uint64_t localN = 0;
        uint64_t mTile = 0;
        uint64_t nTile = 0;
        if (!DecodeCompactSlot(slot, wfIndex, localM, localN, mTile, nTile) || !IsRemoteSlot(slot)) {
            continue;
        }
        if (tilelet.commCoreNum == 0) {
            ComputeDirectTile(mTile, nTile, 0);
            continue;
        }
        WaitAbSignalForWavefront(wfIndex);
        if (IsDirectDWavefront(wfIndex)) {
            ComputeRemoteTileDirectD(wfIndex, localM, localN, mTile, nTile);
        } else {
            ComputeRemoteTile(wfIndex, localM, localN, mTile, nTile);
            PipeBarrier<PIPE_ALL>();
            WriteSignal(dSignalGlobal_, slot, 1);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void
TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::CopySignalAB(
    uint64_t sigIndex, uint64_t kGroup, uint64_t coreIdx)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const TCubeTiling& tiling = block_.matmulFp32TilingData_->tCubeTiling;
    const uint64_t tileM = tiling.singleCoreM;
    const uint64_t tileN = tiling.singleCoreN;
    const bool transB = block_.matmulFp32TilingData_->matmulFp32RunInfo.transB != 0;
    const uint64_t baseK = tiling.baseK == 0 ? 1 : static_cast<uint64_t>(tiling.baseK);
    const uint64_t commKTiles = tilelet.commKTiles == 0 ? 32 : static_cast<uint64_t>(tilelet.commKTiles);
    const uint64_t kTileCount = (static_cast<uint64_t>(tiling.Ka) + baseK - 1) / baseK;
    const uint64_t kTileBegin = kGroup * commKTiles;
    const uint64_t kTileEnd = Min(kTileBegin + commKTiles, kTileCount);
    const uint64_t kBegin = Min(kTileBegin * baseK, static_cast<uint64_t>(tiling.Ka));
    const uint64_t kEnd = Min(kTileEnd * baseK, static_cast<uint64_t>(tiling.Ka));
    const uint64_t kSpan = kEnd > kBegin ? kEnd - kBegin : 0;
    uint64_t wfRow = 0;
    uint64_t wfCol = 0;
    SignalOrigin(sigIndex, wfRow, wfCol);
    const uint64_t mStart = wfRow * tilelet.wavefrontM * tileM;
    const uint64_t nStart = wfCol * tilelet.wavefrontN * tileN;
    const uint64_t mSpan = mStart >= tiling.M ? 0 : Min(static_cast<uint64_t>(tiling.M) - mStart,
                                                        static_cast<uint64_t>(tilelet.wavefrontM) * tileM);
    const uint64_t nSpan = nStart >= tiling.N ? 0 : Min(static_cast<uint64_t>(tiling.N) - nStart,
                                                        static_cast<uint64_t>(tilelet.wavefrontN) * tileN);
    const uint64_t wfNStride = static_cast<uint64_t>(tilelet.wavefrontN) * tileN;
    const uint64_t bStageStride = tilelet.enableDCopyback == 0 ? static_cast<uint64_t>(tiling.N) : wfNStride;
    if (wfCol == 0) {
        const uint64_t aBase = wfRow * tilelet.aSlabElements;
        CopyGm2DPart(aStageGlobal_, aBase + kBegin, tiling.Ka, aGlobal_, mStart * tiling.Ka + kBegin, tiling.Ka,
                     mSpan, kSpan, coreIdx, tilelet.commCoreNum);
    }
    if (wfRow == 0) {
        const uint64_t bBase = wfCol * tilelet.bSlabElements;
        if (transB) {
            CopyGm2DPart(bStageGlobal_, bBase + kBegin, tiling.Kb, bGlobal_, nStart * tiling.Kb + kBegin,
                         tiling.Kb, nSpan, kSpan, coreIdx, tilelet.commCoreNum);
        } else {
            CopyGm2DPart(bStageGlobal_, bBase + kBegin * bStageStride, bStageStride, bGlobal_,
                         kBegin * tiling.N + nStart, tiling.N, kSpan, nSpan, coreIdx, tilelet.commCoreNum);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::ComputeDirectTile(uint64_t mTile, uint64_t nTile,
                                                                             uint8_t enAtomic)
{
    const TCubeTiling& tiling = block_.matmulFp32TilingData_->tCubeTiling;
    const uint64_t tileM = tiling.singleCoreM;
    const uint64_t tileN = tiling.singleCoreN;
    const uint64_t mStart = mTile * tileM;
    const uint64_t nStart = nTile * tileN;
    const uint64_t mUse = Min(tileM, static_cast<uint64_t>(tiling.M) - mStart);
    const uint64_t nUse = Min(tileN, static_cast<uint64_t>(tiling.N) - nStart);
    const bool transB = block_.matmulFp32TilingData_->matmulFp32RunInfo.transB != 0;
    const uint64_t bOffset = transB ? nStart * tiling.Kb : nStart;

    mm_.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, tiling.N);
    mm_.SetSingleShape(mUse, nUse, tiling.singleCoreK);
    mm_.SetTensorA(aGlobal_[mStart * tiling.Ka], false);
    mm_.SetTensorB(bGlobal_[bOffset], transB);
    if (tiling.isBias) {
        mm_.SetBias(biasGlobal_[nStart]);
    }
    mm_.Iterate();
    mm_.GetTensorC(cGlobal_[mStart * tiling.N + nStart], enAtomic);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::ComputeRemoteTile(uint64_t wfIndex, uint64_t localM,
                                                                             uint64_t localN, uint64_t mTile,
                                                                             uint64_t nTile)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const TCubeTiling& tiling = block_.matmulFp32TilingData_->tCubeTiling;
    const uint64_t tileM = tiling.singleCoreM;
    const uint64_t tileN = tiling.singleCoreN;
    const uint64_t mStart = mTile * tileM;
    const uint64_t nStart = nTile * tileN;
    const uint64_t mUse = Min(tileM, static_cast<uint64_t>(tiling.M) - mStart);
    const uint64_t nUse = Min(tileN, static_cast<uint64_t>(tiling.N) - nStart);
    const uint64_t wfNStride = static_cast<uint64_t>(tilelet.wavefrontN) * tileN;
    const uint64_t wfRow = wfIndex / tilelet.wavefrontCols;
    const uint64_t wfCol = wfIndex - wfRow * tilelet.wavefrontCols;
    const bool transB = block_.matmulFp32TilingData_->matmulFp32RunInfo.transB != 0;
    const uint64_t aOffset = wfRow * tilelet.aSlabElements + localM * tileM * tiling.Ka;
    const uint64_t bOffset =
        wfCol * tilelet.bSlabElements + (transB ? localN * tileN * tiling.Kb : localN * tileN);
    const uint64_t dOffset =
        wfIndex * tilelet.dSlabElements + localM * tileM * wfNStride + localN * tileN;

    mm_.SetOrgShape(static_cast<uint64_t>(tilelet.wavefrontM) * tileM, wfNStride, tiling.Ka, tiling.Kb, wfNStride);
    mm_.SetSingleShape(mUse, nUse, tiling.singleCoreK);
    mm_.SetTensorA(aStageGlobal_[aOffset], false);
    mm_.SetTensorB(bStageGlobal_[bOffset], transB);
    if (tiling.isBias) {
        mm_.SetBias(biasGlobal_[nStart]);
    }
    mm_.Iterate();
    mm_.GetTensorC(dStageGlobal_[dOffset], 0);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::ComputeRemoteTileDirectD(uint64_t wfIndex, uint64_t localM,
                                                                                    uint64_t localN, uint64_t mTile,
                                                                                    uint64_t nTile)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const TCubeTiling& tiling = block_.matmulFp32TilingData_->tCubeTiling;
    const uint64_t tileM = tiling.singleCoreM;
    const uint64_t tileN = tiling.singleCoreN;
    const uint64_t mStart = mTile * tileM;
    const uint64_t nStart = nTile * tileN;
    const uint64_t mUse = Min(tileM, static_cast<uint64_t>(tiling.M) - mStart);
    const uint64_t nUse = Min(tileN, static_cast<uint64_t>(tiling.N) - nStart);
    const uint64_t wfRow = wfIndex / tilelet.wavefrontCols;
    const uint64_t wfCol = wfIndex - wfRow * tilelet.wavefrontCols;
    const bool transB = block_.matmulFp32TilingData_->matmulFp32RunInfo.transB != 0;
    const uint64_t aOffset = wfRow * tilelet.aSlabElements + localM * tileM * tiling.Ka;
    const uint64_t bOffset =
        wfCol * tilelet.bSlabElements + (transB ? localN * tileN * tiling.Kb : localN * tileN);

    mm_.SetOrgShape(static_cast<uint64_t>(tilelet.wavefrontM) * tileM, tiling.N, tiling.Ka, tiling.Kb, tiling.N);
    mm_.SetSingleShape(mUse, nUse, tiling.singleCoreK);
    mm_.SetTensorA(aStageGlobal_[aOffset], false);
    mm_.SetTensorB(bStageGlobal_[bOffset], transB);
    if (tiling.isBias) {
        mm_.SetBias(biasGlobal_[nStart]);
    }
    mm_.Iterate();
    mm_.GetTensorC(cGlobal_[mStart * tiling.N + nStart], 0);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::CopyBackRemoteTile(uint64_t wfIndex, uint64_t localM,
                                                                              uint64_t localN, uint64_t mTile,
                                                                              uint64_t nTile)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const TCubeTiling& tiling = block_.matmulFp32TilingData_->tCubeTiling;
    const uint64_t tileM = tiling.singleCoreM;
    const uint64_t tileN = tiling.singleCoreN;
    const uint64_t mStart = mTile * tileM;
    const uint64_t nStart = nTile * tileN;
    const uint64_t mUse = Min(tileM, static_cast<uint64_t>(tiling.M) - mStart);
    const uint64_t nUse = Min(tileN, static_cast<uint64_t>(tiling.N) - nStart);
    const uint64_t wfNStride = static_cast<uint64_t>(tilelet.wavefrontN) * tileN;
    const uint64_t srcBase =
        wfIndex * tilelet.dSlabElements + localM * tileM * wfNStride + localN * tileN;
    const uint64_t dstBase = mStart * tiling.N + nStart;

    for (uint64_t row = 0; row < mUse; ++row) {
        CopyGmRange(cGlobal_, dstBase + row * tiling.N, dStageGlobal_, srcBase + row * wfNStride, nUse);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::CopyBackRemoteWavefront(uint64_t wfIndex, uint64_t coreIdx)
{
    const TileletMatmulFp32TileletInfo& tilelet = block_.matmulFp32TilingData_->tileletInfo;
    const uint64_t slotsPerWavefront = static_cast<uint64_t>(tilelet.wavefrontM) * tilelet.wavefrontN;
    const uint64_t wfStart = wfIndex * slotsPerWavefront;
    const uint64_t wfEnd = Min(wfStart + slotsPerWavefront, static_cast<uint64_t>(tilelet.compactTileSlots));
    uint64_t remoteOrdinal = 0;
    for (uint64_t slot = wfStart; slot < wfEnd; ++slot) {
        uint64_t slotWf = 0;
        uint64_t localM = 0;
        uint64_t localN = 0;
        uint64_t mTile = 0;
        uint64_t nTile = 0;
        if (!IsRemoteSlot(slot) || !DecodeCompactSlot(slot, slotWf, localM, localN, mTile, nTile)) {
            continue;
        }
        if (remoteOrdinal % tilelet.commCoreNum == coreIdx) {
            CopyBackRemoteTile(slotWf, localM, localN, mTile, nTile);
        }
        ++remoteOrdinal;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
template <typename T>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::CopyGmRange(GlobalTensor<T>& dst, uint64_t dstOffset,
                                                                       GlobalTensor<T>& src, uint64_t srcOffset,
                                                                       uint64_t elemCount)
{
    if (elemCount == 0) {
        return;
    }
    constexpr uint32_t elemPerBlock = ONE_BLK_SIZE / sizeof(T);
    constexpr uint32_t maxElems = TILELET_MATMUL_FP32_COPY_UB_BYTES / sizeof(T);
    LocalTensor<T> local = copyBuf_.template Get<T>();
    uint64_t copied = 0;
    while (copied < elemCount) {
        const uint64_t current = Min(static_cast<uint64_t>(maxElems), elemCount - copied);
        if (current % elemPerBlock == 0) {
            DataCopy(local, src[srcOffset + copied], static_cast<uint32_t>(current));
            PipeBarrier<PIPE_ALL>();
            DataCopy(dst[dstOffset + copied], local, static_cast<uint32_t>(current));
        } else {
            const uint32_t rightPad = static_cast<uint32_t>((elemPerBlock - current % elemPerBlock) % elemPerBlock);
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(current * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{rightPad != 0, 0, static_cast<uint8_t>(rightPad), static_cast<T>(0)};
            DataCopyPad(local, src[srcOffset + copied], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(dst[dstOffset + copied], local, copyParams);
        }
        PipeBarrier<PIPE_ALL>();
        copied += current;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
template <typename T>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::CopyGmRangePart(GlobalTensor<T>& dst, uint64_t dstOffset,
                                                                           GlobalTensor<T>& src, uint64_t srcOffset,
                                                                           uint64_t elemCount, uint64_t worker,
                                                                           uint64_t workers)
{
    if (elemCount == 0 || workers == 0 || worker >= workers) {
        return;
    }
    const uint64_t begin = elemCount * worker / workers;
    const uint64_t end = elemCount * (worker + 1) / workers;
    if (end <= begin) {
        return;
    }
    CopyGmRange(dst, dstOffset + begin, src, srcOffset + begin, end - begin);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
template <typename T>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::CopyGm2D(GlobalTensor<T>& dst, uint64_t dstOffset,
                                                                    uint64_t dstRowStride, GlobalTensor<T>& src,
                                                                    uint64_t srcOffset, uint64_t srcRowStride,
                                                                    uint64_t rows, uint64_t rowElems)
{
    if (rows == 0 || rowElems == 0) {
        return;
    }
    if (rowElems == srcRowStride && rowElems == dstRowStride) {
        CopyGmRange(dst, dstOffset, src, srcOffset, rows * rowElems);
        return;
    }

    for (uint64_t row = 0; row < rows; ++row) {
        CopyGmRange(dst, dstOffset + row * dstRowStride, src, srcOffset + row * srcRowStride, rowElems);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
template <typename T>
__aicore__ inline void TileletMatmulFp32BaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                   MM_CB>::CopyGm2DPart(GlobalTensor<T>& dst, uint64_t dstOffset,
                                                                        uint64_t dstRowStride, GlobalTensor<T>& src,
                                                                        uint64_t srcOffset, uint64_t srcRowStride,
                                                                        uint64_t rows, uint64_t rowElems,
                                                                        uint64_t worker, uint64_t workers)
{
    if (rows == 0 || rowElems == 0 || workers == 0 || worker >= workers) {
        return;
    }
    const uint64_t rowBegin = rows * worker / workers;
    const uint64_t rowEnd = rows * (worker + 1) / workers;
    if (rowEnd <= rowBegin) {
        return;
    }
    CopyGm2D(dst, dstOffset + rowBegin * dstRowStride, dstRowStride, src, srcOffset + rowBegin * srcRowStride,
             srcRowStride, rowEnd - rowBegin, rowElems);
}

#endif // __TILELET_MATMUL_FP32_BASE_KERNEL_H__
