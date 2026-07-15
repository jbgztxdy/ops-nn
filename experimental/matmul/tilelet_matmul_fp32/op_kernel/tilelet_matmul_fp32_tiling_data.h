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
 * \file tilelet_matmul_fp32_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __TILELET_MATMUL_FP32_TILING_DATA_H__
#define __TILELET_MATMUL_FP32_TILING_DATA_H__

#include "kernel_tiling/kernel_tiling.h"

struct TileletMatmulFp32RunInfo {
    uint32_t transA;
    uint32_t transB;
};

struct TileletMatmulFp32TileletInfo {
    uint32_t wavefrontM;
    uint32_t wavefrontN;
    uint32_t commCoreNum;
    uint32_t computeCoreNum;
    uint32_t totalCoreNum;
    uint32_t mTileCnt;
    uint32_t nTileCnt;
    uint32_t wavefrontRows;
    uint32_t wavefrontCols;
    uint32_t wavefrontCount;
    uint32_t commKTiles;
    uint32_t kGroupCount;
    uint32_t enableDCopyback;
    uint32_t compactTileSlots;
    uint32_t remoteTileStart;
    uint32_t remoteTileEnd;
    uint64_t aSlabElements;
    uint64_t bSlabElements;
    uint64_t dSlabElements;
    uint64_t abSignalByteOffset;
    uint64_t dSignalByteOffset;
    uint64_t aSlabByteOffset;
    uint64_t bSlabByteOffset;
    uint64_t dSlabByteOffset;
    uint64_t userWorkspaceBytes;
};

struct TileletMatmulFp32TilingData {
    TCubeTiling tCubeTiling;
    TileletMatmulFp32RunInfo matmulFp32RunInfo;
    TileletMatmulFp32TileletInfo tileletInfo;
};
#endif
