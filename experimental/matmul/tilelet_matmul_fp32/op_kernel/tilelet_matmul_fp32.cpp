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
 * \file tilelet_matmul_fp32.cpp
 * \brief
 */

#include "tilelet_matmul_fp32_common.h"
#include "tilelet_matmul_fp32_base_kernel.h"
#include "tilelet_matmul_fp32_bl1_fullload_kernel.h"
#include "tilelet_matmul_fp32_tiling_key.h"

using namespace AscendC;
using namespace matmul;

constexpr CubeFormat format_x1 = CubeFormat::ND;
constexpr CubeFormat format_x2 = CubeFormat::ND;
constexpr CubeFormat format_y = CubeFormat::ND;

#define MM_FP32_IMPL_CLASS(templateClass, aFormat, ...)                                                    \
    do {                                                                                                   \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, D_T>;                                   \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, D_T>;                          \
        TPipe pipe;                                                                                        \
        if (tilingData.matmulFp32RunInfo.transA == 0 && tilingData.matmulFp32RunInfo.transB == 0) {        \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, D_T, false>;                         \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, D_T, false>;                       \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                  \
            op.Init(aGM, bGM, cGM, biasGM, user, arenaGM, peerOutGM, peerDsignalGM, &tilingData, &pipe);                  \
            op.Process();                                                                                  \
        } else if (tilingData.matmulFp32RunInfo.transA == 1 && tilingData.matmulFp32RunInfo.transB == 0) { \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, D_T, true>;                          \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, D_T, false>;                       \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                  \
            op.Init(aGM, bGM, cGM, biasGM, user, arenaGM, peerOutGM, peerDsignalGM, &tilingData, &pipe);                  \
            op.Process();                                                                                  \
        } else if (tilingData.matmulFp32RunInfo.transA == 0 && tilingData.matmulFp32RunInfo.transB == 1) { \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, D_T, false>;                         \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, D_T, true>;                        \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                  \
            op.Init(aGM, bGM, cGM, biasGM, user, arenaGM, peerOutGM, peerDsignalGM, &tilingData, &pipe);                  \
            op.Process();                                                                                  \
        } else {                                                                                           \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, D_T, true>;                          \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, D_T, true>;                        \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                  \
            op.Init(aGM, bGM, cGM, biasGM, user, arenaGM, peerOutGM, peerDsignalGM, &tilingData, &pipe);                  \
            op.Process();                                                                                  \
        }                                                                                                  \
    } while (0)

template <typename D_T, int LOADMODE, int SPLITCOREMODE, int FIXOPTI, int MIXND2NZ>
__global__ __aicore__ void tilelet_matmul_fp32(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR arenaGM,
                                       GM_ADDR peerOutGM, GM_ADDR peerDsignalGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                       GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(TileletMatmulFp32TilingData);
    GET_TILING_DATA_WITH_STRUCT(TileletMatmulFp32TilingData, tilingData, tilingGM);
    __gm__ uint8_t* user = GetUserWorkspace(workspaceGM);
    if constexpr (LOADMODE == TILELET_MATMUL_FP32_NO_FULLLOAD && SPLITCOREMODE == TILELET_MATMUL_FP32_BASE_SPLIT_K &&
                  FIXOPTI == TILELET_MATMUL_FP32_BASE_FIXOPTI && MIXND2NZ == TILELET_MATMUL_FP32_MIXND2NZ_TRUE) {
        MM_FP32_IMPL_CLASS(TileletMatmulFp32BaseKernel, format_x1, TileletMatmulFp32BaseBlock, MM_CFG_NO_PRELOAD);
    } else if constexpr (LOADMODE == TILELET_MATMUL_FP32_BL1_FULLLOAD && SPLITCOREMODE == TILELET_MATMUL_FP32_BASE_SPLIT_K &&
                         FIXOPTI == TILELET_MATMUL_FP32_BASE_FIXOPTI && MIXND2NZ == TILELET_MATMUL_FP32_MIXND2NZ_TRUE) {
        MM_FP32_IMPL_CLASS(TileletMatmulFp32BL1FullLoadKernel, format_x1, TileletMatmulFp32BaseBlock, MM_CFG_NO_PRELOAD);
    }
}
