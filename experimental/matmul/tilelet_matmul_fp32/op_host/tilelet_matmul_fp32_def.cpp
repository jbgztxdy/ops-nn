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
 * \file tilelet_matmul_fp32_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class TileletMatmulFp32 : public OpDef {
public:
    explicit TileletMatmulFp32(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_BF16})
            .FormatForBinQuery({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_BF16})
            .FormatForBinQuery({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_BF16})
            .FormatForBinQuery({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // Cross-card ("peer 直存") staging/signal arena. Peer-visible buffer
        // shared between the source and compute ranks. When absent the op runs
        // the original single-card path.
        this->Input("arena")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_BF16})
            .FormatForBinQuery({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // Compute rank only: base address of the source rank's output C, so
        // remote tiles can be written back directly across the card link.
        this->Input("peer_out")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_BF16})
            .FormatForBinQuery({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_BF16})
            .FormatForBinQuery({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("transpose_x1").AttrType(OPTIONAL).Bool(false);
        this->Attr("transpose_x2").AttrType(OPTIONAL).Bool(false);
        this->Attr("remote_tile_start").AttrType(OPTIONAL).Int(0);
        this->Attr("remote_tile_count").AttrType(OPTIONAL).Int(0);
        this->Attr("wavefront_m").AttrType(OPTIONAL).Int(16);
        this->Attr("wavefront_n").AttrType(OPTIONAL).Int(8);
        this->Attr("comm_core_num").AttrType(OPTIONAL).Int(8);
        this->Attr("comm_k_tiles").AttrType(OPTIONAL).Int(32);
        this->Attr("enable_d_copyback").AttrType(OPTIONAL).Bool(false);
        // Cross-card role: 0 = single-card/source, 1 = remote compute rank.
        // rank_id/world_size are reserved for multi-rank scheduling; only role
        // drives kernel behavior in the current peer-store implementation.
        this->Attr("role").AttrType(OPTIONAL).Int(0);
        this->Attr("rank_id").AttrType(OPTIONAL).Int(0);
        this->Attr("world_size").AttrType(OPTIONAL).Int(1);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(TileletMatmulFp32);
} // namespace ops
