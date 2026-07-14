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
        this->Input("x1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("x2").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("transpose_x1").AttrType(OPTIONAL).Bool(false);
        this->Attr("transpose_x2").AttrType(OPTIONAL).Bool(false);
        this->Attr("remote_tile_start").AttrType(OPTIONAL).Int(0);
        this->Attr("remote_tile_count").AttrType(OPTIONAL).Int(0);
        this->Attr("wavefront_m").AttrType(OPTIONAL).Int(8);
        this->Attr("wavefront_n").AttrType(OPTIONAL).Int(4);
        this->Attr("comm_core_num").AttrType(OPTIONAL).Int(2);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(TileletMatmulFp32);
} // namespace ops
