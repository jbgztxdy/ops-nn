/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <tuple>
#include <vector>

using OP_RES = std::tuple<const uint8_t*, const uint8_t*>;
using OP_RUNTIME_KB_RES = std::vector<OP_RES>;

namespace l0op {
void* TileletMatmulFp32TuningRegisterResource()
{
    return nullptr;
}

const OP_RUNTIME_KB_RES& TileletMatmulFp32TuningResource()
{
    static const OP_RUNTIME_KB_RES resource = {};
    return resource;
}
} // namespace l0op
