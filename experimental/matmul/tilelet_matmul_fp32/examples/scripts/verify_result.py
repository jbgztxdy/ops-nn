#!/usr/bin/python3
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import math
import os
import struct
import sys


def read_f32(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not aligned to float32")
    return struct.unpack(f"<{len(data) // 4}f", data)


def read_bf16(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 2 != 0:
        raise ValueError(f"{path} size is not aligned to bf16")
    values = struct.unpack(f"<{len(data) // 2}H", data)
    return tuple(struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in values)


def tolerances():
    dtype = os.getenv("TILELET_DTYPE", "fp32").lower()
    if dtype == "bf16":
        return 5e-2, 5e-2, 1e-3
    return 1e-4, 1e-3, 1e-4


def verify_result(output_path, golden_path):
    dtype = os.getenv("TILELET_DTYPE", "fp32").lower()
    output = read_bf16(output_path) if dtype == "bf16" else read_f32(output_path)
    golden = read_f32(golden_path)
    if len(output) != len(golden):
        raise ValueError(f"size mismatch, output={len(output)}, golden={len(golden)}")

    relative_tol, absolute_tol, error_tol = tolerances()
    out_tolerance_num = 0
    printed = 0
    for index, (actual, expected) in enumerate(zip(output, golden)):
        if math.isclose(actual, expected, rel_tol=relative_tol, abs_tol=absolute_tol):
            continue
        denom = max(abs(expected), absolute_tol)
        rdiff = abs(actual - expected) / denom
        if printed < 100:
            print("data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
                  (index, expected, actual, rdiff))
            printed += 1
        if rdiff > error_tol:
            out_tolerance_num += 1
    error_ratio = float(out_tolerance_num) / float(len(golden))
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol


if __name__ == "__main__":
    try:
        if not verify_result(sys.argv[1], sys.argv[2]):
            raise ValueError("[ERROR] result error")
        print("test pass")
    except Exception as err:
        print(err)
        sys.exit(1)
