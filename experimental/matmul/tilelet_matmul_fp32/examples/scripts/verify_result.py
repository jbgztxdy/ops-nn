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
import struct
import sys

RELATIVE_TOL = 1e-4
ABSOLUTE_TOL = 1e-3
ERROR_TOL = 1e-4


def read_f32(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not aligned to float32")
    return struct.unpack(f"{len(data) // 4}f", data)


def verify_result(output_path, golden_path):
    output = read_f32(output_path)
    golden = read_f32(golden_path)
    if len(output) != len(golden):
        raise ValueError(f"size mismatch, output={len(output)}, golden={len(golden)}")

    out_tolerance_num = 0
    printed = 0
    for index, (actual, expected) in enumerate(zip(output, golden)):
        if math.isclose(actual, expected, rel_tol=RELATIVE_TOL, abs_tol=ABSOLUTE_TOL):
            continue
        denom = max(abs(expected), ABSOLUTE_TOL)
        rdiff = abs(actual - expected) / denom
        if printed < 100:
            print("data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
                  (index, expected, actual, rdiff))
            printed += 1
        if rdiff > ERROR_TOL:
            out_tolerance_num += 1
    error_ratio = float(out_tolerance_num) / float(len(golden))
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, ERROR_TOL))
    return error_ratio <= ERROR_TOL


if __name__ == "__main__":
    try:
        if not verify_result(sys.argv[1], sys.argv[2]):
            raise ValueError("[ERROR] result error")
        print("test pass")
    except Exception as err:
        print(err)
        sys.exit(1)
