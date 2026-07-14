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

import os
import struct


M = 256
N = 512
K = 64


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def write_f32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"{len(values)}f", *values))


def gen_input(size, mul, mod, bias):
    return [f32(((i * mul) % mod) / float(mod) + bias) for i in range(size)]


def gen_golden_data():
    input_a = gen_input(M * K, 13, 97, -0.5)
    input_b = gen_input(K * N, 17, 89, -0.25)
    input_bias = gen_input(N, 19, 83, -0.125)
    golden = [0.0] * (M * N)
    for m in range(M):
        a_row = m * K
        c_row = m * N
        for n in range(N):
            acc = float(input_bias[n])
            for k in range(K):
                acc += float(input_a[a_row + k]) * float(input_b[k * N + n])
            golden[c_row + n] = f32(acc)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    write_f32("./input/input_a.bin", input_a)
    write_f32("./input/input_b.bin", input_b)
    write_f32("./input/input_bias.bin", input_bias)
    write_f32("./output/golden.bin", golden)


if __name__ == "__main__":
    gen_golden_data()
