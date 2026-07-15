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


def env_i64(name, default):
    value = os.getenv(name)
    return default if value is None else int(value)


M = env_i64("TILELET_M", 256)
N = env_i64("TILELET_N", 512)
K = env_i64("TILELET_K", 64)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def write_f32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_f32(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def f32_to_bf16_bits(value):
    bits = f32_bits(value)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def bf16_bits_to_f32(value):
    return bits_f32(value << 16)


def to_bf16_values(values):
    bits = [f32_to_bf16_bits(value) for value in values]
    return bits, [bf16_bits_to_f32(value) for value in bits]


def write_bf16(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def gen_input(size, mul, mod, bias):
    return [f32(((i * mul) % mod) / float(mod) + bias) for i in range(size)]


def gen_golden_data():
    transpose_x2 = os.getenv("TILELET_TRANSPOSE_X2", "0") not in ("", "0", "false", "False")
    use_bias = os.getenv("TILELET_USE_BIAS", "1") not in ("", "0", "false", "False")
    dtype = os.getenv("TILELET_DTYPE", "fp32").lower()
    raw_a = gen_input(M * K, 13, 97, -0.5)
    raw_b = gen_input(K * N, 17, 89, -0.25)
    raw_bias = gen_input(N, 19, 83, -0.125)
    if dtype == "bf16":
        input_a_bits, input_a = to_bf16_values(raw_a)
        input_b_bits, input_b = to_bf16_values(raw_b)
        input_bias_bits, input_bias = to_bf16_values(raw_bias)
    else:
        input_a_bits, input_a = None, raw_a
        input_b_bits, input_b = None, raw_b
        input_bias_bits, input_bias = None, raw_bias
    golden = [0.0] * (M * N)
    for m in range(M):
        a_row = m * K
        c_row = m * N
        for n in range(N):
            acc = float(input_bias[n]) if use_bias else 0.0
            for k in range(K):
                b_index = n * K + k if transpose_x2 else k * N + n
                acc += float(input_a[a_row + k]) * float(input_b[b_index])
            golden[c_row + n] = f32(acc)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    if dtype == "bf16":
        write_bf16("./input/input_a.bin", input_a_bits)
        write_bf16("./input/input_b.bin", input_b_bits)
        if use_bias:
            write_bf16("./input/input_bias.bin", input_bias_bits)
    else:
        write_f32("./input/input_a.bin", input_a)
        write_f32("./input/input_b.bin", input_b)
        if use_bias:
            write_f32("./input/input_bias.bin", input_bias)
    write_f32("./output/golden.bin", golden)


if __name__ == "__main__":
    gen_golden_data()
