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
#
# Fast (numpy) data + golden generator for TileletMatmulFp32. The golden is a
# high-precision (float64-accumulation) reference of
#     output[M,N] = x1[M,K] @ x2^(T)  (+ bias)
# with inputs quantized to the requested dtype (bf16 or fp32). It is a vectorized
# drop-in for the original triple-loop generator and is bit-equivalent to it:
#   - identical deterministic input formula ((i*mul)%mod)/mod + bias, rounded to f32
#   - identical round-to-nearest-even bf16 quantization
#   - identical float64 accumulation, final result rounded to f32
# Rewriting the golden in numpy is what makes large (vLLM-scale) shapes tractable;
# the old pure-Python loop was O(M*N*K) in the interpreter and unusable past a few
# hundred K elements. The pure-Python path is kept as a fallback when numpy is
# unavailable.

import os
import struct

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - fallback only
    HAVE_NUMPY = False


def env_i64(name, default):
    value = os.getenv(name)
    return default if value is None else int(value)


M = env_i64("TILELET_M", 256)
N = env_i64("TILELET_N", 512)
K = env_i64("TILELET_K", 64)


def _flags():
    transpose_x2 = os.getenv("TILELET_TRANSPOSE_X2", "0") not in ("", "0", "false", "False")
    use_bias = os.getenv("TILELET_USE_BIAS", "1") not in ("", "0", "false", "False")
    dtype = os.getenv("TILELET_DTYPE", "fp32").lower()
    return transpose_x2, use_bias, dtype


# ------------------------------------------------------------------ numpy path


def _np_f32_to_bf16_bits(values_f32):
    # Round-to-nearest-even f32 -> bf16, matching the scalar reference exactly.
    bits = values_f32.astype(np.float32).view(np.uint32).astype(np.uint64)
    rounded = bits + np.uint64(0x7FFF) + ((bits >> np.uint64(16)) & np.uint64(1))
    return ((rounded >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16)


def _np_bf16_bits_to_f32(bits_u16):
    return (bits_u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _np_gen_input(size, mul, mod, bias):
    idx = np.arange(size, dtype=np.int64)
    raw = ((idx * mul) % mod).astype(np.float64) / float(mod) + bias
    return raw.astype(np.float32)


def _np_gen():
    transpose_x2, use_bias, dtype = _flags()
    a_f32 = _np_gen_input(M * K, 13, 97, -0.5)
    b_f32 = _np_gen_input(K * N, 17, 89, -0.25)
    bias_f32 = _np_gen_input(N, 19, 83, -0.125)

    if dtype == "bf16":
        a_bits = _np_f32_to_bf16_bits(a_f32)
        b_bits = _np_f32_to_bf16_bits(b_f32)
        bias_bits = _np_f32_to_bf16_bits(bias_f32)
        a_val = _np_bf16_bits_to_f32(a_bits)
        b_val = _np_bf16_bits_to_f32(b_bits)
        bias_val = _np_bf16_bits_to_f32(bias_bits)
    else:
        a_bits = b_bits = bias_bits = None
        a_val, b_val, bias_val = a_f32, b_f32, bias_f32

    # float64 accumulation (matches Python-float reference), reshaped by layout.
    a_mat = a_val.astype(np.float64).reshape(M, K)
    if transpose_x2:
        b_mat = b_val.astype(np.float64).reshape(N, K)      # x2[N,K], compute A @ B^T
        golden = a_mat @ b_mat.T
    else:
        b_mat = b_val.astype(np.float64).reshape(K, N)      # x2[K,N], compute A @ B
        golden = a_mat @ b_mat
    if use_bias:
        golden = golden + bias_val.astype(np.float64).reshape(1, N)
    golden = golden.astype(np.float32).reshape(-1)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    if dtype == "bf16":
        a_bits.tofile("./input/input_a.bin")
        b_bits.tofile("./input/input_b.bin")
        if use_bias:
            bias_bits.tofile("./input/input_bias.bin")
    else:
        a_val.astype(np.float32).tofile("./input/input_a.bin")
        b_val.astype(np.float32).tofile("./input/input_b.bin")
        if use_bias:
            bias_val.astype(np.float32).tofile("./input/input_bias.bin")
    golden.tofile("./output/golden.bin")


# ------------------------------------------------------- pure-Python fallback


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


def _py_gen():
    transpose_x2, use_bias, dtype = _flags()
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


def gen_golden_data():
    if HAVE_NUMPY:
        _np_gen()
    else:
        _py_gen()


if __name__ == "__main__":
    gen_golden_data()
