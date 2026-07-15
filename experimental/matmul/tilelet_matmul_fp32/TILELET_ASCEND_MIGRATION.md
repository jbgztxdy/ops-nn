# Tilelet Ascend Migration Notes

本文记录 `TileletMatmulFp32` 算子从 CUDA/CUTLASS tilelet kernel 迁移到 Ascend-C/ACLLN 的工作内容。路径均按 workspace 根目录书写，例如 `ops-nn/...`、`vllm/...`、`tilelet-sched/cutlass/...`，不要依赖机器上的绝对路径。

## 目标

迁移目标不是普通 matmul，而是 vLLM disaggregated prefill benchmark 中使用的 tilelet remote linear 算子。该 CUDA 路径把 GEMM 的一部分 CTA 分给 remote GPU 计算，同时由一部分 SM 负责 copy A/B staged 数据，remote 计算结束后按配置直接写 D 或 copyback。benchmark 当前使用的是 direct-D 路径。

对应关系：

- vLLM 调用入口：`vllm/vllm/tilelet.py` 的 `remote_linear()`。
- vLLM benchmark 环境变量设置：`vllm/eval/disaggregated_prefill_benchmark.py` 和 `vllm/eval/config/tilelet_fixed.json`。
- vLLM C++ binding：`vllm/csrc/tilelet/torch_bindings.cpp`。
- CUDA tilelet copy/compute：`tilelet-sched/cutlass/include/cutlass/gemm/device/tilelet_remote_copy.h`、`tilelet_remote_gemm.h`、`tilelet-sched/cutlass/include/cutlass/gemm/kernel/tilelet_gemm.h`。
- Ascend 迁移目标：`ops-nn/experimental/matmul/tilelet_matmul_fp32`。

## vLLM Benchmark 使用的语义

vLLM 中 `remote_linear(x, weight)` 的数学语义是：

```text
output[M, N] = x[M, K] @ weight[N, K]^T
```

benchmark 配置中的关键 tilelet 参数：

```text
comm_sms = 8
comm_k_tiles = 32
d_copyback = false
min_m = 128
linear_prefix_allowlist = qkv_proj,gate_up_proj,down_proj
```

CUDA binding 中会将该调用转成 `GemmLayoutMode::kLinearTN`：

```text
A = weight[N, K]
B = x[M, K]
C/D = output[M, N]
```

由于 Ascend 算子接口保留的是常规 matmul 形态，本迁移在示例中用：

```text
TILELET_DTYPE=bf16
TILELET_USE_BIAS=0
TILELET_TRANSPOSE_X2=1
TILELET_COMM_SMS=8
TILELET_COMM_K_TILES=32
TILELET_D_COPYBACK=0
```

来表达同一语义，即输入 `x1[M,K]` 和 `x2[N,K]`，设置 `transpose_x2=true` 后计算 `x1 @ x2^T`。

## 已修改的主要文件

### Op 定义

文件：`ops-nn/experimental/matmul/tilelet_matmul_fp32/op_host/tilelet_matmul_fp32_def.cpp`

目的：

- 支持 FP32 和 BF16。
- 支持 optional bias，便于对齐 vLLM 的 no-bias remote linear。
- 新增 `comm_k_tiles` 和 `enable_d_copyback` attr。
- 默认值对齐 vLLM benchmark：`wavefront_m=16`、`wavefront_n=8`、`comm_core_num=8`、`comm_k_tiles=32`、`enable_d_copyback=false`。

### Host Tiling

文件：`ops-nn/experimental/matmul/tilelet_matmul_fp32/op_host/tilelet_matmul_fp32_tiling.cpp`

目的：

- 解析 `transpose_x2`，支持 `x[M,K] @ weight[N,K]^T`。
- 允许 FP32/BF16，并要求 `x1/x2/y/bias` dtype 一致。
- 计算 tilelet wavefront 信息：
  - tile 大小来自 Ascend matmul tiling，目标路径为 128x256。
  - wavefront 默认为 16x8。
  - remote tile 范围由 `remote_tile_start` 和 `remote_tile_count` 表示。
  - `comm_k_tiles=32` 映射为每个 copy signal 覆盖 32 个 K tile。
- 计算 staged A/B/D workspace 和 signal workspace。
- 根据 dtype 生成 tiling key，确保 FP32 和 BF16 使用各自 bin。

注意：vLLM CUDA 路径通过 `min_remote_cta_ratio/max_remote_cta_ratio` 动态得到 remote CTA 数；当前 Ascend 单算子接口用 `remote_tile_start/remote_tile_count` 显式表达 remote tile 范围。后续接入 Ascend vLLM 时，需要在框架层把 ratio 调度结果映射为这两个 attr。

### Kernel Tiling Data

文件：`ops-nn/experimental/matmul/tilelet_matmul_fp32/op_kernel/tilelet_matmul_fp32_tiling_data.h`

目的：

- 在 tiling data 中增加：
  - `commKTiles`
  - `kGroupCount`
  - `enableDCopyback`

这些字段用于在 kernel 内复现 CUDA tilelet 的 `(signal, kGroup)` 同步粒度。

### Kernel Template / BF16

文件：

- `ops-nn/experimental/matmul/tilelet_matmul_fp32/op_kernel/tilelet_matmul_fp32.cpp`
- `ops-nn/experimental/matmul/tilelet_matmul_fp32/op_kernel/tilelet_matmul_fp32_tiling_key.h`

目的：

- 将原来固定 FP32 的 kernel 改为 dtype 模板参数 `D_T`。
- 生成 FP32 和 BF16 两套 kernel bin。
- BF16 是 vLLM tilelet remote linear 的主路径。

### Tilelet Kernel 主逻辑

文件：`ops-nn/experimental/matmul/tilelet_matmul_fp32/op_kernel/tilelet_matmul_fp32_base_kernel.h`

这是迁移的核心文件。

已对齐 CUDA/CUTLASS tilelet 的关键逻辑：

1. 分出通信 core
   - CUDA 中是 `comm_sms` 个 SM 运行 remote copy kernel。
   - Ascend 中用 `commCoreNum` 个 AIC core 运行 `ProcessCommCore()`。

2. copy 顺序一致
   - CUTLASS `TileletRemoteCopyKernel` 按 `ordered_sig` 遍历 signal。
   - 顺序是：
     - `sig=0`：copy `A0 + B0`
     - 后续先 copy A wavefront 行
     - 再 copy B wavefront 列
   - Ascend `ProcessCommCopyIn()` 同样按 `sig` 从小到大遍历，并由 `SignalOrigin()`/`SignalForA()`/`SignalForB()` 表达相同映射。

3. K group 同步粒度一致
   - CUTLASS 中：
     - `group_count = ceil(k_tile_count / CommKTiles)`
     - signal table 索引为 `signals[sig_idx * group_count + group]`
   - Ascend 中：
     - `kGroupCount = ceil(k_tile_count / commKTiles)`
     - signal table 索引为 `AbSignalIndex(sig, group)`
   - compute core 等待 `max(sigA, sigB)` 的全部 K group 完成后再计算 remote tile。

4. A/B staged copy 范围一致
   - 对 A：按 wavefront row 复制 `M` 范围和当前 K group。
   - 对 B：按 wavefront col 复制 `N` 范围和当前 K group。
   - `transpose_x2=true` 时，B staged copy 使用 `weight[N,K]` 布局。

5. remote tile 计算方式一致
   - direct tile 使用原始全局 A/B。
   - remote tile 使用 staged A/B。
   - no-bias BF16 路径对齐 vLLM。

6. `D_COPYBACK=0` 路径一致
   - vLLM benchmark 使用 `d_copyback=false`。
   - CUDA 路径此时 remote GEMM 直接写最终 D。
   - Ascend `enableDCopyback=false` 时，remote tile 通过 `ComputeRemoteTileDirectD()` 直接写最终 C。

`enableDCopyback=true` 路径仍保留为 staged D 后 copyback，但这不是当前 vLLM benchmark 的主路径。

## 示例程序修改

文件：

- `ops-nn/experimental/matmul/tilelet_matmul_fp32/examples/src/main.cpp`
- `ops-nn/experimental/matmul/tilelet_matmul_fp32/examples/src/op_runner.cpp`
- `ops-nn/experimental/matmul/tilelet_matmul_fp32/examples/scripts/gen_data.py`
- `ops-nn/experimental/matmul/tilelet_matmul_fp32/examples/scripts/verify_result.py`

目的：

- 用环境变量配置单算子测试形态。
- 支持 BF16 输入输出。
- 支持 no-bias。
- 支持 `transpose_x2=true` 的 `x @ weight^T`。
- 支持读取 `TILELET_COMM_SMS`、`TILELET_COMM_K_TILES`、`TILELET_D_COPYBACK`。
- 生成与 BF16 量化输入一致的 FP32 golden。

示例环境变量：

```bash
TILELET_DTYPE=bf16
TILELET_USE_BIAS=0
TILELET_TRANSPOSE_X2=1
TILELET_REMOTE_TILE_START=0
TILELET_REMOTE_TILE_COUNT=4
TILELET_WAVEFRONT_M=16
TILELET_WAVEFRONT_N=8
TILELET_COMM_SMS=8
TILELET_COMM_K_TILES=32
TILELET_D_COPYBACK=0
```

这些环境变量只用于示例程序。框架接入时可以直接把这些值作为 ACLNN attr 传入，不一定继续使用环境变量。

## 已验证内容

在 910B 上已经完成以下验证：

- BF16、no bias、`transpose_x2=true`、`comm_sms=8`、`comm_k_tiles=32`、`d_copyback=false`。
- FP32、no bias、`transpose_x2=true` 回归。
- FP32、bias、非 transpose 回归。

BF16 vLLM 形态验证结果：

```text
Run op success
error ratio: 0.0000
test pass
```

这说明当前单算子可以在 Ascend 设备上运行，且 tilelet 算法划分和数值正确。

## 迁移到 910C 后建议复验

以下命令以 `ops-nn` 为当前目录；不要直接复制机器相关绝对路径，按 910C 机器上的 CANN 安装位置设置变量。

### 构建

```bash
cmake -S . -B build
cmake --build build --target ascendc_bin_ascend910b_tilelet_matmul_fp32
cmake --build build --target binary -j8
python3 scripts/util/build_opp_kernel_static.py StaticCompile -s ascend910b -b build -n 0 -a aarch64
python3 scripts/util/build_opp_kernel_static.py GenStaticOpResourceIni -s ascend910b -b build
cmake --build build --target cann_nn_static -j8
cmake --build experimental/matmul/tilelet_matmul_fp32/examples/build -j8
```

如果 910C 使用不同 SoC 名称，需要把 `ascend910b` 替换成目标环境实际支持的 SoC，例如 CANN 工具链识别出的 910C 对应名字。

### 运行 vLLM 形态单算子测试

进入：

```bash
cd experimental/matmul/tilelet_matmul_fp32/examples
```

生成输入：

```bash
env TILELET_DTYPE=bf16 TILELET_USE_BIAS=0 TILELET_TRANSPOSE_X2=1 \
  python3 scripts/gen_data.py
```

运行：

```bash
cd output
env \
  LD_LIBRARY_PATH=../../../../build/common/stub/op_api:../../../../build:${CANN_ROOT}/lib64:${DRIVER_ROOT}/lib64:${DRIVER_ROOT}/lib64/common:${DRIVER_ROOT}/lib64/driver:${LD_LIBRARY_PATH} \
  ASCEND_OPP_PATH=${CANN_ROOT}/opp \
  TMPDIR=../../../../../.tmp-codex \
  TILELET_DTYPE=bf16 \
  TILELET_USE_BIAS=0 \
  TILELET_TRANSPOSE_X2=1 \
  TILELET_REMOTE_TILE_START=0 \
  TILELET_REMOTE_TILE_COUNT=4 \
  TILELET_WAVEFRONT_M=16 \
  TILELET_WAVEFRONT_N=8 \
  TILELET_COMM_SMS=8 \
  TILELET_COMM_K_TILES=32 \
  TILELET_D_COPYBACK=0 \
  ./execute_tilelet_matmul_fp32_op
```

校验：

```bash
cd ..
env TILELET_DTYPE=bf16 python3 scripts/verify_result.py output/output_z.bin output/golden.bin
```

期望看到：

```text
test pass
```

## 常见问题

### `The binary bin not found`

这通常是 op info、kernel JSON、`binary_info_config.json` 或静态 resource cpp 没更新。检查点：

- `build/autogen/aic-*-ops-info.ini` 中 `TileletMatmulFp32` 是否包含 `float32,bfloat16`。
- `build/binary/*/bin/config/*/tilelet_matmul_fp32.json` 是否包含 FP32 和 BF16 两个 bin。
- `build/binary/*/bin/config/*/binary_info_config.json` 是否包含新 attr：
  - `comm_k_tiles`
  - `enable_d_copyback`
- `build/autogen/*/aclnnop_resource/TileletMatmulFp32_op_resource.cpp` 是否注册了新 FP32/BF16 bin。

如果资源 cpp 没刷新，重新运行：

```bash
python3 scripts/util/build_opp_kernel_static.py GenStaticOpResourceIni -s <soc> -b build
cmake --build build --target cann_nn_static -j8
cmake --build experimental/matmul/tilelet_matmul_fp32/examples/build -j8
```

### CANN simulator 与真机 profile

CANN simulator 的 serial mode 只能验证算法划分和数值，不可靠验证真实 AIC copy 与 compute overlap。910C 上需要用真机 profiling 观察：

- AIC task timeline。
- HCCS/PCIe 或卡间链路吞吐。
- custom op task duration。
- comm core 与 compute core 是否有时间重叠。

910B 低带宽机器上正确性已经验证，但性能长尾不能代表高带宽 910C。

## 当前边界

已完成：

- vLLM benchmark 使用的 tilelet remote linear kernel 语义迁移。
- BF16/no-bias/`x @ weight^T` 路径。
- `comm_sms=8`、`comm_k_tiles=32`、`d_copyback=false` 路径。
- copy 顺序、K group signal 粒度、remote tile 计算方式对齐 CUTLASS tilelet。
- Ascend 设备单算子正确性验证。

尚未完成：

- Ascend 版 vLLM 端到端接入。
- 将 vLLM 的 ratio scheduler 自动映射为 Ascend op 的 `remote_tile_start/remote_tile_count`。
- 910C 高带宽环境下的 overlap 和端到端性能复现。

