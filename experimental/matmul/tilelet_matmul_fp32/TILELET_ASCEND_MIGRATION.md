# Tilelet Ascend Migration Notes

本文记录 `TileletMatmulFp32` 算子从 CUDA/CUTLASS tilelet kernel 迁移到 Ascend-C/ACLNN 的工作。
读者是接手在 **910C** 上继续的 agent：请先读本文，再读代码，再动手。

路径一律按 workspace 根目录书写（`ops-nn/...`、`vllm/...`、`tilelet-sched/cutlass/...`，脚本用
`ops-nn/scripts/...`），**不要**写机器绝对路径。凡是临时目录，放在数据盘（大盘）下自建目录，
不要用根分区 `/`（本机 `/` 已满，见"构建注意事项"）。

## 命名说明（先读）

算子名 `tilelet_matmul_fp32` 里的 `fp32` 是**历史命名**（fork 自 fp32 matmul 基座），**不代表只支持 fp32**：

- 算子**同时支持 fp32 和 bf16**：`op_host/tilelet_matmul_fp32_def.cpp` 所有张量 dtype 为
  `{ge::DT_FLOAT, ge::DT_BF16}`，构建会产出 fp32 与 bf16 两套 kernel bin。
- **vLLM tilelet remote_linear 强制 bf16**：`vllm/vllm/tilelet.py` 里非 bf16 输入直接 fallback（`return "dtype"`），
  模型 `Meta-Llama-3-8B` 走 bf16。故 **vLLM 相关精度就是 bf16**。
- 本项目的跨卡验证**全部在 bf16 下完成**（即 vLLM 路径）；fp32 只是顺带保留的回归路径。
- 不建议重命名（涉及目录 / op type / aclnn 接口，改动大、收益低），以本说明为准即可。

## 目标

迁移目标不是普通 matmul，而是 vLLM disaggregated prefill benchmark 中使用的 tilelet remote linear
算子。CUDA 路径把 GEMM 的一部分 CTA 分给 remote GPU 计算，同时一部分 SM 负责 copy A/B staged 数据，
remote 计算结束后按配置直接写 D 或 copyback。benchmark 用的是 direct-D 路径。

对应关系：

- vLLM 调用入口：`vllm/vllm/tilelet.py` 的 `remote_linear()`。
- vLLM benchmark 与配置：`vllm/eval/disaggregated_prefill_benchmark.py`、`vllm/eval/config/tilelet_fixed.json`。
- vLLM C++ binding：`vllm/csrc/tilelet/torch_bindings.cpp`。
- CUDA tilelet copy/compute kernel：`tilelet-sched/cutlass/include/cutlass/gemm/device/tilelet_remote_copy.h`、
  `tilelet_remote_gemm.h`、`tilelet-sched/cutlass/include/cutlass/gemm/kernel/tilelet_gemm.h`。
- CUDA 跨卡框架层（Ascend 已完成单算子两 rank gate；完整 vLLM/框架编排层仍需接入时对照）：
  `tilelet-sched/src/tilelet_same_va_allocator.cu`、`tilelet-sched/src/tilelet_multiprocess_gemm.cu`。
- Ascend 迁移目标：`ops-nn/experimental/matmul/tilelet_matmul_fp32`。

## vLLM benchmark 使用的语义

`remote_linear(x, weight)` 的数学语义：`output[M,N] = x[M,K] @ weight[N,K]^T`。

benchmark 关键 tilelet 参数（`tilelet_fixed.json`）：`comm_sms=8`、`comm_k_tiles=32`、`d_copyback=false`、
`min_m=128`、`linear_prefix_allowlist=qkv_proj,gate_up_proj,down_proj`、
`min_remote_cta_ratio=0.0`、`max_remote_cta_ratio=0.4`。

CUDA binding 把该调用转成 `GemmLayoutMode::kLinearTN`：`A=weight[N,K]`、`B=x[M,K]`、`C/D=output[M,N]`。

Ascend 单算子接口保留常规 matmul 形态，用环境变量表达同一语义：
`TILELET_DTYPE=bf16`、`TILELET_USE_BIAS=0`、`TILELET_TRANSPOSE_X2=1`、`TILELET_COMM_SMS=8`、
`TILELET_COMM_K_TILES=32`、`TILELET_D_COPYBACK=0`，即 `x1[M,K]` 和 `x2[N,K]`、`transpose_x2=true` 算
`x1 @ x2^T`。

---

# 第一阶段：单卡迁移（已完成）

把 CUDA tilelet 的 **tiling/wavefront 切分 + signal 同步结构** 在**单卡内**复刻，用一块 workspace 承载
staged A/B/D 与 signal。这一阶段与 CUTLASS 对齐的要点（详见各文件注释与代码）：

- 通信 core：CUDA 的 `comm_sms` 个 SM ↔ Ascend `commCoreNum` 个 AIV core 跑 `ProcessCommCore()`。
- copy 顺序：CUTLASS `TileletRemoteCopyKernel` 按 `ordered_sig` 遍历（sig=0 copy A0+B0，之后先 A 行后 B 列）；
  Ascend `ProcessCommCopyIn()` 用 `SignalOrigin/SignalForA/SignalForB` 表达同一映射。
- K group 粒度：`group_count = ceil(k_tile_count / CommKTiles)`，signal 索引 `sig*group_count+group`
  （Ascend `kGroupCount` / `AbSignalIndex`）。
- direct-D（`d_copyback=false`）：remote tile 由 `ComputeRemoteTileDirectD()` 直接写最终 C。

涉及文件：`op_host/tilelet_matmul_fp32_def.cpp`、`op_host/tilelet_matmul_fp32_tiling.cpp`、
`op_kernel/tilelet_matmul_fp32_tiling_data.h`、`op_kernel/tilelet_matmul_fp32.cpp`、
`op_kernel/tilelet_matmul_fp32_base_kernel.h`（核心）、`op_kernel/tilelet_matmul_fp32_tiling_key.h`。

单卡验证（910B）：BF16 / FP32、no-bias / bias、transpose / 非 transpose 回归均 `test pass`。

---

# 第二阶段：跨卡迁移（peer 直存，本轮新增，已跨卡验证）

## 机制选择

CUDA tilelet 的跨卡靠 same-VA allocator（`tilelet_same_va_allocator.cu`）把对端 GPU 显存映射到同一 VA，
kernel 里裸 peer store（`dst[i]=src[i]`）即写到对端卡。昇腾 910B 已确认支持 P2P
（`aclrtDeviceCanAccessPeer` / `aclrtDeviceEnablePeerAccess`），本轮采用**最贴近 tilelet 的 peer 直存**：
kernel 内的 `DataCopy`/signal 直接指向 peer 可见内存。已用两卡 gate 端到端证明该机制在 910B 上成立
（kernel 跨 HCCS 直写对端 HBM + 跨卡 flag 轮询都可靠）。

（注：独立的 AscendC kernel-launch spike 因该 CANN 版本 `ascendc_library` 的 host-stub 分支有 bug 而放弃，
改为直接在真算子 + 单算子测试驱动里验证，即下面的两卡 gate。）

## 设计（direct-D，两 rank 协同）

两张卡各 launch **同一个算子**，用 `role` attr 区分：

- **source rank（role=0，dev0）**：持有 A/B/C。
  (a) AIC 计算 direct（非 remote）tiles → 本卡 C；
  (b) AIV comm core 把 remote wavefront 的 A/B staging **跨卡写入 compute 卡的 arena**、并置 AB signal；
  复用单卡的 `ProcessCommCopyIn / CopySignalAB / SignalForA/B / AbSignalIndex`，切分/顺序不变。
- **compute rank（role=1，dev1）**：AIC 等 AB signal（本地轮询 arena）→ 读 arena 的 staged A/B →
  计算 remote tiles → 用 `ComputeRemoteTileDirectD()` **直接写回 source 卡的 C**（`cGlobal_` 重定向到
  `peer_out`）。

**arena** 放在 compute 卡，承载 `abSignal / dSignal / aSlab / bSlab`，布局偏移与单卡 workspace **逐字节一致**
（两 rank 用同一 problem 跑同一 tiling，偏移天然对齐）。host 在 launch 前把 arena 清零（省去 in-kernel
ResetSignals），并在两卡各自 stream 上同步（省去 in-kernel D-done 轮询）——**这是与 CUDA 的一处简化，见下**。

## 本轮改动的文件

### 算子定义 `op_host/tilelet_matmul_fp32_def.cpp`
- 新增可选输入 `arena`（peer 可见共享 staging+signal 缓冲）、`peer_out`（compute rank = source 的 C 基址）。
- 新增可选 attr `role`(0=source,1=compute,默认0)、`rank_id`、`world_size`。
- **缺省不传新输入 / role=0 时行为与单卡完全一致**（回归安全）。
- 生成的 aclnn 签名（顺序）：
  `aclnnTileletMatmulFp32GetWorkspaceSize(x1, x2, bias, arena, peer_out, transposeX1, transposeX2,`
  `remoteTileStart, remoteTileCount, wavefrontM, wavefrontN, commCoreNum, commKTiles, enableDCopyback,`
  `role, rankId, worldSize, out, workspaceSize, executor)`。

### Host tiling `op_host/tilelet_matmul_fp32_tiling.cpp`
- 解析 attr 9/10/11（role/rank_id/world_size），填入 tiling data，并输出 `arenaBytes`（= 单卡 workspace 的
  staging+signal 布局大小，供 host 驱动分配 arena）。

### Kernel tiling data `op_kernel/tilelet_matmul_fp32_tiling_data.h`
- `TileletMatmulFp32TileletInfo` 增加 `role / rankId / worldSize / arenaBytes`。

### Kernel `op_kernel/tilelet_matmul_fp32.cpp` + `op_kernel/tilelet_matmul_fp32_base_kernel.h`（核心）
- 入口 kernel 签名新增 `arenaGM / peerOutGM`（在 biasGM 与 cGM 之间），透传给 `Init/InitInputs`。
- `InitInputs`：`crossCard_ = (arenaGM != nullptr)`；跨卡时 signal/slab 的 `SetGlobalBuffer` 基址由本卡
  workspace 改为 `arenaGM + offset`；compute rank（role=1 且有 peer_out）把 `cGlobal_` 重定向到 `peerOutGM`。
- `Process`：`crossCard_` 时按 role 分派 `ProcessCrossCardSource()` / `ProcessCrossCardCompute()`；否则走原单卡逻辑。
- 新增 `ProcessCrossCardSource`（AIV comm 走 `ProcessCommCopyIn`，AIC 走 `ProcessComputeCoreDirect`）、
  `ProcessCrossCardCompute`（AIC：等 AB signal → `ComputeRemoteTileDirectD` 写 peer C）。
- `bl1_fullload_kernel.h` 的 `Init` 也加了 `arenaGM/peerOutGM` 形参（仅签名对齐，单卡全载路径忽略）。

### 两卡测试驱动 `examples/src/main_crosscard.cpp`（新增，非 vLLM）+ `examples/src/CMakeLists.txt`
- 自包含两卡驱动：开 dev0/dev1、双向 `aclrtDeviceEnablePeerAccess`、每卡一个 stream；
  在 compute 卡分配并清零 arena、在 source 卡分配 C；把 arena/peer_out 地址 + role 作为新输入/attr 传给两次
  aclnn 调用（dev0=source role0，dev1=compute role1）；并发 launch 后同步两 stream；读回 source 的 C 写
  `output_z.bin`。
- 复用 `examples/scripts/gen_data.py` / `verify_result.py`（BF16 golden）；`gen_data.py` 支持
  `TILELET_M/N/K`、`TILELET_DTYPE`、`TILELET_TRANSPOSE_X2`、`TILELET_USE_BIAS`。
- CMake 里新增可执行 `execute_tilelet_matmul_fp32_crosscard`。
- 顺带把单卡驱动 `examples/src/op_runner.cpp` 的 aclnn 调用改成新签名（arena/peer_out 传 nullptr，role=0）。

### 构建脚本修复（在 `ops-nn/scripts/`，对 910C 依然需要，除非 CANN 版本已修）
1. `ops-nn/scripts/kernel/binary_script/build_binary_opc_gen_task.sh`：对 tilelet 强制 `thread_num=1`
   （单编译分片）。否则该 op 的 bin 会被切成多分片但只有 1 个编译 worker，**丢掉 bf16 分片**。
2. `ops-nn/scripts/util/build_opp_kernel_static.py`：`shell_exec` 原来是 `subprocess.Popen(cmd, shell)`
   （把 `shell` 当成 `bufsize` 传），在较新 Python 下导致 objcopy `Bad file descriptor`；改为
   `subprocess.run(cmd, shell=shell, check=True)`。

## 构建与运行（完整流程）

以 `ops-nn` 为当前目录。SoC 名按 910C 工具链实际识别为准（本文按 910B 写作 `ascend910b` /
`Ascend910B1`，910C 换成对应名字）。**注意事项见下节**。

```bash
# 0) 环境：设临时目录到数据盘（勿用根分区 /）
export TMPDIR=<data-disk>/tmp   # 自建；不要用 /tmp

# 1) 配置
cmake -S . -B build

# 2) kernel bin
cmake --build build --target ascendc_bin_ascend910b_tilelet_matmul_fp32 -j8
cmake --build build --target binary -j8              # 应产出 fp32+bf16 两套 bin

# 3) 静态打包（-b 必须是绝对路径，见注意事项）
python3 scripts/util/build_opp_kernel_static.py StaticCompile          -s ascend910b -b <abs>/build -n 1 -a aarch64
python3 scripts/util/build_opp_kernel_static.py GenStaticOpResourceIni  -s ascend910b -b <abs>/build
cmake --build build --target cann_nn_static -j8

# 4) examples（两卡驱动）
export DDK_PATH=<cann>/aarch64-linux NPU_HOST_LIB=<cann>/aarch64-linux/devlib
cmake -S experimental/matmul/tilelet_matmul_fp32/examples/src \
      -B experimental/matmul/tilelet_matmul_fp32/examples/build
cmake --build experimental/matmul/tilelet_matmul_fp32/examples/build -j8

# 5) 生成 BF16 数据 + golden
cd experimental/matmul/tilelet_matmul_fp32/examples
env TILELET_DTYPE=bf16 TILELET_USE_BIAS=0 TILELET_TRANSPOSE_X2=1 python3 scripts/gen_data.py

# 6) 跑两卡 gate（ASCEND_RT_VISIBLE_DEVICES 必须显式设两张卡）
cd output
env LD_LIBRARY_PATH=<cann>/aarch64-linux/lib64:<driver>/lib64:<driver>/lib64/driver:<driver>/lib64/common:$LD_LIBRARY_PATH \
    ASCEND_OPP_PATH=<cann>/opp ASCEND_RT_VISIBLE_DEVICES=0,1 \
    TILELET_DTYPE=bf16 TILELET_USE_BIAS=0 TILELET_TRANSPOSE_X2=1 \
    TILELET_M=256 TILELET_K=64 TILELET_N=512 TILELET_WAVEFRONT_M=16 TILELET_WAVEFRONT_N=8 \
    TILELET_COMM_CORE_NUM=8 TILELET_COMM_K_TILES=32 \
    TILELET_REMOTE_TILE_START=2 TILELET_REMOTE_TILE_COUNT=2 \
    ./execute_tilelet_matmul_fp32_crosscard
# 校验
cd ..
env TILELET_DTYPE=bf16 python3 scripts/verify_result.py output/output_z.bin output/golden.bin   # 期望 test pass
```

单卡自检：把 `ASCEND_RT_VISIBLE_DEVICES=0` 并跑 `./execute_tilelet_matmul_fp32_op`（同样应 `test pass`）。

### 构建注意事项（踩过的坑，910C 大概率复现）

- **`ASCEND_RT_VISIBLE_DEVICES` 必须显式设**（如 `0,1`），否则 acl runtime 报 `device count = 0`
  （`npu-smi` 能看到卡不代表进程能看到）。
- **StaticCompile / GenStaticOpResourceIni 的 `-b` 必须是绝对路径**：脚本内部 `cd <输入目录> && objcopy ...
  <相对输出>`，相对 build 路径会把输出解析到不存在的嵌套目录，报 `Bad file descriptor` / `cannot open`。
- 静态打包会给每个嵌入 json 注入 `filePath` 字段；若绕过脚本手工嵌入会漏掉它，运行期
  `op_binary_resource_manager AddBinary: filePath ... null` → aclnn `161001`。**一定走脚本**。
- 改了 `op_host/*_def.cpp` 的输入/attr 后，**必须重跑 `cmake -S . -B build`**：aclnn 接口只在 configure
  时从 def 重生成，单独 build 目标不会刷新（日志出现 `Opbuild generating sources - done` 才算刷新）。
- 若 `binary` 报某 dtype 分片 `not generate output json`，就是上面的 `thread_num` 分片问题（确认补丁生效）。
- 清理该 op 的构建产物时注意：opc 生成的目录可能是怪权限（`d-wxr-xr-T`），先 `chmod -R u+rwX` 再删；
  别把 `build/tbe/config/<soc>/tilelet_matmul_fp32` 误建成文件。

## 已跨卡验证（910B，仅正确性）

两卡 gate（dev0=source / dev1=compute，BF16 / no-bias / transpose_x2 / direct-D），`verify_result.py` 均
**error ratio 0.0000, test pass**：

- 默认目标配置：`M=256,N=512,K=64, remote=[2,4), comm_sms=8, comm_k_tiles=32`。
- 多 wavefront：`M=2176,N=2304,K=8, remote=[0,384), wavefront=16x8, comm_sms=8, comm_k_tiles=32`，
  覆盖 `sig0=A0+B0`、后续 A-row signal、后续 B-col signal，以及 source/compute 分工同时存在。
- 多 K-group：`M=256,N=512,K=128, remote=[2,4), comm_sms=8, comm_k_tiles=1`。
- 单卡目标配置也复核通过：`M=256,N=512,K=64, remote=[0,4), comm_sms=8, comm_k_tiles=32`。

即：kernel 跨卡 `DataCopy`、跨卡 AB signal 轮询、compute 直写 source C 三者都成立；多 wavefront / 多 K-group
的 copy 顺序与 signal 索引在跨卡下已做正确性覆盖（不只是单 wavefront 平凡情形）。

---

# 第三阶段：910C 落地（本轮，2026-07）

机器换到 **910C**（SoC `Ascend910_9362`，工具链短名 `ascend910_93` / 长名 `Ascend910_9391`，CANN 8.5.2；
每 die 20 个 AIC core；16 die，P2P 全通）。本轮完成：

- **910C 构建跑通**。文档原来的纯 cmake 流程在干净树上不成立（experimental 算子要 `build.sh` 先做
  PREPROCESS 配置生成依赖表）。正确配方：
  `bash build.sh --static --experimental --soc=ascend910_93 --ops=tilelet_matmul_fp32 --vendor_name=custom -j8`
  （产出 fp32+bf16 两套 kernel bin + `build/bin_tmp/ascend910_93/libcann_nn_static.a` + resource cpp）。
- **examples 两处 910C 适配**（已改）：`examples/src/CMakeLists.txt` 的 `ascend910b` 路径改为
  `-DTILELET_SOC`（默认 `ascend910_93`）；配置 examples 时必须加 `-DCMAKE_SKIP_BUILD_RPATH=ON`，否则 RPATH
  烧进 `aarch64-linux/devlib` 的 **stub `libascend_hal.so`**，运行期 `aclInit` 报 `init soc version failed`。
  运行环境用 `set_env.sh` + `LD_LIBRARY_PATH=<ops-nn>/build:...`，**不要**再前置 `aarch64-linux/lib64`。
- **单卡 + 两卡 gate 在 910C 全过**（BF16 / transpose_x2 / direct-D，`error ratio 0.0000, test pass`）：
  默认 256×512×64、多 wavefront 2176×2304×8（remote=[0,384)）、多 K-group 256×512×128（`comm_k_tiles=1`）。
- **任务2 部分完成**：`gen_data.py` 的 golden 改写为 numpy（与旧纯 Python golden **逐字节一致**，快 ~24×）；
  `main_crosscard.cpp` 的 arena 改为按 **算子精确 `arenaBytes`** 分配（新增头
  `examples/inc/tilelet_host_layout.h`，与 `InitTileletInfo` 逐字节对齐，已用算子 `userWs` 日志校验：
  256×512×64 kt32→8922624、kt1→8923136）。两卡 gate 用动态 arena（9.5MB / 33MB）复跑通过。
- **任务4 完成**：`tilelet_host_layout.h` 内实现 `MapRatioToRemoteTiles`（镜像 CUDA `ScheduleRemoteLinear`
  + `ClampRemoteCtasByRatio`：按两卡 AIC 数平衡 remote/local wavefront → CTA → `[ceil(total·min),
  floor(total·max)]` 截断，remote 取 compact-slot 前缀 start=0）。`main_crosscard.cpp` 加
  `TILELET_MIN/MAX_REMOTE_CTA_RATIO`(+`TILELET_SOURCE_SM/COMPUTE_SM`)：设了就用 ratio 推导 remote 范围。
  实测 `ratio[0.0,0.4]` src=cmp=20 → 2176×2304×8 得 `remote=[0,204)`（4 wavefront 平衡取 2，40% 截断），`test pass`。
- **任务3「in-kernel 跨卡 D-done」完成**：新增可选输入 `peer_dsignal`（**放在 source 卡**、对 compute 卡
  peer 可见）。compute rank 每写完一块 remote C（peer 写 source）后 `PipeBarrier<PIPE_ALL>` + 写 D-done 到
  `peer_dsignal`（同为 source 卡，故与 C 写有序）；source rank 的 AIC core0 在 kernel 内轮询全部 remote slot
  的 D-done 再返回，于是 **source stream 完成即代表 C 组装完毕，不再依赖 host 同步做正确性**。缺省不传
  `peer_dsignal` 时回落到原 host 同步行为（回归安全，单卡/旧两卡路径 `test pass` 不变）。aclnn 签名新增
  `peer_dsignal`（在 `peer_out` 与 `transposeX1` 之间）；kernel 入口、`Init/InitInputs`、bl1 kernel、
  `op_runner.cpp` 均已对齐。

- **任务3「跨卡 copyback」完成**：`enable_d_copyback=true` 下，compute rank 的 **AIC** 把非直写
  wavefront 的 remote tile 写入本地 `dStage`（arena）并置 **compute-local** `dSignalGlobal_`（仍在 arena）；
  compute rank 的 **AIV** comm core 走 `ProcessCommCopyBack`：等该 signal → `CopyBackRemoteTile` 把 dStage
  跨卡 DMA 回 source C（`cGlobal_` 已重定向到 peer_out）→ 置 `peerDsignalGlobal_` 的 D-done。**最后一个
  remote wavefront 仍 direct-D**（`IsDirectDWavefront`，对齐 CUDA `direct_d_wavefront_begin=last,count=1`）。
  为此把 D 信号拆成两块：`dSignalGlobal_`（compute 卡 arena，AIC↔AIV 本地握手 + DCopyDone）与
  `peerDsignalGlobal_`（source 卡 peer_dsignal，跨卡「C tile ready」，direct-D 由 AIC 置、copyback 由 AIV 置）。
  缺省 `d_copyback=false` 时全部不触发。910C 实测通过：多 wavefront 2176×2304×8、copyback+多 K-group
  (kt=1,K=64)、copyback+ratio[0,0.6]，均 `error ratio 0.0000, test pass`；direct-D 与单卡回归不变。

**910C 仍缺（本轮受限）**：`msprof` 性能/overlap（任务1）与 vLLM 级大 shape 实测（任务2 尾巴）——本机 16 die
被生产 `vllm serve GLM-5.1` 占满 HBM（每 die ~62/64GB），只够跑小 shape 正确性 gate；且共享机上 profiling
噪声大、会挤生产任务。需空闲卡后再做。（vLLM 16384×28672×28672 经 `tilelet_host_layout.h` 估算 arena≈3.3GB。）

---

# 尚未完成 / 需要在 910C 上继续

按优先级（勾选为本轮已处理）：

1. **[已做，结论：提速 1.3–1.5×]** 910C 空闲卡循环稳态实测：**offload 更快**（4096³ 1.32×、8192³ 1.53×），
   overlap 有效、通信被盖住。关键修复 = 把跨卡 staging 拷贝改成大块跨步 `DataCopyPad`（详见「已实测结论」）。
   注意单次 msprof 会误判（冷加载 + 自旋等待），必须循环计时。

2. **[部分完成] 大 shape 跨卡实测**。✅ 已写快速 numpy golden、✅ arena 改为按 `arenaBytes` 动态分配、
   ✅ 已写扫描脚本 `examples/scripts/sweep_large_shapes.sh`。❌ vLLM 级大 shape（`scratch_max_m=16384`、
   `n=28672`、`k=28672`，arena≈3.3GB）实测仍缺——需空闲卡（每 die 需数 GB 空闲 HBM）。

3. **补齐 v1 相对 CUDA 的简化**：
   - ✅ **in-kernel 跨卡 D-done 轮询（本轮完成）**：见上「第三阶段」。经可选输入 `peer_dsignal`（source 卡、
     compute peer 可见）实现，source kernel 内轮询、不再依赖 host 同步做正确性；缺省不传则回落旧行为。
   - ✅ **copyback 路径（d_copyback=true）跨卡（本轮完成）**：见下「第三阶段」补充。compute AIC 把 remote tile
     写本地 dStage + 置 compute-local dSignal；compute AIV comm core 等该 signal 后把 dStage 跨卡 DMA 回
     source C 并置 D-done；**最后一个 remote wavefront 仍走 direct-D**（对齐 CUDA `direct_d_wavefront_begin`）。
     缺省 `d_copyback=false` 时该路径完全不触发（direct-D 与单卡回归均 `test pass`）。
   - ❌ **arena 分配器**：目前测试驱动直接 `aclrtMalloc` 一块 peer 可见 buffer（大小已按 `arenaBytes` 精确）；
     真接入要一个 peer 可见分配器（对应 CUDA same-VA allocator），处理生命周期与多算子复用——归入框架层。

4. **[已完成] ratio 调度映射**。见上「第三阶段」：`tilelet_host_layout.h::MapRatioToRemoteTiles` 镜像
   `ClampRemoteCtasByRatio`/`ScheduleRemoteLinear`，`main_crosscard.cpp` 已接受 `TILELET_MIN/MAX_REMOTE_CTA_RATIO`
   并实测通过。框架接入时复用同一头。

5. **[脚手架已搭，待硬件收尾] 框架接入层**。已在 `vllm-ascend` 里搭好接入脚手架（见
   `vllm-ascend/csrc/tilelet/README.md`）：`csrc/tilelet/` 下的 IPC peer-arena 分配器（ACL IPC-mem key API +
   `ENABLE_PEER_ACCESS`，即 CUDA same-VA allocator 的昇腾对应）、`tilelet_remote_linear{,_compute}` torch 算子
   （调自定义 aclnn 算子 `aclnnTileletMatmulFp32`）、IPC key export/import/close 算子；`vllm_ascend/tilelet/`
   下的 `maybe_tilelet_linear` 路由 + `TileletSession`（持久 scratch + rendezvous）+ ZMQ 控制通道 +
   `tilelet_compute_serve`（decode 侧 role=1 服务循环）；`vllm_ascend/ops/linear.py` 的
   `AscendUnquantizedLinearMethod.apply` 已插 fail-closed 钩子；`envs.py` 加了 `VLLM_ASCEND_TILELET_*` 开关。
   已做无卡编译就绪检查（两个 adapter 头 `g++ -fsyntax-only` 过、aclnn 签名与生成头一致、`csrc` 加入 include 路径）。
   **待空闲卡收尾**：首次完整编译、ZMQ 控制通道时延与 decode serve 循环的交错调优、以及端到端 PD benchmark 实测。
   注：ratio→tile 已复用同一份 `tilelet_host_layout.h`（C++ 与 `vllm_ascend/tilelet/layout.py` 两份，已对齐）。

# 空闲卡上的 profiling / 性能验证操作指南（交接）

**这是迁移到有空闲卡的机器后最该做的事**（tilelet 的价值 = overlap 提速，目前只验了正确性、没验提速）。
按序执行；速查清单见 `RESUME_ON_FREE_CARD.md`。**共享机注意**：跑完/被杀后务必
`pkill -9 -f execute_tilelet_matmul_fp32`；跨进程测试用**专用 die**（脏 teardown 会在复用 die 上泄漏 IPC 注册）。

## ✅ 已实测结论（2026-07-16，910C 空闲卡，同封装 die 0,1）

**offload 确实提速。** 用**循环稳态计时**（`TILELET_ITERS=20`，跳过 warmup，host 端 `clock_gettime`，
launch 双 rank + sync 双 stream 的端到端）：

| shape | 单卡 baseline（remote=0） | offload ratio0.4（batched copy） | 加速 |
|---|---|---|---|
| 4096³ | 1.507 ms | 1.143 ms | **1.32×** |
| 8192³ | 11.982 ms | 7.838 ms | **1.53×** |

- 8192³ offload 7.838ms ≈ 理想双卡 2-die 分工下界（0.6×12=7.2ms），说明**通信被计算基本盖住、overlap 有效**。
- 加速随 shape 增大（1.32×→1.53×），符合"固定开销被摊薄 + 更大 GEMM 更划算"。

**Meta-Llama-3-8B prefill 三个 linear（M=4096 tokens，disaggregated_prefill_benchmark 的模型），tilelet off vs on：**

| linear | off | on(ratio0.4) | 加速 |
|---|---|---|---|
| qkv_proj (N=6144,K=4096) | 2.235 ms | 1.636 ms | 1.37× |
| gate_up_proj (N=28672,K=4096) | 10.663 ms | 7.215 ms | 1.48× |
| down_proj (N=4096,K=14336) | 6.278 ms | 4.561 ms | 1.38× |
| 单层合计 | 19.18 ms | 13.41 ms | **1.43×** |
| ×32 层（仅 prefill linear） | 613.6 ms | 429.2 ms | **1.43×** |

即 **prefill 的 linear GEMM 部分提速 ~1.43×**（就是 disaggregated_prefill_benchmark 里 TTFT 改善的来源）。
**注意**：这是 prefill linear 的 GEMM 时间对比（off=单 die、on=卸载到第二 die ratio0.4），**不是端到端 serving TTFT**——
完整 vLLM PD serving benchmark 本机跑不了（无 vllm/vllm_ascend 安装、且 vllm-ascend main 要 CANN 9.0.1 而本机 8.5.2）。
这个数是"有一块空闲第二 die 时"的 prefill-linear 上界；真实 DP 里第二 die 是 decode rank 的 comm-bound 空窗，收益取决于空窗多少。

**关键的踩坑教训（务必看）：单次 `msprof` 的 `op_summary` Task Duration 极不可靠**——首次运行含 kernel 冷加载，
且 source 的 op 时长把 **in-kernel D-done 自旋等待**也算进去，导致同一配置在 4096³ 上从 5.6ms 抖到 12.8ms、
一度**误判为 3–8× regression**。**正确做法是循环稳态计时**（本仓 `main_crosscard.cpp` 已加 `TILELET_ITERS`）。

**性能来自的关键修复**：把跨卡 A/B staging 的拷贝从"16KB UB 逐行 `DataCopy`"改成"128KB UB + 多块跨步
`DataCopyPad`（一次搬一个 tile-block 的多行）"——对齐 CUDA tilelet 的 tile-block 拷贝（不是逐行）。见
`base_kernel.h` 的 `TILELET_MATMUL_FP32_COPY_UB_BYTES=128*1024` 与 `CopyGm2D` 的批量跨步分支。
（copyback d_copyback=true 对性能无额外帮助：瓶颈在 source 的 staging，两条路径共用同一 staging。）

正确性侧全部通过（见下「大 shape」）：4096³、8192×8192×2048、16384×8192×2048 跨卡 `test pass`（batched copy 后复验仍过）。

## 0. 前置：构建
按上文「构建与运行」用 `build.sh --static --experimental --soc=<对应 soc> --ops=tilelet_matmul_fp32` 出算子 +
examples（910C 上 soc 为 `ascend910_93`，两处 910C 适配见「第三阶段」）。

## 1. 已测的片间带宽（解读 profiling 的基准）
910C 上用 `aclrtMemcpy` D2D（SDMA/copy-engine）实测的**单向**片间带宽：
- **同封装两 die**（如 0↔1）：写 ~191 GB/s、读 ~202 GB/s；双向合计 ~327 GB/s。
- **跨封装**（走 HCCS，如 0↔2）：写 ~138 GB/s、读 ~165 GB/s；双向合计 ~253 GB/s。
- **IPC-import 的 peer buffer（我们实际用的机制）带宽与普通 peer 完全一致**（191.87 / 137.86 GB/s 写），
  即 IPC-key vs CUDA same-VA **不损带宽**（差别只在地址空间，不在带宽）。
- 参考：本地 HBM copy ~620 GB/s（≈1.2TB/s 有效），故跨 die 约为本地的 11–16%。

结论：**PD 部署把 prefill die 与 compute(decode) die 放同一封装内**（0-1/2-3/…/6-7），带宽最优（比跨封装高 ~40%）。
**注意**：以上是 SDMA 链路上限；tilelet kernel 实际走 **AIV `DataCopy` peer-store**，受同一链路约束但可能低于 SDMA 峰值——
**这条 kernel 路径的带宽尚未单独测**（要么写个专用 kernel，要么在真算子 profiling 里看 HCCS 吞吐）。

## 2. 单算子 overlap / 性能 profiling（第一优先级）
用 `examples/scripts/profile_crosscard.sh`（msprof 封装，参数化 shape）：
```
cd experimental/matmul/tilelet_matmul_fp32/examples
TILELET_M=4096 TILELET_N=4096 TILELET_K=4096 \
TILELET_MIN_REMOTE_CTA_RATIO=0.0 TILELET_MAX_REMOTE_CTA_RATIO=0.4 \
bash scripts/profile_crosscard.sh <srcDev> <cmpDev>
```
看 `output/msprof_result` 要回答的关键问题：
- **AIV comm core 的拷贝是否与 AIC compute core 的 matmul 在时间上重叠**（tilelet 立身之本）；
- **HCCS 实际吞吐** vs 上面的链路上限（判断是否被带宽卡住）；
- **端到端 duration** vs 单卡全算基线（同脚本设 `TILELET_REMOTE_TILE_COUNT=0` 即不 offload，做对照）；
- source(direct tiles) 与 compute(remote tiles) 两卡是否真并行。

## 3. vLLM 级大 shape 正确性实测
`bash examples/scripts/sweep_large_shapes.sh <srcDev> <cmpDev>`（含逼近 vLLM 的 shape；numpy golden 已快、arena 动态）。
需两 die 各有数 GB 空闲 HBM。同时复核大 K 下 BF16 累加误差（唯一没法纯代码核对的点）。

## 4. PD 分离 benchmark（真实提速）
按 `vllm-ascend/csrc/tilelet/README.md`：装自定义算子包（`build.sh --pkg ...`）→ 重编 vllm-ascend →
补齐并验证 ZMQ 控制通道 + decode serve 循环 → 同节点起两实例（prefill=source、decode=compute，
`VLLM_ASCEND_TILELET_*`，两 die 同封装）→ 测 prefill TTFT/吞吐 vs `VLLM_ASCEND_TILELET_ENABLE=0`。

---

## 与 CUDA tilelet 的逐行比对结论（2026-07 已完成）

已把下列点**逐行**对照 `tilelet-sched/cutlass/...` 源码（此前只是数值验证）。结论：**核心机理逐行一致**，
差异都是正确性无关的（硬件/坐标系/调度策略）。每条给出源码位置。

**逐行一致：**
- **A/B signal 映射**：`SignalForA`=wfRow、`SignalForB`=wavefrontRows-1+wfCol（`base_kernel.h:439/448`）
  == `signal_for_a/b_wavefront` 非转置分支（`tilelet_gemm.h:199/210`、`tilelet_remote_copy.h:772/782`）。
- **signal 起点 / 需否拷贝**：`SignalOrigin`/`SignalNeededForRemoteTiles`（`base_kernel.h:458/474`）
  == `signal_wavefront_origin`/`signal_needed_for_wavefront_range`（`tilelet_remote_copy.h:792/814`）。
- **copy 顺序 / A行B列**：`ProcessCommCopyIn`+`CopySignalAB`（`wfCol==0` 拷 A、`wfRow==0` 拷 B）
  （`base_kernel.h:626/848/853`）== `ordered_sig` 循环 `need_a=(wf_col==0)`、`need_b=(wf_row==0)`
  （`tilelet_remote_copy.h:1140/1162/1163`）。
- **K-group signal 粒度**：`AbSignalIndex=sig*kGroupCount+group`、`kGroupCount=ceil(kTiles/commKTiles)`
  （`base_kernel.h:400`）== `signals[sig_idx*group_count+group]`、`group_count=ceil(k_tile/interval)`
  （`tilelet_gemm.h:450/453`）。等待逻辑 `waitSig=max(sigA,sigB)`（`base_kernel.h:507`）也一致。
- **完成→发布两阶段协议**：每 worker 写 completion、汇总后发 (sig,group) 信号
  （`base_kernel.h:638-644` vs `tilelet_remote_copy.h:1200-1215`）——结构一致。
- **compact slot → tile 线性化**：`DecodeCompactSlot`（groupM/N=2 分块 + 可整除/回退两分支，
  `base_kernel.h:288-337`）与 `get_wavefront_local_tile_offset`（`threadblock_swizzle.h:365-411`）
  **逐字节一致**（仅变量名 innerN↔local_n 之差）。

**差异（正确性无关，需知晓）：**
1. **remote tile 放置**：CUDA compute rank 可用 `reverse_wavefront_order` 取尾部 wavefront
   （`tilelet_multiprocess_gemm.cu:1207`、`tilelet_remote_gemm.h:1074`）；Ascend 用前缀
   `remote_tile_start=0..count`。→ **compact slot 编号完全一致**，只是"哪几块归 remote"的选择不同，
   切分大小与最终 C 相同；Ascend 未实现 `reverse_wavefront_order`（不需要）。
2. **transpose_signal_mapping**：Ascend 固定 false；CUDA remote 路径默认也 false。一致。
3. **区域内拷贝微观粒度**：CUDA per-thread `cp.async`/LDG 混合；Ascend per-row `DataCopy` 经 UB。
   覆盖相同、微观分工不同（硬件差异，预期）。
4. **direct-D 写回坐标系**：CUDA 列主序（`ldc=tilelet_m`，TN 交换坐标）；Ascend 行主序
   `output[M,N]`（`cGlobal_[mStart*N+nStart]`）。都产出正确 `output[M,N]`（golden 验证），设计上非逐字节可比。
5. **数值累加/取整**：两边都 FP32 累加；epilogue 硬件不同（cube vs tensor core），容差内等价。
   **唯一仍需真机大 shape 复核**：大 K 累加误差（受限于共享机 HBM，见上）。

**epilogue / 语义等价（2026-07 逐条核对）**：CUDA `remote_linear` 是**纯 GEMM**——`alpha=1.0`、
`beta=0.0`（`torch_bindings.cpp:1029-1030`）、无融合 activation/scale、输出 `[*,N]` 行主序、仅在
`bias=None` 时走 tilelet（`tilelet.py` 的 `_linear_eligibility_reason`）。Ascend 算子同为纯 matmul、
无 alpha/beta 缩放、bias 可选但 vLLM 路径不传、输出行主序 `[M,N]`。→ **功能语义逐条等价**，无遗漏的融合/缩放。

## 接口一致性（算子设置 vs CUDA tilelet）

CUDA 的功能开关分两层：**host 调度层**（`remote_linear(...)` 的 ratio / rank / prefix）和 **kernel 参数层**
（`TileletMultiprocessGemm::Arguments`：`compute_ctas`/`comm_sms`/`comm_k_tiles`/`enable_d_copyback`/`config`）。
Ascend 算子对应的是 **kernel 参数层**；ratio 那层是框架职责（CUDA 里也是 host 先把 ratio 折成 `cta_begin/count`
再进 kernel，见 `ScheduleRemoteLinear`）。逐项对照：

| 功能 | CUDA（kernel 层） | Ascend 算子 attr/input | 一致性 |
|---|---|---|---|
| **是否 copyback** | `enable_d_copyback`（默认 false） | `enable_d_copyback`（默认 false） | ✅ **名称/默认/语义完全一致** |
| comm 核数 | `comm_sms` | `comm_core_num`（默认 8） | ⚠️ 语义一致，名称不同（Ascend 用 AIV core，非 SM）；env 层已统一为 `TILELET_COMM_SMS` |
| K 分组 | `comm_k_tiles` | `comm_k_tiles`（默认 32） | ✅ 完全一致 |
| remote 块数 | `compute_ctas`（host 由 ratio 折算） | `remote_tile_start`/`remote_tile_count`（host 由 ratio 折算，见 `tilelet_host_layout.h`） | ✅ 结构一致（kernel 层都是具体块数，ratio 在 host） |
| wavefront 形状 | 模板 `WavefrontM/N`（16/8） | attr `wavefront_m`/`wavefront_n`（16/8） | ✅ 默认一致（Ascend 做成运行期 attr，更灵活） |
| 两 rank | `source_rank`/`compute_rank` | `role`/`rank_id`/`world_size` | ⚠️ 模型不同（Ascend 两次 launch 用 role 区分） |
| 布局/转置 | `GemmLayoutMode::kLinearTN` | `transpose_x1`/`transpose_x2` | ✅ `transpose_x2=1` 表达 `x1@x2^T` |
| dtype | bf16（vLLM 强制） | x1/x2/y dtype（bf16/fp32） | ✅ bf16 路径一致 |

结论：**决定"做什么"的功能开关（尤其 copyback、comm_k_tiles、wavefront、ratio 折算方式）与 CUDA 一致**；
唯一实质差异是命名 `comm_core_num`↔`comm_sms`（Ascend 上 SM 术语不准确，故保留 core 命名，env 层已用
`TILELET_COMM_SMS` 对齐）与两 rank 表达方式（role vs rank——源于 Ascend 两次单算子 launch 的设计）。
CUDA 特有的 `worker_green_sms`（green context 分核）无 Ascend 对应，属框架层性能旋钮。

## 当前边界（一句话）

**核心计算算子已在 910C 迁移完成、正确性全部验证，且实测提速。**（拷贝顺序/signal/slot 线性化与 CUTLASS
逐行一致，direct-D/copyback/in-kernel D-done/ratio→tile 齐全，多 wavefront/K-group 及大 shape 到
16384×8192×2048 均 `test pass`。）**2026-07-16 空闲卡循环稳态实测：offload 比单卡快 1.32×（4096³）、1.53×
（8192³）**，overlap 有效。关键修复：把跨卡 A/B staging 从 16KB UB 逐行拷贝改成 128KB UB 多块跨步 `DataCopyPad`
（对齐 CUDA 的 tile-block 拷贝）。教训：单次 msprof 会因冷加载+自旋等待误判，必须循环计时。**下一步**：vLLM
接入脚手架已就绪（`vllm-ascend/csrc/tilelet/`），可推进端到端 PD benchmark。
