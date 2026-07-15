# Tilelet Ascend Migration Notes

本文记录 `TileletMatmulFp32` 算子从 CUDA/CUTLASS tilelet kernel 迁移到 Ascend-C/ACLNN 的工作。
读者是接手在 **910C** 上继续的 agent：请先读本文，再读代码，再动手。

路径一律按 workspace 根目录书写（`ops-nn/...`、`vllm/...`、`tilelet-sched/cutlass/...`，脚本用
`ops-nn/scripts/...`），**不要**写机器绝对路径。凡是临时目录，放在数据盘（大盘）下自建目录，
不要用根分区 `/`（本机 `/` 已满，见"构建注意事项"）。

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
- CUDA 跨卡框架层（**未移植**，接入时要对照）：`tilelet-sched/src/tilelet_same_va_allocator.cu`、
  `tilelet-sched/src/tilelet_multiprocess_gemm.cu`。
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
- 复用 `examples/scripts/gen_data.py` / `verify_result.py`（BF16 golden）。
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

两卡 gate（dev0=source / dev1=compute，BF16 / no-bias / transpose_x2 / comm_sms=8 / comm_k_tiles=32 /
direct-D），`verify_result.py` 均 **error ratio 0.0000, test pass**：

- `remote=[2,4)`：source 本地算 tile 0,1 + compute 跨卡算 tile 2,3 并 peer 写回。
- `remote=[0,4)`：全 offload，compute 算全部、source 只 staging。
- `remote=[0,0)`：退化本地。
- 强制多 wavefront：`wavefront=1x1`（4 个独立 wavefront）、`2x1`（跨列 `SignalForB`）——均 pass。
- 强制多 K-group：`comm_k_tiles=1`——pass。

即：kernel 跨卡 `DataCopy`、跨卡 AB signal 轮询、compute 直写 source C 三者都成立；多 wavefront / 多 K-group
的 copy 顺序与 signal 索引在跨卡下也正确（不只是单 wavefront 平凡情形）。

---

# 尚未完成 / 需要在 910C 上继续

按优先级：

1. **性能 / overlap 验证（最关键，910B 没做）**。目前只在低带宽 910B 验了**正确性**。tilelet 的价值是
   通信与计算 overlap 带来的提速，必须在 910C 真机 profiling：AIC timeline、HCCS 吞吐、comm core 与
   compute core 是否时间重叠、端到端 duration。

2. **大 shape 跨卡实测**。只验证了 256×512×64（外加强制多 wavefront/K-group 的参数变体）。vLLM 真实 shape
   很大（`scratch_max_m=16384`、`n=28672`、`k=28672`）。阻碍：`examples/scripts/gen_data.py` 的 golden 是
   纯 Python 三重循环，大 shape 极慢——**先写一个快速 golden（numpy / 设备端参考）**，再放大 M/N/K
   （`gen_data.py` 目前 M/N/K 硬编码在文件头，需要改成可配置），同时把 `main_crosscard.cpp` 里固定 512MB 的
   arena 改为按算子报出的 `arenaBytes` 分配。

3. **补齐 v1 相对 CUDA 的简化**（当前对 benchmark 路径功能等价，但不同构）：
   - **in-kernel 跨卡 D-done 轮询**：现在靠 host 同步两 stream；CUDA 是 compute rank 发 D-done、source 在
     kernel 内等（`ComputeRemoteTileDirectD` 完成后 compute 无 D-done 写；`ProcessCommCopyBack`/
     `WaitCopyBackWavefrontDone` 是单卡 copyback 路径的遗留）。真接入若 source 不能靠 host 同步就往下走，
     需要把 D-done 做成跨卡可见并在 source kernel 内等待。
   - **copyback 路径（d_copyback=true）跨卡未验证**：只做了 direct-D。若 benchmark 之外要支持 copyback，
     `CopyBackRemoteTile / CopyBackRemoteWavefront / dStage` 需要在跨卡布局下重新走通。
   - **arena 分配器**：目前测试驱动直接 `aclrtMalloc` 一块 peer 可见 buffer；真接入要一个 peer 可见分配器
     （对应 CUDA 的 same-VA allocator），并处理生命周期与多算子复用。

4. **ratio 调度映射**。CUDA 用 `min/max_remote_cta_ratio`（见 `torch_bindings.cpp` 的
   `ClampRemoteCtasByRatio`）动态决定 remote CTA 数；Ascend 侧用显式 `remote_tile_start/count`。接入时要在
   框架层把 ratio 调度结果映射为这两个 attr。

5. **框架接入层（非平凡）**。要跑进 Ascend vLLM，需要：peer 可见 arena 分配器 + 两 rank launch 编排
   （对应 `tilelet_multiprocess_gemm.cu`）+ ratio→tile 映射 + peer access 使能 + 接到 `vllm/vllm/tilelet.py`
   的 NPU `remote_linear`。这是一层真实框架代码，不是改几个参数。

## 需要进一步和 CUDA tilelet 对齐 / 核对的点

在 910C 上建议逐条对照 `tilelet-sched/cutlass/...` 源码核实（本轮是复用单卡机制 + 端到端数值验证，
未逐字节比对 CUDA）：

- **copy 顺序**：`cutlass/include/cutlass/gemm/device/tilelet_remote_copy.h` 的 `ordered_sig` 遍历、
  A-row / B-col 的先后与切分，对照 `base_kernel.h` 的 `ProcessCommCopyIn` / `CopySignalAB` /
  `SignalOrigin` / `SignalForA` / `SignalForB`。
- **K-group signal 粒度**：`cutlass/include/cutlass/gemm/kernel/tilelet_gemm.h` 里 `k_iteration_signals`、
  `group_count`、`wait_sig` 的算法，对照 `AbSignalIndex` / `kGroupCount` / `WaitAbSignalForWavefront`。
- **remote CTA 范围**：CUDA `tilelet_gemm.h` 用 `cta_begin/cta_count/wavefront_begin`、
  `reverse_wavefront_order` 切范围；Ascend 用 `remote_tile_start/count` + `DecodeCompactSlot`。确认
  compact slot 的编号顺序（`groupM/groupN=2` 分块）与 CUDA 的 tile 线性化一致。
- **direct-D 写回**：CUDA `params_direct_D` vs `ComputeRemoteTileDirectD` 写 `cGlobal_`(=peer_out) 的
  offset（`mStart*N + nStart`）。
- **数值累加/取整**：BF16 输入、FP32 累加的取整与 CUDA epilogue 是否一致（当前 golden 是 FP32 参考，
  容差 1e-3 通过；大 shape 下需复核累加误差）。

## 当前边界（一句话）

**核心计算算子迁移已完成并在 910B 跨卡验证通过**（能跑、真跨卡、拷贝顺序/signal 逻辑与 CUTLASS 对齐、
多 wavefront/K-group 均过、覆盖 benchmark 核心配置）。**整体迁移尚未完成**：缺性能/overlap 验证、大 shape
实测、几处 v1 简化补齐、ratio→tile 映射、以及一层跨卡编排/框架接入。
