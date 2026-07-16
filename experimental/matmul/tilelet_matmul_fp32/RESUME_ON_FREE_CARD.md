# Resume checklist — run these when NPU dies are free

Everything below was blocked only by the shared box being HBM-saturated by a prod
vLLM job (see the migration doc). All code is in place; this is the run/measure order.

## 0. Build (if not already)
```bash
cd ops-nn && export TMPDIR=<data-disk>/tmp
bash build.sh --static --experimental --soc=ascend910_93 --ops=tilelet_matmul_fp32 --vendor_name=custom -j8
CANN=/usr/local/Ascend/ascend-toolkit/latest
export DDK_PATH=$CANN/aarch64-linux NPU_HOST_LIB=$CANN/aarch64-linux/devlib
EX=experimental/matmul/tilelet_matmul_fp32/examples
cmake -S $EX/src -B $EX/build -DTILELET_SOC=ascend910_93 -DCMAKE_SKIP_BUILD_RPATH=ON && cmake --build $EX/build -j8
```

## 1. Sanity gates (small; ~any free die pair)
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$(pwd)/build:$LD_LIBRARY_PATH
cd $EX && env TILELET_DTYPE=bf16 TILELET_USE_BIAS=0 TILELET_TRANSPOSE_X2=1 python3 scripts/gen_data.py
# single card + two card (dev 0,1) — expect "test pass" (see migration doc for full cmds)
```

## 2. #1 PRIORITY — overlap / performance profiling
```bash
cd $EX
TILELET_M=4096 TILELET_N=4096 TILELET_K=4096 \
TILELET_MIN_REMOTE_CTA_RATIO=0.0 TILELET_MAX_REMOTE_CTA_RATIO=0.4 \
bash scripts/profile_crosscard.sh 0 1
```
Look at `output/msprof_result`: AIC timeline, comm(AIV)↔compute(AIC) time overlap, HCCS
GB/s vs the measured link caps (same-pkg ~191 / cross-pkg ~138 GB/s unidir, ~327/253 bidir),
and e2e vs a single-card baseline (rerun with `TILELET_REMOTE_TILE_COUNT=0`).

## 3. Large-shape correctness sweep (doc task #2 tail)
```bash
cd $EX && bash scripts/sweep_large_shapes.sh 0 1   # needs 2 dies with several GB free each
```

## 4. vLLM-Ascend PD-disaggregated integration (see vllm-ascend/csrc/tilelet/README.md)
1. Install the tilelet op as a custom aclnn package:
   `bash build.sh --pkg --experimental --soc=ascend910_93 --ops=tilelet_matmul_fp32 --vendor_name=custom`
   then run the produced `.run`.
2. Rebuild vllm-ascend (`pip install -e .`); fix any first-compile issues in `csrc/tilelet/`.
3. Finish + validate the two scaffold pieces on hardware: the source→compute ZMQ control
   channel and the decode-side `tilelet_compute_serve` interleaving.
4. Launch a same-node PD run with `VLLM_ASCEND_TILELET_ENABLE=1` (source on prefill,
   compute on decode) and measure prefill TTFT/throughput vs `=0`.

## Safety on the shared box
- Always `pkill -9 -f execute_tilelet_matmul_fp32` after a killed/hung run (frees HBM).
- Verify the prod vLLM job is unaffected (`ps aux | grep 'vllm serve'`, `npu-smi info`).
- Repeatable cross-process runs need DEDICATED dies (unclean IPC teardown can leak
  registrations on reused dies — see the de-risk note).
