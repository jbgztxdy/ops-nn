#!/bin/bash
# Purpose: Correctness sweep of the tilelet two-card gate across increasingly
#   large (toward vLLM-scale) shapes. The numpy golden is fast and the arena is
#   sized dynamically, so this just needs two free dies with enough HBM. Run when
#   cards free to close the "large-shape cross-card实测" item (doc task #2).
# Usage: bash sweep_large_shapes.sh [SRC_DEV] [CMP_DEV]
# Prereq: op + examples built; run from examples/ dir.
set -e
SRC_DEV=${1:-0}; CMP_DEV=${2:-1}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$(cd ../../../../.. && pwd)/build:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$SRC_DEV,$CMP_DEV}

# M N K  (ramp up; the last rows approach vLLM Meta-Llama-3-8B proj/mlp sizes).
# HBM per shape ~ (M*K + N*K + M*N)*2 + arena; ensure the two dies have room.
SHAPES=(
  "1024 4096 4096"
  "4096 4096 4096"
  "4096 14336 4096"
  "8192 4096 14336"
  "8192 28672 8192"
  "16384 28672 8192"
)
pass=0; fail=0
for s in "${SHAPES[@]}"; do
  read M N K <<< "$s"
  echo "=================== ${M}x${N}x${K} ==================="
  env TILELET_DTYPE=bf16 TILELET_USE_BIAS=0 TILELET_TRANSPOSE_X2=1 \
      TILELET_M=$M TILELET_N=$N TILELET_K=$K python3 scripts/gen_data.py
  ( cd output && timeout 600 env ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES \
      TILELET_DTYPE=bf16 TILELET_USE_BIAS=0 TILELET_TRANSPOSE_X2=1 \
      TILELET_M=$M TILELET_N=$N TILELET_K=$K TILELET_WAVEFRONT_M=16 TILELET_WAVEFRONT_N=8 \
      TILELET_COMM_CORE_NUM=8 TILELET_COMM_K_TILES=32 \
      TILELET_MIN_REMOTE_CTA_RATIO=0.0 TILELET_MAX_REMOTE_CTA_RATIO=0.4 \
      ./execute_tilelet_matmul_fp32_crosscard ) || { echo "RUN FAILED"; fail=$((fail+1)); continue; }
  if env TILELET_DTYPE=bf16 python3 scripts/verify_result.py output/output_z.bin output/golden.bin | grep -q "test pass"; then
    echo "  PASS"; pass=$((pass+1))
  else
    echo "  VERIFY FAILED"; fail=$((fail+1))
  fi
  pkill -9 -f execute_tilelet_matmul_fp32 2>/dev/null || true
done
echo "=================== sweep done: $pass pass, $fail fail ==================="
