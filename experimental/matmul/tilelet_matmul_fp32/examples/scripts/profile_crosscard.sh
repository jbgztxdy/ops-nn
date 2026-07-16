#!/bin/bash
# Purpose: msprof profiling harness for the tilelet two-card gate on 910C (the
#   #1 remaining task: overlap/performance). Runs the cross-card driver under
#   msprof so you can inspect AIC timeline, HCCS throughput, and whether the comm
#   cores (AIV, source rank) overlap in time with the compute cores (AIC, compute
#   rank), plus end-to-end duration. Ready to run the moment two dies are free.
# Usage:
#   bash profile_crosscard.sh [SRC_DEV] [CMP_DEV]
#   Env: TILELET_M/N/K, TILELET_REMOTE_TILE_START/COUNT, TILELET_COMM_CORE_NUM,
#        TILELET_COMM_K_TILES, TILELET_WAVEFRONT_M/N, TILELET_D_COPYBACK.
# Prereq: op + examples already built (see TILELET_ASCEND_MIGRATION.md); run from
#   the examples/ dir (needs scripts/ and output/).
set -e
SRC_DEV=${1:-0}
CMP_DEV=${2:-1}
: "${TILELET_M:=4096}"; : "${TILELET_N:=4096}"; : "${TILELET_K:=4096}"
: "${TILELET_COMM_CORE_NUM:=8}"; : "${TILELET_COMM_K_TILES:=32}"
: "${TILELET_WAVEFRONT_M:=16}"; : "${TILELET_WAVEFRONT_N:=8}"
: "${TILELET_REMOTE_TILE_START:=0}"; : "${TILELET_REMOTE_TILE_COUNT:=-1}"  # -1 = auto (use ratio driver instead)
: "${TILELET_D_COPYBACK:=0}"
: "${PROF_DIR:=$(pwd)/output/msprof_result}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$(cd ../../../../.. && pwd)/build:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$SRC_DEV,$CMP_DEV}
mkdir -p "$PROF_DIR"

echo "[profile] generating data for ${TILELET_M}x${TILELET_N}x${TILELET_K}"
env TILELET_DTYPE=bf16 TILELET_USE_BIAS=0 TILELET_TRANSPOSE_X2=1 \
    TILELET_M=$TILELET_M TILELET_N=$TILELET_N TILELET_K=$TILELET_K python3 scripts/gen_data.py

RUN_ENV="TILELET_DTYPE=bf16 TILELET_USE_BIAS=0 TILELET_TRANSPOSE_X2=1 \
  TILELET_M=$TILELET_M TILELET_N=$TILELET_N TILELET_K=$TILELET_K \
  TILELET_WAVEFRONT_M=$TILELET_WAVEFRONT_M TILELET_WAVEFRONT_N=$TILELET_WAVEFRONT_N \
  TILELET_COMM_CORE_NUM=$TILELET_COMM_CORE_NUM TILELET_COMM_K_TILES=$TILELET_COMM_K_TILES \
  TILELET_REMOTE_TILE_START=$TILELET_REMOTE_TILE_START TILELET_REMOTE_TILE_COUNT=$TILELET_REMOTE_TILE_COUNT \
  TILELET_D_COPYBACK=$TILELET_D_COPYBACK"

cd output
echo "[profile] msprof -> $PROF_DIR"
# aic-metrics/PipeUtilization exposes cube/vector pipe busy ratios (overlap);
# task-time + ascendcl + runtime-api give the timeline. Tune flags per CANN.
msprof --output="$PROF_DIR" \
       --aic-metrics=PipeUtilization \
       --ai-core=on --task-time=on --ascendcl=on --runtime-api=on \
       --application="env $RUN_ENV ./execute_tilelet_matmul_fp32_crosscard" || {
  echo "[profile] msprof failed; running once without profiler for a baseline timing:"
  time env $RUN_ENV ./execute_tilelet_matmul_fp32_crosscard
}
cd ..
echo "[profile] verify correctness of the profiled run:"
env TILELET_DTYPE=bf16 python3 scripts/verify_result.py output/output_z.bin output/golden.bin || true
echo "[profile] done. Inspect $PROF_DIR (msprof summary/timeline). Key questions:"
echo "  - do AIV comm-core copies overlap in time with AIC compute?"
echo "  - HCCS bytes/s achieved vs the ~191 GB/s (same-pkg) / ~138 GB/s (cross-pkg) link cap?"
echo "  - e2e duration vs a single-card full-compute baseline (remote_count=0)?"
