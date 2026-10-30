#!/usr/bin/env bash
# Host launch of the corrected factorial (plan v2 §B17), tag rev2. Detached; marker = "DONE e8 tag=rev2" then EXIT 0.
# Worker count 20 while the e3v2 run (24 workers) is still on the CPU (48 cores; thread vars = 1).
cd $(git rev-parse --show-toplevel)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=${PY:-python}
echo "START $(date -u +%FT%TZ) common.py=$(sha256sum test/experiments/revision/common.py | cut -c1-12) e8_factorial.py=$(sha256sum test/experiments/revision/e8_factorial.py | cut -c1-12) plan_v2=$(cut -c1-12 research/REVISION_PLAN_v2.sha256)"
$PY test/experiments/revision/e8_factorial.py --n-draws 10 --n-jobs ${E8_JOBS:-20} --tag rev2
echo "EXIT $? $(date -u +%FT%TZ)"
