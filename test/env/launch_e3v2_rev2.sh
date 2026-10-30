#!/usr/bin/env bash
# Host launch of the full E3a-v2/E3b-v2 run (plan v2 §B18–B19), tag rev2. Detached; marker = DONE e3v2 tag=rev2 in the log.
cd $(git rev-parse --show-toplevel)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=${PY:-python}
echo "START $(date -u +%FT%TZ) common.py=$(sha256sum test/experiments/revision/common.py | cut -c1-12) e3v2=$(sha256sum test/experiments/revision/e3v2_subject_info.py | cut -c1-12)"
$PY test/experiments/revision/e3v2_subject_info.py --n-jobs ${E3_JOBS:-12} --tag rev2
echo "EXIT $? $(date -u +%FT%TZ)"
