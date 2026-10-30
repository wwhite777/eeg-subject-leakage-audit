#!/usr/bin/env bash
# Host launch of the matched controls (plan v2 §B20–B21), tag rev2: classical chain (sequential) + three GPU chains (GPUs 1–3).
# Worker count for the classical chain is 24 while the E8 rerun (20 workers) shares the 48 cores. Markers: "CLASSICAL CHAIN DONE", "GPU<k> CHAIN DONE".
cd $(git rev-parse --show-toplevel)
PY=${PY:-python}; E=test/experiments/revision; TAG=rev2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
echo "START $(date -u +%FT%TZ) common.py=$(sha256sum $E/common.py | cut -c1-12) e1=$(sha256sum $E/e1_ladder_classical.py | cut -c1-12) e17=$(sha256sum $E/e17_fixed_test.py | cut -c1-12) e9=$(sha256sum $E/e9_deep_ladder.py | cut -c1-12) e10=$(sha256sum $E/e10_bci2a_deep.py | cut -c1-12) deep_common=$(sha256sum $E/deep_common.py | cut -c1-12) plan_v2=$(cut -c1-12 research/REVISION_PLAN_v2.sha256)"
run() { local name=$1; shift; echo "START $name $(date -u +%FT%TZ)" >> logs/rev2/chain_$CHAIN.log; "$@" > logs/rev2/$name.log 2>&1; echo "EXIT $name $? $(date -u +%FT%TZ)" >> logs/rev2/chain_$CHAIN.log; }
export -f run; export PY E TAG
CHAIN=classical setsid nohup bash -c '
export CHAIN=classical CUDA_VISIBLE_DEVICES=""
run e16_eegbci $PY $E/e1_ladder_classical.py --dataset eegbci --protocols trial_random_sc --n-seeds 5 --n-jobs 24 --tag $TAG
run e16_bci2a  $PY $E/e1_ladder_classical.py --dataset bci2a --protocols trial_random_6_sc,trial_random_2_sc --n-seeds 5 --n-jobs 24 --tag $TAG
run e17_eegbci $PY $E/e17_fixed_test.py --dataset eegbci --n-seeds 5 --n-jobs 24 --tag $TAG
run e17_bci2a  $PY $E/e17_fixed_test.py --dataset bci2a --n-seeds 5 --n-jobs 24 --tag $TAG
run e15_p0s    $PY $E/e15_stratification_artifact.py --protocols trial_random_sc --n-seeds 5 --tag $TAG
echo "CLASSICAL CHAIN DONE $(date -u +%FT%TZ)" >> logs/rev2/chain_classical.log
' > logs/rev2/chain_classical.out 2>&1 < /dev/null &
T12=imagery_left_right_fist,real_left_right_fist; T34=imagery_fists_feet,real_fists_feet; SEEDS=20260916,20260917,20260918
CHAIN=gpu1 setsid nohup bash -c '
export CHAIN=gpu1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2
run e16_deep_t12 $PY $E/e9_deep_ladder.py --tasks '$T12' --protocols trial_random_sc --seeds '$SEEDS' --gpu 0 --tag ${TAG}_t12
run e17_deep_t12 $PY $E/e9_deep_ladder.py --tasks '$T12' --fixed-test --seeds '$SEEDS' --gpu 0 --tag ${TAG}_fixed_t12
echo "GPU1 CHAIN DONE $(date -u +%FT%TZ)" >> logs/rev2/chain_gpu1.log
' > logs/rev2/chain_gpu1.out 2>&1 < /dev/null &
CHAIN=gpu2 setsid nohup bash -c '
export CHAIN=gpu2 CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
run e16_deep_t34 $PY $E/e9_deep_ladder.py --tasks '$T34' --protocols trial_random_sc --seeds '$SEEDS' --gpu 0 --tag ${TAG}_t34
run e17_deep_t34 $PY $E/e9_deep_ladder.py --tasks '$T34' --fixed-test --seeds '$SEEDS' --gpu 0 --tag ${TAG}_fixed_t34
echo "GPU2 CHAIN DONE $(date -u +%FT%TZ)" >> logs/rev2/chain_gpu2.log
' > logs/rev2/chain_gpu2.out 2>&1 < /dev/null &
CHAIN=gpu3 setsid nohup bash -c '
export CHAIN=gpu3 CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2
run e16_deep_bci2a $PY $E/e10_bci2a_deep.py --protocols trial_random_6_sc,trial_random_2_sc --seeds '$SEEDS' --gpu 0 --tag $TAG
run e17_deep_bci2a $PY $E/e10_bci2a_deep.py --fixed-test --seeds '$SEEDS' --gpu 0 --tag ${TAG}_fixed
echo "GPU3 CHAIN DONE $(date -u +%FT%TZ)" >> logs/rev2/chain_gpu3.log
' > logs/rev2/chain_gpu3.out 2>&1 < /dev/null &
sleep 3; echo "launched $(date -u +%FT%TZ)"
