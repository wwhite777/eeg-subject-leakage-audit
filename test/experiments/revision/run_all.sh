#!/usr/bin/env bash
# Launch the full revision experiment suite (REVISION_PLAN_v1.md) as detached chains:
# one CPU chain (classical analyses, sequential) and one chain per GPU. Each script writes its own
# log; the chain log records exit codes. Usage: bash run_all.sh <TAG>
set -u
TAG=${1:?tag}
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
E=test/experiments/revision
mkdir -p logs/$TAG
run() {  # run <name> <cmd...>  (sequential inside a chain; exit code recorded)
  local name=$1; shift
  echo "START $name $(date -u +%FT%TZ)" >> logs/$TAG/chain_$CHAIN.log
  "$@" > logs/$TAG/$name.log 2>&1
  echo "EXIT $name $? $(date -u +%FT%TZ)" >> logs/$TAG/chain_$CHAIN.log
}
export -f run
export TAG PY E

# ---------------- CPU chain
CHAIN=cpu setsid nohup bash -c '
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CHAIN=cpu
run e1_eegbci  $PY $E/e1_ladder_classical.py --dataset eegbci --n-seeds 5 --n-jobs 36 --tag $TAG
run e1_bci2a   $PY $E/e1_ladder_classical.py --dataset bci2a  --n-seeds 5 --n-jobs 36 --tag $TAG
run e2         $PY $E/e2_sweep.py        --n-seeds 5 --n-jobs 36 --tag $TAG
run e5         $PY $E/e5_classifiers.py  --n-seeds 5 --n-jobs 36 --tag $TAG
run e4         $PY $E/e4_pca_sweep.py    --n-seeds 5 --n-jobs 36 --tag $TAG
run e3         $PY $E/e3_subject_info.py --n-seeds 5 --n-jobs 36 --tag $TAG
run e6         $PY $E/e6_reference_target.py --n-jobs 36 --tag $TAG
run e8         $PY $E/e8_factorial.py    --n-draws 10 --n-jobs 36 --tag $TAG
run e13_classical $PY $E/e1_ladder_classical.py --dataset eegbci --preproc ztrial --n-seeds 5 --n-jobs 36 --skip-within --tag ${TAG}_ztrial
echo "CPU CHAIN DONE $(date -u +%FT%TZ)" >> logs/$TAG/chain_cpu.log
' > logs/$TAG/chain_cpu.out 2>&1 &
echo "cpu chain pid $!"

# ---------------- GPU chains
T12=imagery_left_right_fist,real_left_right_fist
T34=imagery_fists_feet,real_fists_feet
CHAIN=gpu0 setsid nohup bash -c '
export OMP_NUM_THREADS=4 CHAIN=gpu0 CUDA_VISIBLE_DEVICES=0
run e9_t12   $PY $E/e9_deep_ladder.py --tasks '$T12' --gpu 0 --tag ${TAG}_t12
run e12_t12  $PY $E/e12_calibration.py --tasks '$T12' --n-draws 3 --n-jobs 8 --gpu 0 --tag ${TAG}_t12
run e13_deep_t12 $PY $E/e9_deep_ladder.py --tasks '$T12' --preproc global --seeds 20260916 --gpu 0 --tag ${TAG}_global_t12
echo "GPU0 CHAIN DONE $(date -u +%FT%TZ)" >> logs/$TAG/chain_gpu0.log
' > logs/$TAG/chain_gpu0.out 2>&1 &
echo "gpu0 chain pid $!"

CHAIN=gpu1 setsid nohup bash -c '
export OMP_NUM_THREADS=4 CHAIN=gpu1 CUDA_VISIBLE_DEVICES=1
run e9_t34   $PY $E/e9_deep_ladder.py --tasks '$T34' --gpu 0 --tag ${TAG}_t34
run e12_t34  $PY $E/e12_calibration.py --tasks '$T34' --n-draws 3 --n-jobs 8 --gpu 0 --tag ${TAG}_t34
run e13_deep_t34 $PY $E/e9_deep_ladder.py --tasks '$T34' --preproc global --seeds 20260916 --gpu 0 --tag ${TAG}_global_t34
echo "GPU1 CHAIN DONE $(date -u +%FT%TZ)" >> logs/$TAG/chain_gpu1.log
' > logs/$TAG/chain_gpu1.out 2>&1 &
echo "gpu1 chain pid $!"

CHAIN=gpu2 setsid nohup bash -c '
export OMP_NUM_THREADS=4 CHAIN=gpu2 CUDA_VISIBLE_DEVICES=2
run e10      $PY $E/e10_bci2a_deep.py --gpu 0 --tag $TAG
run e7_shallow $PY $E/e9_deep_ladder.py --models shallow_fbcsp --widths 5,10,20,40,80 --protocols trial_random,subject_disjoint --seeds 20260916 --gpu 0 --tag ${TAG}_width_shallow
run e11_t12  $PY $E/e11_within_deep.py --tasks '$T12' --gpu 0 --tag ${TAG}_t12
echo "GPU2 CHAIN DONE $(date -u +%FT%TZ)" >> logs/$TAG/chain_gpu2.log
' > logs/$TAG/chain_gpu2.out 2>&1 &
echo "gpu2 chain pid $!"

CHAIN=gpu3 setsid nohup bash -c '
export OMP_NUM_THREADS=4 CHAIN=gpu3 CUDA_VISIBLE_DEVICES=3
run e7_eegnet $PY $E/e9_deep_ladder.py --models eegnet_v4 --widths 2,4,8,16,32 --protocols trial_random,subject_disjoint --seeds 20260916 --gpu 0 --tag ${TAG}_width_eegnet
run e11_t34  $PY $E/e11_within_deep.py --tasks '$T34' --gpu 0 --tag ${TAG}_t34
echo "GPU3 CHAIN DONE $(date -u +%FT%TZ)" >> logs/$TAG/chain_gpu3.log
' > logs/$TAG/chain_gpu3.out 2>&1 &
echo "gpu3 chain pid $!"
sleep 2
ps -eo pid,args | grep -E "revision/e[0-9]+_" | grep -v grep | awk '{print $1, $4, $5, $6}'
