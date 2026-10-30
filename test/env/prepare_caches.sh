#!/usr/bin/env bash
# Rebuild the epoch caches after the raw EDFs are in place (data preparation notes 2026-10-02):
# six parallel EEGBCI chunks (as on 2026-09-16) + BCI-IV-2a via MOABB (network). Then compare against the
# 2026-10-02 hash list. Marker = CACHES_DONE with the npz count. Usage: bash test/env/prepare_caches.sh > logs/prepare_caches_<ts>.log 2>&1
set -uo pipefail
cd $(git rev-parse --show-toplevel)
PY=${PY:-python}
TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "START $TS"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
pids=()
for range in "1 18" "19 36" "37 54" "55 72" "73 90" "91 109"; do
  set -- $range
  $PY test/experiments/eegbci_full/eegbci_full_dataset_prepare.py --root data_public/eegbci_full --first-subject "$1" --last-subject "$2" \
     > "logs/eegbci_prepare_${1}_${2}_$TS.log" 2>&1 &
  pids+=($!)
done
$PY test/experiments/bci2a/bci2a_dataset_prepare.py --out-dir data_public/bci2a > "logs/bci2a_prepare_$TS.log" 2>&1 &
pids+=($!)
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "prepare exits: rc=$rc"
# the per-chunk runs each overwrite the summary json; rebuild one summary over all 109 subjects (cache hits only, fast)
$PY test/experiments/eegbci_full/eegbci_full_dataset_prepare.py --root data_public/eegbci_full --first-subject 1 --last-subject 109 > "logs/eegbci_prepare_summary_$TS.log" 2>&1 || rc=1
N=$(ls data_public/eegbci_full/cache/*.npz 2>/dev/null | wc -l)
echo "eegbci npz=$N (expected 436); bci2a: $(ls -la data_public/bci2a/cache/ 2>/dev/null | tail -1)"
( cd data_public && sha256sum -c cache_sha256_20261002T0646Z.txt > "../logs/cache_hash_check_$TS.log" 2>&1; echo "hash check exit=$? (OK lines: $(grep -c ': OK' ../logs/cache_hash_check_$TS.log), FAILED: $(grep -c 'FAILED' ../logs/cache_hash_check_$TS.log))" )
echo "END $(date -u +%FT%TZ)"
if [ "$rc" -eq 0 ] && [ "$N" -eq 436 ] && [ -s data_public/bci2a/cache/bci2a_leftright.npz ]; then echo "CACHES_DONE npz=$N"; else echo "CACHES_INCOMPLETE npz=$N rc=$rc"; exit 1; fi
