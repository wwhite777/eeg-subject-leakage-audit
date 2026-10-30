#!/usr/bin/env bash
# Re-fetch the 1308 PhysioNet EEGBCI EDFs (109 subjects x runs 3-14) from the public S3 mirror into the MNE layout
# (data preparation notes 2026-09-16; the mirror is ~40x faster than physionet.org). 16 parallel curls, idempotent
# (existing files are kept). Marker = S3_FETCH_DONE_EXIT0 with the file count.
# Usage: bash test/env/fetch_eegbci_s3.sh > logs/s3_fetch_<ts>.log 2>&1   (run detached)
set -uo pipefail
ROOT=$(git rev-parse --show-toplevel)/data_public/eegbci_full/MNE-eegbci-data/files/eegmmidb/1.0.0
BASE=https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0
echo "START $(date -u +%FT%TZ) root=$ROOT"
mkdir -p "$ROOT"
LIST=$(mktemp "$ROOT/.fetch_list.XXXXXX")
for s in $(seq 1 109); do
  S=$(printf 'S%03d' "$s")
  for r in 3 4 5 6 7 8 9 10 11 12 13 14; do
    R=$(printf 'R%02d' "$r")
    echo "$S/$S$R.edf"
  done
done > "$LIST"
fetch_one() {
  local rel=$1
  local out="$ROOT/$rel"
  if [ -s "$out" ]; then echo "SKIP $rel"; return 0; fi
  if curl -s -f --retry 3 --retry-delay 2 --create-dirs -o "$out" "$BASE/$rel"; then echo "OK $rel"; else echo "FAIL $rel"; rm -f "$out"; return 1; fi
}
export -f fetch_one
export ROOT BASE
xargs -P 16 -I{} bash -c 'fetch_one "$@"' _ {} < "$LIST"
rm -f "$LIST"
N=$(find "$ROOT" -name 'S*R*.edf' -size +0 | wc -l)
echo "END $(date -u +%FT%TZ) files=$N"
if [ "$N" -eq 1308 ]; then echo "S3_FETCH_DONE_EXIT0 files=$N"; else echo "S3_FETCH_INCOMPLETE files=$N"; exit 1; fi
