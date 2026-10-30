#!/usr/bin/env bash
# Rebuild the analysis environment from the pinned freeze (test/env/requirements.freeze.txt).
# Usage: bash test/env/rebuild_venv.sh > logs/venv_rebuild_<ts>.log 2>&1   (run detached; marker = VENV_REBUILD_DONE_EXIT0)
set -euo pipefail
VENV=${VENV:-.venv}
FREEZE=$(dirname "$0")/requirements.freeze.txt
echo "START $(date -u +%FT%TZ) venv=$VENV freeze=$FREEZE"
test -f "$FREEZE"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12.12 "$VENV"
fi
# personal venv only (rules C2): activate it and install through its interpreter; both indexes are trusted, so
# uv may take each pin from whichever index carries it (the 2026-10-30 first attempt failed on certifi otherwise)
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 -r "$FREEZE"
"$VENV/bin/python" - <<'PY'
import torch, mne, moabb, braindecode, sklearn, pyriemann, scipy, numpy
print("VENV OK torch", torch.__version__, "mne", mne.__version__, "sklearn", sklearn.__version__,
      "scipy", scipy.__version__, "numpy", numpy.__version__, "pyriemann", pyriemann.__version__,
      "braindecode", braindecode.__version__, "cuda", torch.cuda.is_available())
PY
"$VENV/bin/python" -m pip freeze 2>/dev/null | sort > "$VENV.freeze_rebuilt_$(date -u +%Y%m%dT%H%M%SZ).txt" || "$VENV/bin/python" -c "import importlib.metadata as m; print('\n'.join(sorted(f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions())))" > "$VENV.freeze_rebuilt_$(date -u +%Y%m%dT%H%M%SZ).txt"
echo "END $(date -u +%FT%TZ)"
echo VENV_REBUILD_DONE_EXIT0
