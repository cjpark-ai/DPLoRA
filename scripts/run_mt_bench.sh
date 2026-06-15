#!/bin/bash
# MT-Bench evaluation launcher — thin wrapper around evaluation/mt_bench.py
# (the single source of truth, also used by train_alpaca.py --run_mt_bench).
# Handles only shell concerns: model-path resolution, sourcing .env, CLI invoke.
#
# Usage: bash scripts/run_mt_bench.sh <model_path> [skip_judgment]
#   Pass ONE concrete path (avoid unquoted globs that may multi-match).
#   bash scripts/run_mt_bench.sh output/.../final_model
#   bash scripts/run_mt_bench.sh output/.../final_model true   # answers only
#
# Run in the same Python/GPU environment used for training.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load OPENAI_API_KEY / HF_TOKEN if present.
if [ -f "$PROJECT_DIR/.env" ]; then
    source "$PROJECT_DIR/.env"
fi

MODEL_PATH=${1:?"Usage: $0 <model_path> [skip_judgment]"}
SKIP_JUDGMENT=${2:-false}

# Guard against an unquoted glob that matched multiple dirs: extra args mean the
# 2nd match was silently absorbed as skip_judgment and the rest dropped.
if [ "$#" -gt 2 ]; then
    echo "ERROR: too many args ($#). An unquoted glob likely matched multiple" >&2
    echo "       models — pass exactly ONE model_path. Got: $*" >&2
    exit 1
fi
case "$SKIP_JUDGMENT" in
    true|false) ;;
    *) echo "ERROR: skip_judgment must be 'true' or 'false', got '$SKIP_JUDGMENT'" >&2
       echo "       (a glob may have leaked a path into this argument)." >&2
       exit 1 ;;
esac

# Validate BEFORE resolving: check existence/config first so the friendly error
# is reachable (a bare 'cd' on a missing path would abort under set -e first).
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model dir not found: $MODEL_PATH" >&2
    exit 1
fi
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: $MODEL_PATH is not a valid HF model dir (no config.json)" >&2
    exit 1
fi
MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"

# PROJECT_DIR on PYTHONPATH so `-m evaluation.mt_bench` imports; the module
# self-injects the local FastChat into its subprocesses' PYTHONPATH.
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
cd "$PROJECT_DIR"

ARGS=( "$MODEL_PATH" --output-dir "$(dirname "$MODEL_PATH")/mt_bench_results" )
if [ "$SKIP_JUDGMENT" = "true" ]; then
    ARGS+=( --skip-judgment )
fi

echo "=========================================="
echo "MT-Bench: model_path    = $MODEL_PATH"
echo "          skip_judgment = $SKIP_JUDGMENT"
echo "=========================================="

# Python logs go to stderr; the final stdout line is the results JSON.
OUT="$(python3 -m evaluation.mt_bench "${ARGS[@]}")"
echo "$OUT"
python3 -c "import json,sys; d=json.loads(sys.argv[1]); print('MT-Bench Overall Score:', d.get('overall_score', '(judgment skipped / not computed)'))" "$OUT"
