#!/usr/bin/env bash
# Supply --account, --partition, --gres, --cpus-per-task, --mem, --time,
# --array=0-N%K, --output and --error at submission after inspecting allocation.
# MATRIX: JSON list of {config, arm, output_root}, all paths relative to checkout.
# For a smoke job, "arms": [arm, ...] can replace "arm" to reuse one server.
# Each config serving_config is a JSON array of explicit vLLM CLI arguments.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${PROJECT_DIR:?Set isolated experiment checkout}"
: "${VENV_DIR:?Set existing project environment}"
: "${MATRIX:?Set frozen matrix path}"
: "${CUDA_MODULE:?Set inspected CUDA module name}"
: "${HF_HOME:?Set existing model cache}"
: "${EXPERIMENT_CACHE:?Set project-local runtime cache directory}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"
test -x "$VENV_DIR/bin/python"
source "$VENV_DIR/bin/activate"
module load "$CUDA_MODULE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$EXPERIMENT_CACHE/xdg"
export VLLM_CACHE_DIR="$EXPERIMENT_CACHE/vllm"
export VLLM_CACHE_ROOT="$EXPERIMENT_CACHE/vllm"
export TRITON_CACHE_DIR="$EXPERIMENT_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$EXPERIMENT_CACHE/torchinductor"
export FLASHINFER_WORKSPACE_BASE="$EXPERIMENT_CACHE"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$EXPERIMENT_CACHE/tiktoken}"
export TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_RS_CACHE_DIR:-$TIKTOKEN_CACHE_DIR}"
mkdir -p "$EXPERIMENT_CACHE" slurm_logs
PY="$VENV_DIR/bin/python"
RUNNER_PY="${RUNNER_PY:-$PY}"
test -x "$RUNNER_PY"
printf 'job=%s array=%s host=%s time=%s\n' "$SLURM_JOB_ID" "${SLURM_ARRAY_TASK_ID:-0}" "$(hostname)" "$(date -Is)"
git rev-parse HEAD
"$PY" --version
"$PY" - <<'PY'
import importlib.metadata as m, json
versions = {}
for name in ('vllm', 'torch', 'transformers', 'flashinfer-python'):
    try:
        versions[name] = m.version(name)
    except m.PackageNotFoundError:
        versions[name] = None
print('Serving packages:', json.dumps(versions, sort_keys=True))
PY
module list
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mapfile -t CELL < <("$PY" - "$MATRIX" "${SLURM_ARRAY_TASK_ID:-0}" <<'PY'
import json, sys
cell = json.load(open(sys.argv[1]))[int(sys.argv[2])]
values = [cell['config'], cell['output_root'], *cell.get('arms', [cell.get('arm')])]
for value in values:
    if not isinstance(value, str):
        raise ValueError('Matrix values must be strings')
    if '\n' in value:
        raise ValueError('Newlines are not allowed in matrix values')
    print(value)
PY
)
test "${#CELL[@]}" -ge 3
mapfile -t SERVER_ARGS < <("$PY" - "${CELL[0]}" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
args = c['serving_config']
if not isinstance(args, list) or not all(isinstance(v, str) and '\n' not in v for v in args):
    raise ValueError('serving_config must be an array of arguments')
if any(v in args for v in ('--model', '--revision', '--port', '--host')):
    raise ValueError('Model, revision, host and port are set by the job script')
for v in ['--model', c['model'], '--revision', c['model_revision'],
          '--served-model-name', c['model'], *args]:
    print(v)
PY
)
test "${#SERVER_ARGS[@]}" -ge 6
SERVER_PID=""
RUNNER_PID=""
cleanup() {
    if [ -n "$RUNNER_PID" ]; then kill "$RUNNER_PID" 2>/dev/null || true; wait "$RUNNER_PID" 2>/dev/null || true; fi
    if [ -n "$SERVER_PID" ]; then kill -- "-$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT USR1
PORT="${VLLM_PORT:-8000}"
export TTBENCH_BASE_URL="http://127.0.0.1:$PORT"
setsid "$PY" -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port "$PORT" "${SERVER_ARGS[@]}" \
    > "slurm_logs/ladder-server-${SLURM_JOB_ID}.out" \
    2> "slurm_logs/ladder-server-${SLURM_JOB_ID}.err" &
SERVER_PID=$!
READY=0
for ((i=0; i<180; i++)); do
    kill -0 "$SERVER_PID" 2>/dev/null || exit 1
    if curl --fail --silent --max-time 2 "$TTBENCH_BASE_URL/v1/models" > /dev/null; then READY=1; break; fi
    sleep 10
done
test "$READY" -eq 1
for ARM in "${CELL[@]:2}"; do
    "$RUNNER_PY" scripts/run_controlled_ladder.py --config "${CELL[0]}" \
        --arm "$ARM" --output-root "${CELL[1]}" &
    RUNNER_PID=$!
    wait "$RUNNER_PID"
    RUNNER_PID=""
done
printf 'finished=%s\n' "$(date -Is)"
