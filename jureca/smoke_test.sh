#!/usr/bin/env bash
# =============================================================================
# Tool-Calling Smoke Test for the TT-Bench model roster (JURECA H100)
# =============================================================================
# Validates that each model's --tool-call-parser actually produces tool calls
# BEFORE committing to the full multi-hour sweep. A mismatched parser fails
# silently — the model returns zero tool_calls and flails — so this catches the
# expensive failure mode cheaply: one 1-component task per model.
#
# Run INTERACTIVELY on an allocated H100 node (this script does not submit to
# Slurm — it drives vLLM directly on whatever node you are on):
#
#   salloc --partition=dc-hwai --account=westai0070 \
#          --nodes=1 --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=01:00:00
#   bash jureca/smoke_test.sh                       # test the whole roster
#   bash jureca/smoke_test.sh qwen2.5-coder-7b      # test only matching model(s)
#
# For each model it prints ✔ / ✘ on tool calling and exits non-zero if any
# model fails, so it is safe to chain:  bash jureca/smoke_test.sh && bash jureca/submit_all.sh
#
# NOTE: the model roster and the parser case-statement below are kept in sync
# by hand with jureca/run_benchmark.sbatch — change both together.
# =============================================================================

set -uo pipefail   # NOT -e: we handle per-model failures and keep going

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Project directory (prefer scratch on JURECA) ─────────────────────────────
if [ -n "${PROJECT_DIR:-}" ]; then
    :
elif [ -d "/p/scratch/westai0070/$USER/tt-bench" ]; then
    PROJECT_DIR="/p/scratch/westai0070/$USER/tt-bench"
elif [ -d "$SCRIPT_DIR/.." ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    echo "ERROR: cannot locate project directory." >&2
    exit 1
fi

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
banner() { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "  ${RED}✘${NC} $1"; }

# ── Model roster (keep in sync with submit_all.sh) ───────────────────────────
# Format: "MODEL_ID|short_name|GPU_count"
MODELS=(
    "Qwen/Qwen2.5-Coder-7B-Instruct|qwen2.5-coder-7b|1"
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct|deepseek-coder-v2-lite|1"
    "google/gemma-4-26B-A4B-it|gemma-4-26b-a4b|1"
    "Qwen/Qwen3.6-35B-A3B|qwen3.6-35b-a3b|1"
    "google/gemma-4-31B|gemma-4-31b|1"
)

# Optional CLI filter: keep only roster entries whose id or short name matches
# any argument (substring match).
if [ "$#" -gt 0 ]; then
    FILTERED=()
    for entry in "${MODELS[@]}"; do
        for pat in "$@"; do
            if [[ "$entry" == *"$pat"* ]]; then FILTERED+=("$entry"); break; fi
        done
    done
    MODELS=("${FILTERED[@]}")
    if [ "${#MODELS[@]}" -eq 0 ]; then
        echo "No roster models match: $*" >&2
        exit 1
    fi
fi

# ── Parser auto-detect (keep in sync with run_benchmark.sbatch) ──────────────
detect_parser() {
    local model_id="$1"
    case "$model_id" in
        *Qwen2.5*|*Qwen2.5-Coder*)        echo "hermes" ;;
        *Qwen3*|*Qwen3.6*|*Qwen3-Coder*)  echo "qwen3_coder" ;;
        *gemma-4*|*Gemma-4*)              echo "gemma4" ;;
        *functiongemma*)                  echo "functiongemma" ;;
        *DeepSeek-Coder-V2*)              echo "deepseek_v3" ;;
        *DeepSeek*V3.1*)                  echo "deepseek_v31" ;;
        *DeepSeek*V3*)                    echo "deepseek_v3" ;;
        *Llama*|*llama*)                  echo "llama3_json" ;;
        *Mistral*|*mistral*)              echo "mistral" ;;
        *hermes*|*Hermes*)                echo "hermes" ;;
        *xlam*|*XLAM*)                    echo "xlam" ;;
        *)                                echo "hermes" ;;
    esac
}

# ── Environment (mirror run_benchmark.sbatch) ────────────────────────────────
VENV_DIR="$PROJECT_DIR/.venv-ttbench"
[ -d "$VENV_DIR" ] || VENV_DIR="/p/scratch/westai0070/$USER/.venv-ttbench"
[ -d "$VENV_DIR" ] || VENV_DIR="$HOME/.venv-ttbench"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    fail "virtual environment not found — run jureca/setup.sh first"; exit 1
fi

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$PROJECT_DIR/.cache/xdg"
export VLLM_CACHE_DIR="$PROJECT_DIR/.cache/vllm"
export FLASHINFER_WORKSPACE_DIR="$PROJECT_DIR/.cache/flashinfer"
export TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_DIR/.cache/torchinductor"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_DIR" "$FLASHINFER_WORKSPACE_DIR" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

module load CUDA/13 2>/dev/null || module load CUDA 2>/dev/null || true

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

VLLM_PORT=8000
SMOKE_DIR="$PROJECT_DIR/benchmark_results/smoke_test"
LOG_DIR="$PROJECT_DIR/slurm_logs"
mkdir -p "$SMOKE_DIR" "$LOG_DIR"

# ── vLLM lifecycle helpers ───────────────────────────────────────────────────
VLLM_PID=""
kill_vllm() {
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
    [ -n "$VLLM_PID" ] && wait "$VLLM_PID" 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    VLLM_PID=""
}
trap 'kill_vllm; exit 130' INT TERM
trap 'kill_vllm' EXIT

wait_for_ready() {   # $1 = timeout seconds
    local waited=0
    while ! curl -s --max-time 2 "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
        # bail early if the server process already died
        if [ -n "$VLLM_PID" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then return 1; fi
        sleep 10; waited=$((waited + 10))
        [ "$waited" -ge "$1" ] && return 1
    done
    return 0
}

# ── Run the roster ───────────────────────────────────────────────────────────
banner "Tool-calling smoke test — ${#MODELS[@]} model(s)"
echo "  Node:      $(hostname)"
echo "  GPUs seen: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "  Project:   $PROJECT_DIR"

RESULTS=()   # "short_name|status|detail"

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id model_name gpu_count <<< "$entry"
    parser="${TOOL_PARSER:-$(detect_parser "$model_id")}"

    banner "$model_name  ($model_id)"
    echo "  parser: $parser   gpus: $gpu_count"

    kill_vllm; sleep 2

    TP_FLAG=""
    [ "$gpu_count" -gt 1 ] && TP_FLAG="--tensor-parallel-size $gpu_count"

    vllm_log="$LOG_DIR/smoke_vllm_${model_name}.log"
    python -m vllm.entrypoints.openai.api_server \
        --model "$model_id" \
        --served-model-name "$model_id" \
        --port "$VLLM_PORT" \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        --dtype auto \
        --enable-auto-tool-choice \
        --tool-call-parser "$parser" \
        ${VLLM_ARGS:-} \
        $TP_FLAG \
        > "$vllm_log" 2>&1 &
    VLLM_PID=$!

    if ! wait_for_ready 600; then
        fail "server did not become ready — last lines of $vllm_log:"
        tail -n 5 "$vllm_log" | sed 's/^/      /'
        RESULTS+=("$model_name|ERROR|server failed to start (see $vllm_log)")
        kill_vllm
        continue
    fi
    ok "server ready"

    out_dir="$SMOKE_DIR/$model_name"
    rm -rf "$out_dir"; mkdir -p "$out_dir"

    cd "$PROJECT_DIR"
    uv run tt-benchmark \
        --provider vllm \
        --model "$model_id" \
        --base-url "http://localhost:${VLLM_PORT}" \
        --api-key none \
        --challenges-dir data/tasks/challenges_1comp \
        --pattern 'tt-official-ch01-1comp.json' \
        --task-type agentic_synthesis \
        --max-tasks 1 --max-turns 5 --max-tokens 8192 \
        --timeout 600 --workers 1 \
        --output-dir "$out_dir" --save-report \
        > "$LOG_DIR/smoke_bench_${model_name}.log" 2>&1

    kill_vllm

    # newest report in the model's output dir
    report="$(ls -t "$out_dir"/benchmark_*.json 2>/dev/null | head -n1)"
    if [ -z "$report" ]; then
        fail "no report produced (benchmark crashed) — see $LOG_DIR/smoke_bench_${model_name}.log"
        RESULTS+=("$model_name|ERROR|no report (benchmark crashed)")
        continue
    fi

    # extract tool_calls_count + success from the single result
    read -r tc succ <<< "$(python3 - "$report" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
r = (d.get("results") or [{}])[0]
print(r.get("metrics", {}).get("tool_calls_count", 0), r.get("success"))
PY
)"

    if [ "${tc:-0}" -gt 0 ]; then
        ok "tool calling WORKS — tool_calls_count=$tc, task_success=$succ"
        RESULTS+=("$model_name|PASS|tool_calls=$tc success=$succ")
    else
        fail "NO tool calls (tool_calls_count=0) — parser '$parser' likely wrong for this model"
        RESULTS+=("$model_name|FAIL|tool_calls=0 (bad parser?)")
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────────
banner "Smoke-test summary"
printf "  %-24s %-6s %s\n" "Model" "Status" "Detail"
echo   "  --------------------------------------------------------------------"
any_bad=0
for row in "${RESULTS[@]}"; do
    IFS='|' read -r name status detail <<< "$row"
    case "$status" in
        PASS) mark="${GREEN}PASS${NC}" ;;
        *)    mark="${RED}${status}${NC}"; any_bad=1 ;;
    esac
    printf "  %-24s ${mark}   %s\n" "$name" "$detail"
done
echo ""

if [ "$any_bad" -eq 0 ]; then
    ok "All models produce tool calls — safe to run jureca/submit_all.sh"
    exit 0
else
    fail "One or more models failed — fix the parser mapping in run_benchmark.sbatch before submitting"
    exit 1
fi
