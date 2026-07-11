#!/usr/bin/env bash
# =============================================================================
# SBATCH — Run TT-Bench with vLLM on JURECA dc-hwai (H100)
# =============================================================================
# This script is designed to be submitted via submit_all.sh, which sets:
#   MODEL_ID       — HuggingFace model ID (e.g. Qwen/Qwen2.5-Coder-32B-Instruct)
#   MODEL_NAME     — Short label for results directory
#   GPU_COUNT      — Number of H100 GPUs to request (1–4)
#   TIERS          — Tier filter (e.g. "1" or "1 2 3 4")
#   PROJECT_DIR    — Path to the TT-Bench project root on JURECA
#
# Usage (standalone):
#   MODEL_ID=Qwen/Qwen2.5-Coder-7B-Instruct MODEL_NAME=qwen2.5-coder-7b \
#   GPU_COUNT=1 TIERS=1 \
#   sbatch --export=ALL jureca/run_benchmark.sbatch
# =============================================================================

#SBATCH --job-name=tt-vllm
#SBATCH --partition=dc-hwai
#SBATCH --account=westai0070
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# --gres=gpu:N is passed on the sbatch command line (see submit_all.sh)
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j-%a.out
#SBATCH --error=slurm_logs/%x-%j-%a.err
#SBATCH --export=ALL

set -euo pipefail

# ── Logging helpers ──────────────────────────────────────────────────────────
BOLD='\033[1m'; GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
banner() { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "  ${RED}✘${NC} $1"; }

# ── Validate required env vars ───────────────────────────────────────────────
: "${MODEL_ID:?MODEL_ID must be set (HuggingFace model ID)}"
: "${MODEL_NAME:?MODEL_NAME must be set (short label)}"
: "${GPU_COUNT:?GPU_COUNT must be set}"
: "${TIERS:?TIERS must be set (e.g. '1')}"
: "${PROJECT_DIR:?PROJECT_DIR must be set (path to tt-bench project)}"

# ── Directories ──────────────────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/slurm_logs"

VENV_DIR="$PROJECT_DIR/.venv-ttbench"
if [ ! -d "$VENV_DIR" ]; then
    VENV_DIR="/p/scratch/westai0070/${SLURM_JOB_USER:-$USER}/.venv-ttbench"
fi
if [ ! -d "$VENV_DIR" ]; then
    VENV_DIR="$HOME/.venv-ttbench"
fi
HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HF_HOME

# Compute nodes have no internet — force offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Redirect ALL cache to scratch (HOME quota is exhausted)
export XDG_CACHE_HOME="$PROJECT_DIR/.cache/xdg"
export VLLM_CACHE_DIR="$PROJECT_DIR/.cache/vllm"
export FLASHINFER_WORKSPACE_DIR="$PROJECT_DIR/.cache/flashinfer"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_DIR" "$FLASHINFER_WORKSPACE_DIR"

# ── Load CUDA module (required by flashinfer JIT compilation) ─────────────────
module load CUDA/13 2>/dev/null || module load CUDA 2>/dev/null || true

RESULTS_BASE="$PROJECT_DIR/benchmark_results/jureca_tier1"
RESULTS_DIR="$RESULTS_BASE/${MODEL_NAME}"
VLLM_PORT=8000

# ── Activate environment ─────────────────────────────────────────────────────
banner "Activating environment"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    fail "Virtual environment not found at $VENV_DIR — run jureca/setup.sh first"
    exit 1
fi
source "$VENV_DIR/bin/activate"

ok "Python: $(python --version)"
ok "CUDA devices: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo 'check on compute node')"

# ── Start vLLM server ────────────────────────────────────────────────────────
banner "Starting vLLM server"
echo "  Model:    $MODEL_ID"
echo "  GPUs:     $GPU_COUNT"
echo "  Port:     $VLLM_PORT"
echo "  HF cache: $HF_HOME"

# Kill any leftover vLLM process
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

# vLLM tensor-parallel when using >1 GPU
TP_FLAG=""
if [ "$GPU_COUNT" -gt 1 ]; then
    TP_FLAG="--tensor-parallel-size $GPU_COUNT"
fi

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --dtype auto \
    ${VLLM_ARGS:-} \
    $TP_FLAG \
    > "$PROJECT_DIR/slurm_logs/vllm_${MODEL_NAME}.log" 2>&1 &

VLLM_PID=$!
ok "vLLM started (PID: $VLLM_PID)"

# ── Wait for vLLM to be ready ────────────────────────────────────────────────
banner "Waiting for vLLM server to be ready"

MAX_WAIT=600  # 10 minutes for model loading
WAITED=0
while ! curl -s --max-time 2 "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [ $WAITED -ge $MAX_WAIT ]; then
        fail "vLLM failed to start within ${MAX_WAIT}s"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "  Waiting... ${WAITED}s elapsed"
done

ok "vLLM ready after ${WAITED}s"

# ── Run benchmark ────────────────────────────────────────────────────────────
banner "Running TT-Bench Tier $TIERS against $MODEL_NAME"

cd "$PROJECT_DIR"

CHALLENGE_SETS=(
    "data/tasks/official/challenges/json|tt-official-ch*.json|official"
    "data/tasks/challenges_1comp|tt-official-ch*.json|1comp"
    "data/tasks/challenges_2comp|tt-official-ch*.json|2comp"
)

ARGS_COMMON="--provider openai \
             --model $MODEL_ID \
             --base-url http://localhost:${VLLM_PORT}/v1 \
             --api-key none \
             --task-type agentic_synthesis \
             --save-report \
             --max-turns 25 \
             --max-tokens 32768 \
             --timeout 600 \
             --workers 8 \
             --tiers $TIERS"

TOTAL_SUCCESS=0
TOTAL_FAILED=0

for cset in "${CHALLENGE_SETS[@]}"; do
    IFS='|' read -r challenges_dir pattern cset_label <<< "$cset"

    OUT_DIR="$RESULTS_DIR/${cset_label}"
    mkdir -p "$OUT_DIR"

    banner "  Set: $cset_label (pattern: $pattern, tiers: $TIERS)"

    set +e
    uv run tt-benchmark \
        --challenges-dir "$challenges_dir" \
        --pattern "$pattern" \
        --output-dir "$OUT_DIR" \
        $ARGS_COMMON 2>&1 | tee "$PROJECT_DIR/slurm_logs/bench_${MODEL_NAME}_${cset_label}.log"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        ok "$cset_label complete"
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    else
        fail "$cset_label failed (exit code $EXIT_CODE)"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
done

# ── Stop vLLM ────────────────────────────────────────────────────────────────
banner "Stopping vLLM server"
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
ok "vLLM stopped"

# ── Summary ──────────────────────────────────────────────────────────────────
banner "Job complete"
echo ""
echo "  Model:     $MODEL_NAME"
echo "  Sets done: $TOTAL_SUCCESS / ${#CHALLENGE_SETS[@]}"
echo "  Results:   $RESULTS_DIR"
echo ""

# Save a job manifest for aggregation
MANIFEST="$RESULTS_BASE/manifest_${MODEL_NAME}.json"
python3 -c "
import json
json.dump({
    'model_name': '${MODEL_NAME}',
    'model_id': '${MODEL_ID}',
    'tiers': '${TIERS}',
    'gpus': ${GPU_COUNT},
    'results_dir': '${RESULTS_DIR}',
    'completed': $( [ $TOTAL_SUCCESS -eq ${#CHALLENGE_SETS[@]} ] && echo true || echo false ),
}, open('${MANIFEST}', 'w'), indent=2)
"

exit 0
