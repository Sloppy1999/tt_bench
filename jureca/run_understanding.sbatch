#!/usr/bin/env bash
# =============================================================================
# SBATCH — Run the procedural-understanding item bank with vLLM on JURECA
# =============================================================================
# Sibling to run_benchmark.sbatch, but for --task-type understanding instead of
# agentic_synthesis: one single-shot QA pass over the official item bank
# (review/understanding_study/PLAN.md), not a multi-turn agentic loop. Kept as
# its own script rather than a branch in run_benchmark.sbatch because the two
# task types share nothing beyond "start vLLM, then call tt-benchmark" — no
# turn budget, no harness-ablation arm, no challenge-set sweep.
#
# Designed to be submitted via submit_understanding.sh, which sets:
#   MODEL_ID       — HuggingFace model ID
#   MODEL_NAME     — Short label; results land in review/understanding_study/runs/$MODEL_NAME
#   GPU_COUNT      — Number of H100 GPUs to request (1-4)
#   PROJECT_DIR    — Path to the TT-Bench project root on JURECA
#
# Usage (standalone):
#   MODEL_ID=Qwen/Qwen2.5-Coder-7B-Instruct MODEL_NAME=qwen2.5-coder-7b \
#   GPU_COUNT=1 PROJECT_DIR=/p/scratch/westai0070/$USER/tt_bench \
#   sbatch --export=ALL jureca/run_understanding.sbatch
# =============================================================================

#SBATCH --job-name=tt-understand
#SBATCH --partition=dc-hwai
#SBATCH --account=westai0070
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
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
: "${PROJECT_DIR:?PROJECT_DIR must be set (path to tt-bench project)}"

mkdir -p "$PROJECT_DIR/slurm_logs"
JOB_TAG="${SLURM_JOB_ID:-manual}"

VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-ttbench}"
if [ ! -d "$VENV_DIR" ]; then
    VENV_DIR="/p/scratch/westai0070/${SLURM_JOB_USER:-$USER}/.venv-ttbench"
fi
if [ ! -d "$VENV_DIR" ]; then
    VENV_DIR="$HOME/.venv-ttbench"
fi
HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Same cache redirection as run_benchmark.sbatch — see that file for why each
# of these is its own variable rather than XDG_CACHE_HOME alone.
export XDG_CACHE_HOME="$PROJECT_DIR/.cache/xdg"
export VLLM_CACHE_DIR="$PROJECT_DIR/.cache/vllm"
export TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_DIR/.cache/torchinductor"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PROJECT_DIR/.cache/uv-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.cache/uv}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$PROJECT_DIR/.cache/tiktoken}"
export TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_RS_CACHE_DIR:-$TIKTOKEN_CACHE_DIR}"
export FLASHINFER_WORKSPACE_BASE="$PROJECT_DIR"
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_DIR" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$PROJECT_DIR/.cache/flashinfer" \
         "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR" "$TIKTOKEN_CACHE_DIR"

module load CUDA/13 2>/dev/null || module load CUDA 2>/dev/null || true

RUNS_DIR="$PROJECT_DIR/review/understanding_study/runs"
OUT_DIR="$RUNS_DIR/$MODEL_NAME"
VLLM_PORT=8000
VLLM_LOG="$PROJECT_DIR/slurm_logs/vllm_understand_${MODEL_NAME}_${JOB_TAG}.log"

# ── Activate environment ─────────────────────────────────────────────────────
banner "Activating environment"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    fail "No usable interpreter at $VENV_DIR/bin/python — run jureca/setup.sh first"
    ls -l "$VENV_DIR/bin/python" 2>&1 | sed 's/^/    /'
    exit 1
fi
source "$VENV_DIR/bin/activate"
VENV_PY="$VENV_DIR/bin/python"

PY_VERSION="$("$VENV_PY" --version 2>&1)" || PY_VERSION=""
if [ -z "$PY_VERSION" ]; then
    fail "$VENV_PY produced no version output — the venv is broken"
    exit 1
fi
ok "Python: $PY_VERSION  ($VENV_PY)"
ok "CUDA devices: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo 'check on compute node')"

# ── Start vLLM server ────────────────────────────────────────────────────────
banner "Starting vLLM server"
echo "  Model:    $MODEL_ID"
echo "  GPUs:     $GPU_COUNT"
echo "  Port:     $VLLM_PORT"
echo "  HF cache: $HF_HOME"

pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

# The understanding task never calls a tool, so the tool-call parser does not
# matter for scoring — but vLLM still requires a VALID one whenever
# --enable-auto-tool-choice is set, and passing it uniformly here keeps this
# script identical in shape to run_benchmark.sbatch. See that file's comment
# for the per-family mapping this mirrors.
if [ -z "${TOOL_PARSER:-}" ]; then
    case "$MODEL_ID" in
        *Qwen2.5*|*Qwen2.5-Coder*)  TOOL_PARSER="hermes" ;;
        *Qwen3*|*Qwen3.6*|*Qwen3-Coder*)  TOOL_PARSER="qwen3_coder" ;;
        *gpt-oss*|*gpt_oss*)        TOOL_PARSER="openai" ;;
        *gemma-4*|*Gemma-4*)        TOOL_PARSER="gemma4" ;;
        *functiongemma*)            TOOL_PARSER="functiongemma" ;;
        *DeepSeek-Coder-V2*)        TOOL_PARSER="deepseek_v3" ;;
        *DeepSeek*V3.1*)            TOOL_PARSER="deepseek_v31" ;;
        *DeepSeek*V3*)              TOOL_PARSER="deepseek_v3" ;;
        *Llama*|*llama*)            TOOL_PARSER="llama3_json" ;;
        *Mistral*|*mistral*)        TOOL_PARSER="mistral" ;;
        *hermes*|*Hermes*)          TOOL_PARSER="hermes" ;;
        *xlam*|*XLAM*)              TOOL_PARSER="xlam" ;;
        *)                          TOOL_PARSER="hermes" ;;
    esac
fi
echo "  Tool parser: $TOOL_PARSER"

TP_FLAG=()
if [ "$GPU_COUNT" -gt 1 ]; then
    TP_FLAG=(--tensor-parallel-size "$GPU_COUNT")
fi

# Understanding prompts are single-shot (board + a handful of questions), far
# shorter than an agentic transcript, so the 131072 target used for
# agentic_synthesis is unneeded headroom here. Kept anyway so a model already
# cached under that context (and, where applicable, its YaRN override) behaves
# identically to how it was served for the agentic runs — no second cache entry,
# no second measurement condition to account for.
TARGET_CTX=131072
HF_OVERRIDE_FLAG=()
set +e
NATIVE_CTX=$("$VENV_PY" - "$MODEL_ID" "$HF_HOME" <<'PY'
import sys, json, glob, os
model_id, hf_home = sys.argv[1], sys.argv[2]
repo = "models--" + model_id.replace("/", "--")
cfgs = sorted(glob.glob(os.path.join(hf_home, "hub", repo, "snapshots", "*", "config.json")))
n = 0
if cfgs:
    c = json.load(open(cfgs[-1]))
    tc = c.get("text_config") if isinstance(c.get("text_config"), dict) else {}
    n = c.get("max_position_embeddings") or tc.get("max_position_embeddings") or c.get("max_seq_len") or 0
print(int(n or 0))
PY
)
set -e
[ -z "$NATIVE_CTX" ] && NATIVE_CTX=0
[ "$NATIVE_CTX" -le 0 ] && NATIVE_CTX=32768
MAX_MODEL_LEN=$TARGET_CTX
if [ "$NATIVE_CTX" -lt "$TARGET_CTX" ]; then
    FACTOR=$("$VENV_PY" -c "print(f'{$TARGET_CTX/$NATIVE_CTX:.2f}')")
    HF_OVERRIDE_FLAG=(--hf-overrides "{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":$FACTOR,\"original_max_position_embeddings\":$NATIVE_CTX}}")
    echo "  Context: $MAX_MODEL_LEN via YaRN (native $NATIVE_CTX, factor $FACTOR)"
else
    echo "  Context: $MAX_MODEL_LEN (native $NATIVE_CTX, no scaling)"
fi

"$VENV_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --served-model-name "$MODEL_ID" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$MAX_MODEL_LEN" \
    ${HF_OVERRIDE_FLAG[@]+"${HF_OVERRIDE_FLAG[@]}"} \
    --dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    ${VLLM_ARGS:-} \
    ${TP_FLAG[@]+"${TP_FLAG[@]}"} \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
ok "vLLM started (PID: $VLLM_PID)"

# ── Wait for vLLM to be ready ────────────────────────────────────────────────
banner "Waiting for vLLM server to be ready"
MAX_WAIT="${VLLM_MAX_WAIT:-1800}"
WAITED=0
while ! curl -s --max-time 2 "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        fail "vLLM process ($VLLM_PID) exited before becoming ready — last log lines:"
        tail -n 25 "$VLLM_LOG" 2>/dev/null | sed 's/^/    /'
        fail "Full server log: $VLLM_LOG"
        exit 1
    fi
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

# ── Run the understanding item bank ─────────────────────────────────────────
banner "Running procedural-understanding item bank against $MODEL_NAME"
cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR"

# Same call shape as review/understanding_study/run_all.sh (the deepseek runs):
# task-type understanding, official pattern, temperature 0.0, max-tokens 8192
# (raised from 4096 by Amendment A1 — see PLAN.md — after truncated JSON
# `reasoning` fields turned budget into a confound). --workers is higher than
# the deepseek study's 4: that was chosen for a rate-limited external API, and
# this is a local vLLM server this job already reserved the whole node for.
#
# WORKERS is the client-side concurrency, and 8 leaves the node idle: a slow
# reasoning model served in eager mode sat at 2% KV-cache use with 8 requests
# running and an empty queue, so the client, not the GPU, set the pace. Raise it
# for such a model — but it is a SERVING CONDITION, so a run that changes it is
# not directly comparable to one that does not, and PROVENANCE.txt records it.
BENCH_LOG="$PROJECT_DIR/slurm_logs/bench_understand_${MODEL_NAME}_${JOB_TAG}.log"
set +e
uv run tt-benchmark \
    --provider vllm \
    --model "$MODEL_ID" \
    --base-url "http://localhost:${VLLM_PORT}" \
    --api-key none \
    --task-type understanding \
    --challenges-dir data/tasks/official/challenges/json \
    --pattern 'tt-official-ch*.json' \
    --temperature 0.0 \
    --max-tokens 8192 \
    --workers "${WORKERS:-8}" \
    --output-dir "$OUT_DIR" \
    --save-report 2>&1 | tee "$BENCH_LOG"
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 0 ]; then
    ok "understanding run complete — $(grep -m1 'Success Rate:' "$BENCH_LOG" 2>/dev/null || echo 'Success Rate: n/a')"
else
    fail "understanding run failed (exit code $EXIT_CODE) — see $BENCH_LOG"
fi

# Per-run provenance, same key=value shape as run_all.sh's runs/PROVENANCE.txt,
# so analyze.py's generic `k=v` parser (which reads any runs/<name>/PROVENANCE.txt)
# picks this run up the same way it picks up the deepseek ones.
{
  echo "utc_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)"
  echo "provider=vllm"
  echo "model_id=$MODEL_ID"
  echo "gpu_count=$GPU_COUNT"
  echo "temperature=0.0"
  echo "max_tokens=8192"
  echo "workers=${WORKERS:-8}"
  echo "task_type=understanding"
  echo "pattern=tt-official-ch*.json"
  echo "job_id=$JOB_TAG"
} > "$OUT_DIR/PROVENANCE.txt"

# ── Stop vLLM ────────────────────────────────────────────────────────────────
banner "Stopping vLLM server"
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
ok "vLLM stopped"

banner "Job complete"
echo "  Model:   $MODEL_NAME"
echo "  Results: $OUT_DIR"

exit $EXIT_CODE
