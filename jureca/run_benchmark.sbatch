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
# dc-hwai allocates WHOLE nodes: the job gets all 4 H100s whether or not
# --gres=gpu:N is passed on the sbatch command line (see submit_all.sh). Set
# GPU_COUNT=4 to actually use them — it drives --tensor-parallel-size below.
#SBATCH --mem=128G
#SBATCH --time=12:00:00
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

# Tag every log this job writes with its Slurm job ID. Without it the per-set
# logs are named only after the model, so each re-run silently overwrites the
# previous one and you cannot tell which run a log belongs to.
JOB_TAG="${SLURM_JOB_ID:-manual}"

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

# Redirect ALL cache to scratch (HOME quota is exhausted). Each tool uses its
# OWN variable and ignores XDG_CACHE_HOME:
#   - Triton / Torch Inductor default to ~/.triton and a HOME path.
#   - flashinfer builds its cache as <FLASHINFER_WORKSPACE_BASE>/.cache/flashinfer,
#     defaulting the base to $HOME. The correct knob is FLASHINFER_WORKSPACE_BASE
#     (NOT FLASHINFER_WORKSPACE_DIR, which flashinfer never reads) — getting it
#     wrong crashes engine init with "Disk quota exceeded: ~/.cache/flashinfer".
#   - uv keeps its managed interpreters and cache under $HOME by default, which
#     is both quota-limited and the first thing to go stale in a filesystem
#     incident. `uv run` below needs them, so point them at scratch too.
export XDG_CACHE_HOME="$PROJECT_DIR/.cache/xdg"
export VLLM_CACHE_DIR="$PROJECT_DIR/.cache/vllm"
export TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_DIR/.cache/torchinductor"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PROJECT_DIR/.cache/uv-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.cache/uv}"
export FLASHINFER_WORKSPACE_BASE="$PROJECT_DIR"   # → $PROJECT_DIR/.cache/flashinfer
export VLLM_USE_FLASHINFER_SAMPLER=0              # skip the flashinfer sampler import path
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_DIR" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$PROJECT_DIR/.cache/flashinfer" \
         "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"

# ── Load CUDA module (required by flashinfer JIT compilation) ─────────────────
module load CUDA/13 2>/dev/null || module load CUDA 2>/dev/null || true

RESULTS_BASE="$PROJECT_DIR/benchmark_results/jureca_tier1"
RESULTS_DIR="$RESULTS_BASE/${MODEL_NAME}"
VLLM_PORT=8000
VLLM_LOG="$PROJECT_DIR/slurm_logs/vllm_${MODEL_NAME}_${JOB_TAG}.log"

# ── Activate environment ─────────────────────────────────────────────────────
banner "Activating environment"

# Guard on the INTERPRETER, not on bin/activate. A venv whose bin/python symlink
# dangles — e.g. uv replaced its managed CPython 3.12.12 with 3.12.13 and pruned
# the old directory — still ships a perfectly readable bin/activate. Checking
# that file passes, activation "succeeds", and the job then dies much later with
# a bare "python: command not found". -x follows the symlink, so it catches it.
if [ ! -x "$VENV_DIR/bin/python" ]; then
    fail "No usable interpreter at $VENV_DIR/bin/python — run jureca/setup.sh first"
    ls -l "$VENV_DIR/bin/python" 2>&1 | sed 's/^/    /'
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Call the venv interpreter by absolute path from here on. PATH is not
# trustworthy inside a batch job: when bin/python dangles, bash silently skips
# it and falls through to the system python3, which still runs stdlib-only
# snippets and therefore MASKS a broken venv until vLLM fails to import.
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

# Kill any leftover vLLM process
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

# ── Auto-detect correct --tool-call-parser for this model ─────────────────
# Can be overridden by setting TOOL_PARSER in the environment.
# Both --enable-auto-tool-choice and --tool-call-parser are REQUIRED
# by vLLM when the request includes tools.
if [ -z "${TOOL_PARSER:-}" ]; then
    case "$MODEL_ID" in
        *Qwen2.5*|*Qwen2.5-Coder*)  TOOL_PARSER="hermes" ;;
        *Qwen3*|*Qwen3.6*|*Qwen3-Coder*)  TOOL_PARSER="qwen3_coder" ;;
        *gemma-4*|*Gemma-4*)        TOOL_PARSER="gemma4" ;;
        *functiongemma*)            TOOL_PARSER="functiongemma" ;;
        *DeepSeek-Coder-V2*)        TOOL_PARSER="deepseek_v3" ;;  # experimental
        *DeepSeek*V3.1*)            TOOL_PARSER="deepseek_v31" ;;
        *DeepSeek*V3*)              TOOL_PARSER="deepseek_v3" ;;
        *Llama*|*llama*)            TOOL_PARSER="llama3_json" ;;
        *Mistral*|*mistral*)        TOOL_PARSER="mistral" ;;
        *hermes*|*Hermes*)          TOOL_PARSER="hermes" ;;
        *xlam*|*XLAM*)              TOOL_PARSER="xlam" ;;
        *)                          TOOL_PARSER="hermes" ;;  # safe fallback
    esac
fi
echo "  Tool parser: $TOOL_PARSER"

# vLLM tensor-parallel when using >1 GPU. Bash arrays, not strings: an unquoted
# string expansion word-splits on whitespace and silently mangles any argument
# that ever grows a space in it. Arrays are expanded below as
# ${ARR[@]+"${ARR[@]}"} — under `set -u` a plain "${ARR[@]}" on an EMPTY array
# is an unbound-variable error in bash < 4.4.
TP_FLAG=()
if [ "$GPU_COUNT" -gt 1 ]; then
    TP_FLAG=(--tensor-parallel-size "$GPU_COUNT")
fi

# ── Context length: raise to 131072 so long agentic transcripts don't overflow.
# The YaRN factor is per-model — apply it only where the model's native context
# is smaller than the target (e.g. Qwen2.5 = 32K → factor 4; DeepSeek-V2-Lite is
# already 160K+ → no scaling). Read native context from the cached config.json.
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
    # vLLM removed the standalone --rope-scaling flag (0.24.0 rejects it with
    # "unrecognized arguments" and the server dies before it ever binds a port).
    # YaRN now goes through --hf-overrides, which patches the model's HuggingFace
    # config rather than taking a vLLM-native parameter — hence the extra
    # {"rope_scaling": ...} nesting level.
    HF_OVERRIDE_FLAG=(--hf-overrides "{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":$FACTOR,\"original_max_position_embeddings\":$NATIVE_CTX}}")
    echo "  Context: $MAX_MODEL_LEN via YaRN (native $NATIVE_CTX, factor $FACTOR)"
    warn "YaRN is active: this model is being evaluated OUTSIDE its native context"
    warn "window ($NATIVE_CTX). Record this in the methodology alongside the factor."
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

MAX_WAIT=600  # 10 minutes for model loading
WAITED=0
while ! curl -s --max-time 2 "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    # Fail fast if the server process already died (don't poll a corpse for 10 min).
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

# ── Run benchmark ────────────────────────────────────────────────────────────
banner "Running TT-Bench Tier $TIERS against $MODEL_NAME"

cd "$PROJECT_DIR"

CHALLENGE_SETS=(
    "data/tasks/official/challenges/json|tt-official-ch*.json|official"
    "data/tasks/challenges_1comp|tt-official-ch*.json|1comp"
    "data/tasks/challenges_2comp|tt-official-ch*.json|2comp"
)
# The scaled sets (variants + insight + unsolvable, ~1013 tier-1 tasks against
# ~10 for the sets above) run ONLY with RUN_SCALED=1 — they take 8-12h on their
# own and would otherwise blow through --time without anyone asking for them.
# They go LAST so the small sets finish and save first if the job does hit its
# wall. NOTE: data/tasks/scaled used to sit in the unconditional array above,
# which contradicted this comment and made every plain run ~100x longer.
if [ "${RUN_SCALED:-}" = "1" ]; then
    CHALLENGE_SETS+=(
        "data/tasks/scaled|tt-official-ch*.json|scaled"
        "data/tasks/challenges_1comp|**/tt-official-ch*.json|scaled_1comp"
        "data/tasks/challenges_2comp|**/tt-official-ch*.json|scaled_2comp"
    )
    warn "RUN_SCALED=1: ~1013 extra tasks queued. 12h of --time may not be enough."
fi

# --max-tokens caps output per turn and must stay below --max-model-len (now
# 131072). 32768 gives verbose turns room without truncating, while leaving
# ~98k of context for the growing agentic transcript.
ARGS_COMMON="--provider vllm \
             --model $MODEL_ID \
             --base-url http://localhost:${VLLM_PORT} \
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

    BENCH_LOG="$PROJECT_DIR/slurm_logs/bench_${MODEL_NAME}_${cset_label}_${JOB_TAG}.log"

    # `set +e` only clears errexit — pipefail stays on (see `set -euo pipefail`
    # at the top), so EXIT_CODE is tt-benchmark's status and not tee's. Without
    # pipefail this would always read 0 and every set would count as a success.
    set +e
    uv run tt-benchmark \
        --challenges-dir "$challenges_dir" \
        --pattern "$pattern" \
        --output-dir "$OUT_DIR" \
        $ARGS_COMMON 2>&1 | tee "$BENCH_LOG"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        # The harness exiting 0 means it RAN, not that the model solved anything.
        # Surface the score here so a 0% run is never mistaken for a good one.
        ok "$cset_label complete — $(grep -m1 'Success Rate:' "$BENCH_LOG" 2>/dev/null || echo 'Success Rate: n/a')"
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    else
        fail "$cset_label failed (exit code $EXIT_CODE) — see $BENCH_LOG"
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
"$VENV_PY" -c "
import json
json.dump({
    'model_name': '${MODEL_NAME}',
    'model_id': '${MODEL_ID}',
    'tiers': '${TIERS}',
    'gpus': ${GPU_COUNT},
    'job_id': '${JOB_TAG}',
    'native_ctx': ${NATIVE_CTX},
    'max_model_len': ${MAX_MODEL_LEN},
    'yarn_factor': ${FACTOR:-None},
    'tool_parser': '${TOOL_PARSER}',
    'run_scaled': '${RUN_SCALED:-0}',
    'results_dir': '${RESULTS_DIR}',
    'completed': $( [ $TOTAL_SUCCESS -eq ${#CHALLENGE_SETS[@]} ] && echo True || echo False ),
}, open('${MANIFEST}', 'w'), indent=2)
"

exit 0
