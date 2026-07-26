#!/usr/bin/env bash
# =============================================================================
# Submit Multiple Models for Tier 1 TT-Bench on JURECA H100s
# =============================================================================
# Submits one Slurm job per model, each running the full Tier 1 benchmark
# across official, 1comp, and 2comp challenge sets.
#
# Usage:
#   bash jureca/submit_all.sh                      # whole roster
#   bash jureca/submit_all.sh gemma-4-31b-it       # one model
#   bash jureca/submit_all.sh --list               # show the roster
#   bash jureca/submit_all.sh -n qwen2.5-coder-7b  # dry run
#
# ALWAYS prefer this wrapper over calling `sbatch run_benchmark.sbatch` by hand.
# It sets all five required env vars (a missing TIERS kills the job in 19s), runs
# pre-flight checks, and carries the roster notes about which models actually
# work. Bypassing it means re-discovering those notes the expensive way.
#
# Prerequisites:
#   - jureca/setup.sh has been run on the login node
#   - Models are available via HuggingFace (JURECA compute nodes cache to
#     $PROJECT_DIR/.cache/huggingface)
#
# Pre-downloading models (run on login node):
#   source ~/.venv-ttbench/bin/activate
#   pip install huggingface_hub
#   export HF_HOME=$PROJECT_DIR/.cache/huggingface
#   huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct
#   # ... repeat for each model
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-detect project directory. On JURECA, prefer scratch storage.
if [ -n "${PROJECT_DIR:-}" ]; then
    :  # already set via environment
elif [ -d "/p/scratch/westai0070/$USER/tt-bench" ]; then
    PROJECT_DIR="/p/scratch/westai0070/$USER/tt-bench"
elif [ -d "$SCRIPT_DIR/.." ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    echo "ERROR: Cannot find project directory."
    echo "Expected at /p/scratch/westai0070/$USER/tt-bench or relative to script."
    exit 1
fi

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

banner() { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "  ${RED}✘${NC} $1"; }

# ── Model roster ─────────────────────────────────────────────────────────────
# Format: "MODEL_ID|short_name|GPU_count"
#
# ONLY instruction-tuned checkpoints belong here. A base/pre-trained model ships
# no chat template, so vLLM answers every /v1/chat/completions with a 400 and the
# whole run scores 0% without a single token generated. For Gemma that means the
# `-it` suffix; for Qwen, `-Instruct`.
#
# GPU_COUNT drives --tensor-parallel-size. dc-hwai allocates WHOLE nodes, so all
# 4 H100s are billed either way — extra GPUs are free KV-cache headroom. Budget
# ~2 bytes/param for bf16 weights and leave room for the 131072-token context:
#   Dense ≤14B      → 1 GPU
#   Dense 26-35B    → 4 GPUs (31B ≈ 62GB of weights; 1 GPU leaves ~10GB of KV
#                     cache at 0.90 util, which will not hold 131072 tokens)
#   MoE <50B total  → 1 GPU (only active params are resident per forward pass)

MODELS=(
    "Qwen/Qwen2.5-Coder-7B-Instruct|qwen2.5-coder-7b|1"
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct|deepseek-coder-v2-lite|1"
    "google/gemma-4-26B-A4B-it|gemma-4-26b-a4b|1"
    "Qwen/Qwen3.6-35B-A3B|qwen3.6-35b-a3b|1"
    # google/gemma-4-31B (base) is deliberately absent: no chat template, 400 on
    # every request. Its instruct sibling is the one to benchmark. Native context
    # is 262144, so no YaRN scaling is applied at TARGET_CTX=131072.
    "google/gemma-4-31B-it|gemma-4-31b-it|4"
)

# ── Defaults (override via flags) ────────────────────────────────────────────
TIERS="1"
RUN_SCALED=""
DRY_RUN=0
SELECTED=()

usage() {
    cat <<EOF
Usage: bash jureca/submit_all.sh [OPTIONS] [SHORT_NAME ...]

Submits one Slurm job per model. With no SHORT_NAME the whole roster runs;
name one or more short labels to submit just those.

Options:
  -l, --list       Print the roster and exit
  -n, --dry-run    Print the sbatch commands without submitting
  -t, --tiers T    Tier filter (default: "$TIERS")
  -s, --scaled     Also run the scaled sets (~1013 extra tasks, 8-12h alone)
  -h, --help       This message

Examples:
  bash jureca/submit_all.sh                      # everything
  bash jureca/submit_all.sh gemma-4-31b-it       # just one
  bash jureca/submit_all.sh -n -t "1 2"          # dry run, tiers 1 and 2
EOF
}

list_roster() {
    printf "  %-42s %-24s %s\n" "MODEL_ID" "SHORT_NAME" "GPUS"
    echo "  ------------------------------------------------------------------------------"
    local e mid mname gcount
    for e in "${MODELS[@]}"; do
        IFS='|' read -r mid mname gcount <<< "$e"
        printf "  %-42s %-24s %s\n" "$mid" "$mname" "$gcount"
    done
}

while [ $# -gt 0 ]; do
    case "$1" in
        -l|--list)    list_roster; exit 0 ;;
        -n|--dry-run) DRY_RUN=1; shift ;;
        -s|--scaled)  RUN_SCALED=1; shift ;;
        -t|--tiers)   TIERS="${2:?--tiers needs a value, e.g. -t '1 2'}"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        -*)           fail "Unknown option: $1"; echo ""; usage; exit 1 ;;
        *)            SELECTED+=("$1"); shift ;;
    esac
done

# Filter the roster down to the requested short names. Fail on an unknown name
# rather than silently submitting nothing — a typo should not look like success.
if [ ${#SELECTED[@]} -gt 0 ]; then
    FILTERED=()
    for want in "${SELECTED[@]}"; do
        match=""
        for entry in "${MODELS[@]}"; do
            IFS='|' read -r _mid mname _gc <<< "$entry"
            [ "$mname" = "$want" ] && match="$entry" && break
        done
        if [ -z "$match" ]; then
            fail "No roster entry named '$want'"
            echo ""
            list_roster
            exit 1
        fi
        FILTERED+=("$match")
    done
    MODELS=("${FILTERED[@]}")
fi

# ── Pre-flight ───────────────────────────────────────────────────────────────
banner "Pre-flight checks"

# Check if we're on JURECA login node
if hostname | grep -q "jrlogin"; then
    ok "Running on JURECA login node ($(hostname))"
else
    warn "Not on jrlogin — are you on JURECA?"
fi

# Check the venv by its INTERPRETER, not by bin/activate. When uv upgrades its
# managed CPython it prunes the old patch directory, leaving bin/python a
# dangling symlink while bin/activate stays perfectly readable — so the old
# `-f bin/activate` test passed and the job died on the compute node instead.
# Validate capability (does it run? does vLLM import?), not mere presence.
VENV_DIR="$PROJECT_DIR/.venv-ttbench"
VENV_PY="$VENV_DIR/bin/python"

if [ ! -x "$VENV_PY" ]; then
    fail "No usable interpreter at $VENV_PY — run jureca/setup.sh first"
    ls -l "$VENV_PY" 2>&1 | sed 's/^/    /'
    exit 1
fi
if ! PY_VERSION="$("$VENV_PY" --version 2>&1)" || [ -z "$PY_VERSION" ]; then
    fail "$VENV_PY exists but does not run — the venv is broken"
    exit 1
fi
ok "Interpreter: $PY_VERSION ($VENV_PY)"

# Last free chance to catch a broken install: on the login node, before the queue.
if "$VENV_PY" -c 'import vllm' >/dev/null 2>&1; then
    ok "vLLM importable: $("$VENV_PY" -c 'import vllm; print(vllm.__version__)')"
else
    fail "'import vllm' fails in $VENV_PY — fix it before burning a queue slot"
    exit 1
fi

# Check project
if [ -f "$PROJECT_DIR/pyproject.toml" ]; then
    ok "Project directory: $PROJECT_DIR"
else
    fail "Project not found at $PROJECT_DIR"
    exit 1
fi

# Check sbatch script
SBATCH_SCRIPT="$SCRIPT_DIR/run_benchmark.sbatch"
if [ -f "$SBATCH_SCRIPT" ]; then
    ok "Slurm script: $SBATCH_SCRIPT"
else
    fail "Slurm script not found"
    exit 1
fi

# Check for Slurm command
if ! command -v sbatch &>/dev/null; then
    fail "sbatch not found — are you on a JURECA login node?"
    exit 1
fi

# Ensure slurm log directory exists
mkdir -p "$PROJECT_DIR/slurm_logs"

# ── Model cache check (informational only, non-blocking) ─────────────────────
banner "HuggingFace cache"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
echo "  Cache directory: $HF_HOME"

# Wrap ls so a no-match glob doesn't trip `set -euo pipefail` and abort
# submission when the cache is empty.
CACHED_COUNT=$( { ls -1d "$HF_HOME/hub/models--"* 2>/dev/null || true; } | wc -l)
ok "Cache contains ${CACHED_COUNT} model(s)"

# ── Submit jobs ──────────────────────────────────────────────────────────────
banner "Submitting Slurm jobs (Tier $TIERS)"

if [ -n "$RUN_SCALED" ]; then
    warn "RUN_SCALED=1 — each job also runs the ~1013-task scaled sets."
    warn "That is 8-12h on its own; 12h of --time may not be enough."
fi
[ "$DRY_RUN" -eq 1 ] && warn "DRY RUN — nothing will be submitted"

SUBMITTED_JOBS=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id model_name gpu_count <<< "$entry"

    echo ""
    echo "  Submitting: $model_name ($model_id) — $gpu_count GPU(s)"

    # Every var run_benchmark.sbatch validates with ${VAR:?} is passed here.
    # RUN_SCALED is set EXPLICITLY on every submission, never just when asked for:
    # --export=ALL copies the whole submitting environment into the job, so a
    # leftover `export RUN_SCALED=1` from an earlier shell would otherwise ride
    # along and silently add ~1013 tasks per job. Later assignments win over ALL.
    EXPORTS="ALL,MODEL_ID=$model_id,MODEL_NAME=$model_name,GPU_COUNT=$gpu_count"
    EXPORTS="$EXPORTS,TIERS=$TIERS,PROJECT_DIR=$PROJECT_DIR"
    EXPORTS="$EXPORTS,RUN_SCALED=${RUN_SCALED:-0}"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "    sbatch --gres=gpu:${gpu_count} \\"
        echo "           --export=$EXPORTS \\"
        echo "           --parsable $SBATCH_SCRIPT"
        continue
    fi

    JOB_ID=$(sbatch \
        --gres="gpu:${gpu_count}" \
        --export="$EXPORTS" \
        --parsable \
        "$SBATCH_SCRIPT")

    if [ -n "$JOB_ID" ]; then
        ok "Submitted as job $JOB_ID"
        SUBMITTED_JOBS+=("$JOB_ID|$model_name")
    else
        fail "Failed to submit $model_name"
    fi

    # Brief pause to avoid overwhelming the scheduler
    sleep 1
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    ok "Dry run complete — ${#MODELS[@]} job(s) would be submitted"
    exit 0
fi

# ── Summary ──────────────────────────────────────────────────────────────────
banner "Submission complete"

echo ""
echo "  Submitted ${#SUBMITTED_JOBS[@]} / ${#MODELS[@]} jobs:"
echo ""

printf "  %-10s %-25s %s\n" "Job ID" "Model" "Status"
echo "  -----------------------------------------------------"
# Guarded expansion: under `set -u` a bare "${ARR[@]}" on an EMPTY array is an
# unbound-variable error in bash < 4.4, and this array is empty if every sbatch
# call failed — exactly the moment you least want the summary to crash.
for entry in ${SUBMITTED_JOBS[@]+"${SUBMITTED_JOBS[@]}"}; do
    IFS='|' read -r jid mname <<< "$entry"
    printf "  %-10s %-25s PENDING\n" "$jid" "$mname"
done

echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Logs:     $PROJECT_DIR/slurm_logs/"
echo "  Results:  $PROJECT_DIR/benchmark_results/jureca_tier1/"
echo ""
echo "  After all jobs finish, aggregate results:"
echo "    bash jureca/aggregate_results.sh"
echo ""
