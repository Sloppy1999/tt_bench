#!/usr/bin/env bash
# =============================================================================
# Submit Multiple Models for Tier 1 TT-Bench on JURECA H100s
# =============================================================================
# Submits one Slurm job per model, each running the full Tier 1 benchmark
# across official, 1comp, and 2comp challenge sets.
#
# Usage:
#   chmod +x jureca/submit_all.sh
#   bash jureca/submit_all.sh
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
# GPU count guidelines for H100 80GB:
#   7-8B models  → 1 GPU
#   14-16B MoE   → 1 GPU
#   32B models   → 1 GPU
#   70B models   → 2 GPUs (tensor parallel)
#   123B+ MoE    → 4 GPUs

MODELS=(
    # Lightweight (1 GPU each) — good baselines
    "Qwen/Qwen2.5-Coder-7B-Instruct|qwen2.5-coder-7b|1"
    "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b|1"
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct|deepseek-coder-v2-lite|1"

    # Mid-size (1 GPU each) — strong syntehsis
    "Qwen/Qwen2.5-Coder-32B-Instruct|qwen2.5-coder-32b|1"
    "mistralai/Mistral-Large-Instruct-2411|mistral-large-2411|1"

    # Large (2 GPUs) — strongest open models
    "meta-llama/Llama-3.1-70B-Instruct|llama-3.1-70b|2"
)

# ── Tier selection ────────────────────────────────────────────────────────────
TIERS="1"

# ── Pre-flight ───────────────────────────────────────────────────────────────
banner "Pre-flight checks"

# Check if we're on JURECA login node
if hostname | grep -q "jrlogin"; then
    ok "Running on JURECA login node ($(hostname))"
else
    warn "Not on jrlogin — are you on JURECA?"
fi

# Check venv
VENV_DIR="$HOME/.venv-ttbench"
if [ -f "$VENV_DIR/bin/activate" ]; then
    ok "Virtual environment found at $VENV_DIR"
else
    fail "Virtual environment not found — run jureca/setup.sh first"
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

# ── Model cache check (optional, non-blocking) ───────────────────────────────
banner "HuggingFace cache"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
echo "  Cache directory: $HF_HOME"

source "$VENV_DIR/bin/activate"

if command -v huggingface-cli &>/dev/null; then
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r model_id _ _ <<< "$entry"
        MODEL_DIR="$HF_HOME/hub/models--$(echo "$model_id" | tr '/' '--')"
        if [ -d "$MODEL_DIR" ]; then
            ok "Cached: $model_id"
        else
            warn "Not cached: $model_id (will download on compute node — may use job time)"
        fi
    done
else
    warn "huggingface-cli not found — models will download on compute nodes"
fi

# ── Submit jobs ──────────────────────────────────────────────────────────────
banner "Submitting Slurm jobs (Tier $TIERS)"

SUBMITTED_JOBS=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id model_name gpu_count <<< "$entry"

    echo ""
    echo "  Submitting: $model_name ($model_id) — $gpu_count GPU(s)"

    JOB_ID=$(sbatch \
        --export=ALL,MODEL_ID="$model_id",MODEL_NAME="$model_name",GPU_COUNT="$gpu_count",TIERS="$TIERS",PROJECT_DIR="$PROJECT_DIR" \
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

# ── Summary ──────────────────────────────────────────────────────────────────
banner "Submission complete"

echo ""
echo "  Submitted ${#SUBMITTED_JOBS[@]} / ${#MODELS[@]} jobs:"
echo ""

printf "  %-10s %-25s %s\n" "Job ID" "Model" "Status"
echo "  -----------------------------------------------------"
for entry in "${SUBMITTED_JOBS[@]}"; do
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
