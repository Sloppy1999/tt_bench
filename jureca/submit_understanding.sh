#!/usr/bin/env bash
# =============================================================================
# Submit the procedural-understanding item bank for the 7-model JURECA roster
# =============================================================================
# Sibling to submit_all.sh, trimmed to what run_understanding.sbatch actually
# needs: no turn budget, harness-ablation arm, challenge-set sweep or sample
# count — the understanding task is a single-shot QA pass over one fixed item
# bank (review/understanding_study/PLAN.md), not the agentic loop.
#
# Usage:
#   bash jureca/submit_understanding.sh                # whole roster
#   bash jureca/submit_understanding.sh gemma-4-31b-it # one model
#   bash jureca/submit_understanding.sh --list         # show the roster
#   bash jureca/submit_understanding.sh -n             # dry run
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -n "${PROJECT_DIR:-}" ]; then
    :
elif [ -d "/p/scratch/westai0070/$USER/tt-bench" ]; then
    PROJECT_DIR="/p/scratch/westai0070/$USER/tt-bench"
elif [ -d "/p/scratch/westai0070/$USER/tt_bench" ]; then
    PROJECT_DIR="/p/scratch/westai0070/$USER/tt_bench"
elif [ -d "$SCRIPT_DIR/.." ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    echo "ERROR: Cannot find project directory."
    exit 1
fi

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
banner() { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "  ${RED}✘${NC} $1"; }

# Same roster as submit_all.sh (see that file for how GPU_COUNT was picked per
# model). Kept as a literal copy rather than a shared source: the two scripts
# run different task types and drifting the roster in lockstep is not a goal.
MODELS=(
    "Qwen/Qwen2.5-Coder-7B-Instruct|qwen2.5-coder-7b|4"
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct|deepseek-coder-v2-lite|1"
    "google/gemma-4-26B-A4B-it|gemma-4-26b-a4b|1"
    "google/gemma-4-31B-it|gemma-4-31b-it|4"
    "Qwen/Qwen3.6-35B-A3B|qwen3.6-35b-a3b|4"
    "openai/gpt-oss-120b|gpt-oss-120b|4"
    "cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4|qwen3.8-flash-next-awq4|4"
)

DRY_RUN=0
SELECTED=()
WALLTIME=""
MAIL_USER="${TT_BENCH_MAIL:-}"
MAIL_TYPE="${TT_BENCH_MAIL_TYPE:-END,FAIL}"

usage() {
    cat <<EOF
Usage: bash jureca/submit_understanding.sh [OPTIONS] [SHORT_NAME ...]

Submits one Slurm job per model, each running the procedural-understanding
item bank (official set, 165 runnable items) once. With no SHORT_NAME the
whole roster runs.

Options:
  -l, --list       Print the roster and exit
  -n, --dry-run    Print the sbatch commands without submitting
      --time D     Slurm walltime, overriding the script's 2h default
      --mail A     Email address for Slurm job notifications
      --mail-type T  When to notify (default: "$MAIL_TYPE")
  -h, --help       This message
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
        --time)       WALLTIME="${2:?--time needs a Slurm duration, e.g. --time 04:00:00}"; shift 2 ;;
        --mail)       MAIL_USER="${2:?--mail needs an address}"; shift 2 ;;
        --mail-type)  MAIL_TYPE="${2:?--mail-type needs a value, e.g. BEGIN,END,FAIL}"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        -*)           fail "Unknown option: $1"; echo ""; usage; exit 1 ;;
        *)            SELECTED+=("$1"); shift ;;
    esac
done

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

if hostname | grep -q "jrlogin"; then
    ok "Running on JURECA login node ($(hostname))"
else
    warn "Not on jrlogin — are you on JURECA?"
fi

VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-ttbench}"
VENV_PY="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PY" ]; then
    fail "No usable interpreter at $VENV_PY — run jureca/setup.sh first"
    exit 1
fi
if ! PY_VERSION="$("$VENV_PY" --version 2>&1)" || [ -z "$PY_VERSION" ]; then
    fail "$VENV_PY exists but does not run — the venv is broken"
    exit 1
fi
ok "Interpreter: $PY_VERSION ($VENV_PY)"

if "$VENV_PY" -c 'import vllm' >/dev/null 2>&1; then
    ok "vLLM importable: $("$VENV_PY" -c 'import vllm; print(vllm.__version__)')"
else
    fail "'import vllm' fails in $VENV_PY — fix it before burning a queue slot"
    exit 1
fi

if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
    fail "Project not found at $PROJECT_DIR"
    exit 1
fi
ok "Project directory: $PROJECT_DIR"

SBATCH_SCRIPT="$SCRIPT_DIR/run_understanding.sbatch"
if [ ! -f "$SBATCH_SCRIPT" ]; then
    fail "Slurm script not found: $SBATCH_SCRIPT"
    exit 1
fi
ok "Slurm script: $SBATCH_SCRIPT"

if ! command -v sbatch &>/dev/null; then
    fail "sbatch not found — are you on a JURECA login node?"
    exit 1
fi

mkdir -p "$PROJECT_DIR/slurm_logs"

N_QUESTIONS=$(ls -1 "$PROJECT_DIR"/data/tasks/official/challenges/json/tt-official-ch*.json 2>/dev/null | wc -l)
if [ "$N_QUESTIONS" -eq 0 ]; then
    fail "No tt-official-ch*.json files under $PROJECT_DIR/data/tasks/official/challenges/json"
    exit 1
fi
ok "Official challenge set: $N_QUESTIONS file(s)"

# ── Model readiness (blocking) ───────────────────────────────────────────────
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
banner "Model readiness"
READY_FAIL=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id model_name _gc <<< "$entry"
    repo="models--${model_id//\//--}"
    snap=$( { ls -d "$HF_HOME/hub/$repo/snapshots/"*/ 2>/dev/null || true; } | head -n1)

    if [ -z "$snap" ]; then
        fail "$model_name: not cached under $HF_HOME/hub/$repo"
        echo "        Pre-download on the login node: HF_HOME=$HF_HOME hf download $model_id"
        READY_FAIL=1
        continue
    fi

    case "$model_id" in
        *gpt-oss*|*gpt_oss*)
            TK="${TIKTOKEN_CACHE_DIR:-$PROJECT_DIR/.cache/tiktoken}"
            if [ -z "$( { ls -A "$TK" 2>/dev/null || true; } )" ]; then
                fail "$model_name: harmony/tiktoken vocab not cached in $TK"
                echo "        Compute nodes cannot download it. Populate it on the login node first."
                READY_FAIL=1
                continue
            fi
            ok "$model_name: harmony vocab cached"
            ;;
    esac

    if [ -f "${snap}chat_template.jinja" ] \
       || grep -q 'chat_template' "${snap}tokenizer_config.json" 2>/dev/null; then
        ok "$model_name: cached, chat template present"
    else
        fail "$model_name: cached but NO chat template — looks like a base checkpoint"
        READY_FAIL=1
    fi
done

if [ "$READY_FAIL" -eq 1 ]; then
    echo ""
    fail "Refusing to submit: fix the models flagged above first."
    exit 1
fi

# ── Submit jobs ──────────────────────────────────────────────────────────────
banner "Submitting Slurm jobs (understanding, time ${WALLTIME:-02:00:00})"

if [ -n "$MAIL_USER" ]; then
    ok "Notifications: $MAIL_TYPE → $MAIL_USER"
else
    warn "No notifications: set TT_BENCH_MAIL or pass --mail to get one on END/FAIL"
fi
[ "$DRY_RUN" -eq 1 ] && warn "DRY RUN — nothing will be submitted"

SUBMITTED_JOBS=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id model_name gpu_count <<< "$entry"

    echo ""
    echo "  Submitting: $model_name ($model_id) — $gpu_count GPU(s)"

    EXPORTS="ALL,MODEL_ID=$model_id,MODEL_NAME=$model_name,GPU_COUNT=$gpu_count,PROJECT_DIR=$PROJECT_DIR"

    TIME_ARGS=()
    [ -n "$WALLTIME" ] && TIME_ARGS=(--time "$WALLTIME")

    NAME_ARGS=(--job-name "tt-understand-$model_name")

    MAIL_ARGS=()
    [ -n "$MAIL_USER" ] && MAIL_ARGS=(--mail-type "$MAIL_TYPE" --mail-user "$MAIL_USER")

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "    sbatch --gres=gpu:${gpu_count} --job-name tt-understand-$model_name \\"
        [ -n "$WALLTIME" ] && echo "           --time $WALLTIME \\"
        [ -n "$MAIL_USER" ] && echo "           --mail-type $MAIL_TYPE --mail-user $MAIL_USER \\"
        echo "           --export=$EXPORTS \\"
        echo "           --parsable $SBATCH_SCRIPT"
        continue
    fi

    JOB_ID=$(sbatch \
        --gres="gpu:${gpu_count}" \
        "${NAME_ARGS[@]}" \
        ${TIME_ARGS[@]+"${TIME_ARGS[@]}"} \
        ${MAIL_ARGS[@]+"${MAIL_ARGS[@]}"} \
        --export="$EXPORTS" \
        --parsable \
        "$SBATCH_SCRIPT")

    if [ -n "$JOB_ID" ]; then
        ok "Submitted as job $JOB_ID"
        SUBMITTED_JOBS+=("$JOB_ID|$model_name")
    else
        fail "Failed to submit $model_name"
    fi
    sleep 1
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    ok "Dry run complete — ${#MODELS[@]} job(s) would be submitted"
    exit 0
fi

banner "Submission complete"
echo ""
echo "  Submitted ${#SUBMITTED_JOBS[@]} / ${#MODELS[@]} jobs:"
echo ""
printf "  %-10s %-25s %s\n" "Job ID" "Model" "Status"
echo "  -----------------------------------------------------"
for entry in ${SUBMITTED_JOBS[@]+"${SUBMITTED_JOBS[@]}"}; do
    IFS='|' read -r jid mname <<< "$entry"
    printf "  %-10s %-25s PENDING\n" "$jid" "$mname"
done
echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Logs:     $PROJECT_DIR/slurm_logs/"
echo "  Results:  $PROJECT_DIR/review/understanding_study/runs/"
