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
    # google/gemma-4-31B (base) is deliberately absent: no chat template, 400 on
    # every request. Its instruct sibling is the one to benchmark. Native context
    # is 262144, so no YaRN scaling is applied at TARGET_CTX=131072.
    "google/gemma-4-31B-it|gemma-4-31b-it|4"
    # 35B dense-equivalent weights (~70GB in bf16) do NOT fit one H100 alongside
    # a 131072-token KV cache. Was listed at 1 GPU, which cannot have worked.
    "Qwen/Qwen3.6-35B-A3B|qwen3.6-35b-a3b|4"
    # gpt-oss ships in MXFP4 (~63GB) so it would fit one H100, but dc-hwai bills
    # whole nodes regardless — the extra GPUs are free KV-cache headroom.
    # Needs its own --tool-call-parser: it emits the harmony format, not hermes.
    "openai/gpt-oss-120b|gpt-oss-120b|4"
)

# ── Defaults (override via flags) ────────────────────────────────────────────
TIERS="1"
RUN_SCALED=""
DRY_RUN=0
SELECTED=()
MAX_TURNS=""        # empty → run_benchmark.sbatch's default of 25
SETS=""             # empty → all challenge sets
WALLTIME=""         # empty → the #SBATCH --time in run_benchmark.sbatch (12h)

usage() {
    cat <<EOF
Usage: bash jureca/submit_all.sh [OPTIONS] [SHORT_NAME ...]

Submits one Slurm job per model. With no SHORT_NAME the whole roster runs;
name one or more short labels to submit just those.

Options:
  -l, --list       Print the roster and exit
  -n, --dry-run    Print the sbatch commands without submitting
  -t, --tiers T    Tier filter (default: "$TIERS")
      --turns N    Agent turn budget (default: 25). Results for N != 25 land in
                   <set>_tN/ so a sweep cannot overwrite itself.
      --sets LIST  Challenge sets to run, comma-separated (default: all).
                   Labels: official, 1comp, 2comp, and with -s also
                   scaled, scaled_1comp, scaled_2comp
      --time D     Slurm walltime, overriding the script's 12h default.
                   -s needs far more than 12h; check the partition ceiling with
                   scontrol show partition dc-hwai | grep MaxTime
  -s, --scaled     Also run the scaled sets. That is ~2900 tier-1 tasks on top
                   of the 21 small ones — hours per model, not minutes.
  -h, --help       This message

Examples:
  bash jureca/submit_all.sh                      # everything
  bash jureca/submit_all.sh gemma-4-31b-it       # just one
  bash jureca/submit_all.sh -n -t "1 2"          # dry run, tiers 1 and 2

  # Turn-budget sweep on the 1-component boards:
  for n in 25 50 100; do
    bash jureca/submit_all.sh --sets 1comp --turns \$n gemma-4-31b-it
  done
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
        --turns)      MAX_TURNS="${2:?--turns needs a number, e.g. --turns 50}"; shift 2 ;;
        --sets)       SETS="${2:?--sets needs a list, e.g. --sets 1comp}"; shift 2 ;;
        --time)       WALLTIME="${2:?--time needs a Slurm duration, e.g. --time 24:00:00}"; shift 2 ;;
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

# ── Model cache check ────────────────────────────────────────────────────────
banner "HuggingFace cache"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
echo "  Cache directory: $HF_HOME"

# Wrap ls so a no-match glob doesn't trip `set -euo pipefail` and abort
# submission when the cache is empty.
CACHED_COUNT=$( { ls -1d "$HF_HOME/hub/models--"* 2>/dev/null || true; } | wc -l)
ok "Cache contains ${CACHED_COUNT} model(s)"

# ── Model readiness (blocking) ───────────────────────────────────────────────
# Two failures that each cost a full queue wait to discover, checked here for free:
#
#   1. Not cached. run_benchmark.sbatch exports HF_HUB_OFFLINE=1 because compute
#      nodes have no internet, so a missing model cannot be fetched at run time.
#
#   2. No chat template. A base (non-instruct) checkpoint ships none, and vLLM
#      then answers every /v1/chat/completions with a 400: the run scores 0%
#      without generating a single token, and nothing in the Slurm log says why.
#      google/gemma-4-31B burned a 12-hour allocation on exactly this.
banner "Model readiness"
READY_FAIL=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id model_name _gc <<< "$entry"
    repo="models--${model_id//\//--}"
    snap=$( { ls -d "$HF_HOME/hub/$repo/snapshots/"*/ 2>/dev/null || true; } | head -n1)

    if [ -z "$snap" ]; then
        fail "$model_name: not cached under $HF_HOME/hub/$repo"
        echo "        Pre-download on the login node (compute nodes are offline):"
        echo "        HF_HOME=$HF_HOME hf download $model_id"
        READY_FAIL=1
        continue
    fi

    # gpt-oss tokenises with the tiktoken/harmony encoding, fetched over the
    # network on first use instead of read from the snapshot. Compute nodes have
    # no route out, so an unpopulated cache means every request returns
    # 500 "error downloading or loading vocab file" — and the run completes with
    # exit 0, zero turns and a 0% score that looks like a measurement.
    case "$model_id" in
        *gpt-oss*|*gpt_oss*)
            TK="${TIKTOKEN_CACHE_DIR:-$PROJECT_DIR/.cache/tiktoken}"
            if [ -z "$( { ls -A "$TK" 2>/dev/null || true; } )" ]; then
                fail "$model_name: harmony/tiktoken vocab not cached in $TK"
                echo "        Compute nodes cannot download it. Populate it here first:"
                echo "        export TIKTOKEN_CACHE_DIR=$TK && mkdir -p \"\$TIKTOKEN_CACHE_DIR\""
                echo "        python -c 'from openai_harmony import load_harmony_encoding,HarmonyEncodingName as N; load_harmony_encoding(N.HARMONY_GPT_OSS)'"
                echo "        Then check the directory is NON-EMPTY. If it stays empty the env"
                echo "        var name is wrong — find the one harmony actually reads."
                READY_FAIL=1
                continue
            fi
            ok "$model_name: harmony vocab cached ($(ls -A "$TK" | wc -l | tr -d ' ') file(s))"
            ;;
    esac

    if [ -f "${snap}chat_template.jinja" ] \
       || grep -q 'chat_template' "${snap}tokenizer_config.json" 2>/dev/null; then
        ok "$model_name: cached, chat template present"
    else
        fail "$model_name: cached but NO chat template — this looks like a base checkpoint"
        echo "        vLLM will reject every chat request with a 400. Use the"
        echo "        instruction-tuned variant instead (-it, -Instruct)."
        READY_FAIL=1
    fi
done

if [ "$READY_FAIL" -eq 1 ]; then
    echo ""
    fail "Refusing to submit: fix the models flagged above first."
    exit 1
fi

# ── Submit jobs ──────────────────────────────────────────────────────────────
banner "Submitting Slurm jobs (Tier $TIERS, turns ${MAX_TURNS:-25}, sets ${SETS:-all}, time ${WALLTIME:-12:00:00})"

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
    # Same reasoning: pin these on every submission so a stale MAX_TURNS or SETS
    # in the submitting shell cannot ride in via --export=ALL and silently change
    # what the experiment measures.
    EXPORTS="$EXPORTS,MAX_TURNS=${MAX_TURNS:-25},SETS=${SETS:-}"

    TIME_ARGS=()
    [ -n "$WALLTIME" ] && TIME_ARGS=(--time "$WALLTIME")

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "    sbatch --gres=gpu:${gpu_count} \\"
        [ -n "$WALLTIME" ] && echo "           --time $WALLTIME \\"
        echo "           --export=$EXPORTS \\"
        echo "           --parsable $SBATCH_SCRIPT"
        continue
    fi

    JOB_ID=$(sbatch \
        --gres="gpu:${gpu_count}" \
        ${TIME_ARGS[@]+"${TIME_ARGS[@]}"} \
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
