#!/usr/bin/env bash
# =============================================================================
# Tier 1 Agentic Synthesis — AcademicCloud Models
# =============================================================================
# Fetches available models from the AcademicCloud API, lets you pick one
# or run all, then benchmarks across three challenge directories.
#
# Usage:
#   ./scripts/run_cloud_benchmark.sh
#
# Prerequisites:
#   - CLOUD_API_KEY in .env (or exported)
#   - uv installed
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CLOUD_BASE="https://chat-ai.academiccloud.de"
RUN_ID=$(date -u +%Y-%m-%dT%H%M%S)
RESULTS_DIR="benchmark_results/cloud_comparison/${RUN_ID}"
SUMMARY_FILE="$RESULTS_DIR/summary.json"

# ── Colour helpers ──────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

banner() { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "  ${RED}✘${NC} $1"; }
info()  { echo -e "  ${CYAN}→${NC} $1"; }

# ── Load API key ────────────────────────────────────────────────────────────
load_api_key() {
    # Try exported env first, then .env file
    if [ -n "${CLOUD_API_KEY:-}" ]; then
        return 0
    fi
    if [ -f "$PROJECT_ROOT/.env" ]; then
        # shellcheck disable=SC1091
        CLOUD_API_KEY=$(grep '^CLOUD_API_KEY=' "$PROJECT_ROOT/.env" | cut -d= -f2- | tr -d '"' | tr -d "'" || true)
    fi
    if [ -z "${CLOUD_API_KEY:-}" ]; then
        fail "CLOUD_API_KEY not found. Set it in .env or export it."
        exit 1
    fi
}

# ── Fetch available models from API ─────────────────────────────────────────
fetch_models() {
    local models_json
    models_json=$(curl -s --max-time 10 \
        -H "Authorization: Bearer $CLOUD_API_KEY" \
        "$CLOUD_BASE/v1/models" 2>/dev/null || true)

    if [ -z "$models_json" ]; then
        return 1
    fi

    # Extract model IDs (status=ready, text-only LLMs — skip omni/vision models)
    echo "$models_json" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for m in data.get('data', []):
        if m.get('status') != 'ready':
            continue
        mid = m.get('id', '')
        # Skip vision-only / omni / medical models
        if any(x in mid.lower() for x in ('internvl', 'medgemma', 'omni')):
            continue
        name = m.get('name', mid)
        demand = m.get('demand', 0)
        print(f'{mid}|{name}|{demand}')
except Exception as e:
    print(f'ERROR:{e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null || return 1
}

# ── Interactive model selection ─────────────────────────────────────────────
select_models() {
    local fetched
    fetched=$(fetch_models) || true

    if [ -z "$fetched" ] || echo "$fetched" | grep -q '^ERROR:'; then
        warn "Could not fetch models from API — using cached list"
        fetched="qwen3-coder-30b-a3b-instruct|Qwen 3 Coder 30B A3B Instruct|0
devstral-2-123b-instruct-2512|Devstral 2 123B Instruct 2512|0
deepseek-r1-distill-llama-70b|DeepSeek R1 Distill Llama 70B|0
mistral-large-3-675b-instruct-2512|Mistral Large 3 675B Instruct 2512|0
qwen3.5-122b-a10b|Qwen 3.5 122B A10B|0
qwen3.5-397b-a17b|Qwen 3.5 397B A17B|0
glm-4.7|GLM-4.7|0
meta-llama-3.1-8b-instruct|Meta Llama 3.1 8B Instruct|0
gemma-4-31b-it|Gemma 4 31B Instruct|0
qwen3-30b-a3b-instruct-2507|Qwen 3 30B A3B Instruct 2507|0
qwen3.6-35b-a3b|Qwen 3.6 35B A3B|0
openai-gpt-oss-120b|OpenAI GPT OSS 120B|0
apertus-70b-instruct-2509|Apertus 70B Instruct 2509|0
teuken-7b-instruct-research|Teuken 7B Instruct Research|0"
    fi

    # Build indexed array (bare assignment = global in bash 3.x)
    MODEL_IDS=()
    MODEL_NAMES=()
    local count=0

    while IFS='|' read -r mid mname _; do
        [ -z "$mid" ] && continue
        ((count++))
        MODEL_IDS+=("$mid")
        MODEL_NAMES+=("$mname")
    done <<< "$fetched"

    if [ "$count" -eq 0 ]; then
        fail "No models available"
        exit 1
    fi

    echo ""
    echo -e "${BOLD}Available models (${count}):${NC}"
    echo ""
    for i in "${!MODEL_IDS[@]}"; do
        printf "  %2d) %-45s %s\n" "$((i + 1))" "${MODEL_IDS[$i]}" "${MODEL_NAMES[$i]}"
    done
    printf "  %2d) %s\n" "$((count + 1))" "ALL MODELS"
    echo ""

    local choice
    while true; do
        read -r -p "Select model [1-${count} or 'all']: " choice
        choice=$(echo "$choice" | tr '[:upper:]' '[:lower:]')

        # "all" or "a"
        if [ "$choice" = "all" ] || [ "$choice" = "a" ]; then
            SELECTED_MODELS=("${MODEL_IDS[@]}")
            echo ""
            info "Running ALL ${count} models"
            return 0
        fi

        # Single number, comma-separated list (e.g. "1,5,10"), or range (e.g. "1-5")
        if [[ "$choice" =~ ^[0-9,[:space:]-]+$ ]]; then
            # Split on commas (and optional whitespace)
            local -a indices=()
            local invalid=false
            IFS=',' read -r -a raw_parts <<< "$choice"
            for part in "${raw_parts[@]}"; do
                part=$(echo "$part" | xargs)  # trim whitespace
                # Range like "1-5"
                if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                    local start="${BASH_REMATCH[1]}"
                    local end="${BASH_REMATCH[2]}"
                    if [ "$start" -lt 1 ] || [ "$end" -gt "$count" ] || [ "$start" -gt "$end" ]; then
                        invalid=true
                        break
                    fi
                    for ((i = start; i <= end; i++)); do
                        indices+=("$i")
                    done
                elif [[ "$part" =~ ^[0-9]+$ ]]; then
                    if [ "$part" -lt 1 ] || [ "$part" -gt "$count" ]; then
                        invalid=true
                        break
                    fi
                    indices+=("$part")
                else
                    invalid=true
                    break
                fi
            done

            if [ "$invalid" = false ] && [ ${#indices[@]} -gt 0 ]; then
                SELECTED_MODELS=()
                local names=()
                for idx in "${indices[@]}"; do
                    SELECTED_MODELS+=("${MODEL_IDS[$((idx - 1))]}")
                    names+=("${MODEL_IDS[$((idx - 1))]}")
                done
                echo ""
                info "Selected ${#SELECTED_MODELS[@]} model(s): ${names[*]}"
                return 0
            fi
        fi

        warn "Invalid choice '$choice'. Enter 1-${count}, comma-separated (e.g. 1,5,10), ranges (e.g. 1-3), or 'all'."
    done
}

# ── Pre-flight ──────────────────────────────────────────────────────────────
banner "Pre-flight checks"

load_api_key
ok "API key loaded"

mkdir -p "$RESULTS_DIR"

# ── Run configuration ───────────────────────────────────────────────────────
SELECTED_MODELS=()  # populated by select_models below
select_models

# Guard against empty selection
if [ ${#SELECTED_MODELS[@]} -eq 0 ]; then
    fail "No models selected"
    exit 1
fi

CHALLENGE_SETS=(
    "data/tasks/official/challenges/json|tt-official-ch*.json|official"
    "data/tasks/challenges_1comp|tt-official-ch*.json|1comp"
    "data/tasks/challenges_2comp|tt-official-ch*.json|2comp"
    "data/tasks/challenges_1comp/variants|tt-official-ch*.json|1comp_var"
    "data/tasks/challenges_2comp/variants|tt-official-ch*.json|2comp_var"
)

ARGS_COMMON="--provider cloud --task-type agentic_synthesis --save-report --max-turns 25 --tiers 1"

# ── Run benchmark ────────────────────────────────────────────────────────────
ALL_REPORTS=()

for model_id in "${SELECTED_MODELS[@]}"; do
    # Derive a short label: strip common prefixes/suffixes
    label=$(echo "$model_id" | sed 's/-instruct//; s/-it$//; s/^meta-//; s/^qwen/qwen/; s/-a3b//; s/-a10b//; s/-a17b//')

    banner "Model: $label"

    for cset in "${CHALLENGE_SETS[@]}"; do
        IFS='|' read -r challenges_dir pattern cset_label <<< "$cset"

        OUT_DIR="$RESULTS_DIR/${label}/${cset_label}"
        mkdir -p "$OUT_DIR"

        echo -e "  Running: ${cset_label} (pattern: $pattern)"

        set +e
        uv run tt-benchmark \
            --model "$model_id" \
            --challenges-dir "$challenges_dir" \
            --pattern "$pattern" \
            --output-dir "$OUT_DIR" \
            $ARGS_COMMON 2>&1
        EXIT_CODE=$?
        set -e

        if [ $EXIT_CODE -eq 0 ]; then
            ok "$cset_label complete"
        else
            fail "$cset_label failed (exit code $EXIT_CODE)"
        fi

        for f in "$OUT_DIR"/benchmark_*.json; do
            [ -f "$f" ] && ALL_REPORTS+=("$f")
        done
    done
done

# ── Aggregate results ────────────────────────────────────────────────────────
banner "Aggregating results"

python3 << PYEOF
import json, sys
from pathlib import Path
from collections import defaultdict

results_dir = Path("${RESULTS_DIR}")
reports = sorted(results_dir.rglob("benchmark_*.json"))

if not reports:
    print("No reports found!")
    sys.exit(0)

summary = {"experiment": "tier1_cloud_comparison", "models": {}}

for rp in reports:
    with open(rp) as f:
        data = json.load(f)

    parts = rp.relative_to(results_dir).parts
    model_label = parts[0]
    challenge_set = parts[1]

    if model_label not in summary["models"]:
        summary["models"][model_label] = {
            "provider": data.get("provider", ""),
            "model": data.get("model", ""),
            "by_set": {},
            "overall": {"total": 0, "successful": 0, "failed": 0},
        }

    entry = summary["models"][model_label]

    if challenge_set not in entry["by_set"]:
        entry["by_set"][challenge_set] = {"total": 0, "successful": 0, "failed": 0}

    s = entry["by_set"][challenge_set]
    s["total"] += data.get("total_tasks", 0)
    s["successful"] += data.get("successful", 0)
    s["failed"] += data.get("failed", 0)

    entry["overall"]["total"] += data.get("total_tasks", 0)
    entry["overall"]["successful"] += data.get("successful", 0)
    entry["overall"]["failed"] += data.get("failed", 0)

    if "tasks" not in entry:
        entry["tasks"] = []
    for task in data.get("results", []):
        entry["tasks"].append({
            "task_id": task.get("task_id"),
            "challenge_set": challenge_set,
            "success": task.get("success"),
            "component_score": task.get("component_score"),
            "tool_calls": len(task.get("predicted", {}).get("tool_calls", [])),
        })

for model_label, mdata in summary["models"].items():
    for cset, csdata in mdata["by_set"].items():
        csdata["success_rate"] = round(csdata["successful"] / csdata["total"] * 100, 1) if csdata["total"] else 0.0
    o = mdata["overall"]
    o["success_rate"] = round(o["successful"] / o["total"] * 100, 1) if o["total"] else 0.0

with open(results_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 80)
print("  TIER 1 AGENTIC SYNTHESIS — CLOUD MODELS")
print("=" * 80)
print()

header = f"{'Model':<30} {'Set':<10} {'Success':>8} {'Total':>6} {'Rate':>8}"
print(header)
print("-" * len(header))

for model_label, mdata in sorted(summary["models"].items()):
    for cset in ["official", "1comp", "1comp_var", "2comp", "2comp_var"]:
        csdata = mdata["by_set"].get(cset)
        if csdata:
            print(f"{model_label:<30} {cset:<10} {csdata['successful']:>6} / {csdata['total']:>4}  {csdata['success_rate']:>6.1f}%")
    o = mdata["overall"]
    print(f"{'':30} {'OVERALL':<10} {o['successful']:>6} / {o['total']:>4}  {o['success_rate']:>6.1f}%")
    print()

print("=" * 80)

print()
print("PER-TASK MATRIX (✔ = solved, ✘ = failed)")
print("-" * 55)

all_task_ids = sorted(set(
    t["task_id"] for mdata in summary["models"].values() for t in mdata.get("tasks", [])
))

matrix = defaultdict(lambda: defaultdict(lambda: "—"))
for model_label, mdata in summary["models"].items():
    for t in mdata.get("tasks", []):
        matrix[model_label][t["task_id"]] = "✔" if t["success"] else "✘"

col_width = max(len(t) for t in all_task_ids)
row_fmt = f"  {{:<{col_width}}}  " + "  ".join(f"{{:<{len(m)}}}" for m in summary["models"])

print(row_fmt.format("Challenge", *summary["models"].keys()))
print("  " + "-" * (col_width + 2 + sum(len(m) + 2 for m in summary["models"])))
for tid in all_task_ids:
    vals = [matrix[m][tid] for m in summary["models"]]
    print(row_fmt.format(tid, *vals))
PYEOF

echo ""
ok "Summary saved to: $SUMMARY_FILE"
echo ""
banner "Experiment complete"
echo "Results: $RESULTS_DIR/"
find "$RESULTS_DIR" -name "benchmark_*.json" | wc -l | xargs echo "  Reports:"
echo "  Summary: $SUMMARY_FILE"
