#!/usr/bin/env bash
# =============================================================================
# Tier 1 Agentic Synthesis Comparison Experiment
# =============================================================================
# Compares deepseek-v4-pro, 3 AcademicCloud models (qwen3-coder, devstral-2,
# deepseek-r1-70b), plus local LMStudio models across three challenge
# directories (official, 1comp, 2comp).
#
# Usage:
#   chmod +x scripts/run_tier1_comparison.sh
#   ./scripts/run_tier1_comparison.sh
#
# Prerequisites:
#   - DEEPSEEK_API_KEY env var set
#   - CLOUD_API_KEY env var set (AcademicCloud)
#   - LMStudio running at http://localhost:1234 with both models loaded
#   - uv installed
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="benchmark_results/tier1_comparison"
SUMMARY_FILE="$RESULTS_DIR/summary.json"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Colour helpers ──────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Colour

banner() {
    echo -e "\n${BOLD}━━━ $1 ━━━${NC}"
}

ok()   { echo -e "  ${GREEN}✔${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✘${NC} $1"; }

# ── Pre-flight checks ───────────────────────────────────────────────────────
banner "Pre-flight checks"

ok "API keys loaded from .env by the benchmark runner"

if curl -s --max-time 3 http://localhost:1234/v1/models > /dev/null 2>&1; then
    MODELS=$(curl -s http://localhost:1234/v1/models | python3 -c "import sys,json; print(', '.join(m['id'] for m in json.load(sys.stdin)['data']))")
    ok "LMStudio reachable — models: $MODELS"
else
    warn "LMStudio not reachable at localhost:1234 — LMStudio runs will be skipped"
fi

rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# ── Discover local models ───────────────────────────────────────────────────
MODEL_LIST=$(curl -s http://localhost:1234/v1/models | python3 -c "
import sys, json
data = json.load(sys.stdin)
llms = [m['id'] for m in data.get('data', []) if 'embed' not in m.get('id','').lower()]
for m in llms:
    print(m)
")

if [ -z "$MODEL_LIST" ]; then
    warn "No LLM models found in LMStudio — LMStudio runs will be skipped"
fi

ok "LMStudio models: $(echo "$MODEL_LIST" | tr '\n' ' ')"

# ── Run configuration ───────────────────────────────────────────────────────
# Build run list: deepseek + cloud models + each local model
RUNS=(
    "deepseek|deepseek-v4-pro|deepseek-v4-pro"
    "cloud|qwen3-coder-30b-a3b-instruct|qwen3-coder-30b"
    "cloud|devstral-2-123b-instruct-2512|devstral-2-123b"
    "cloud|deepseek-r1-distill-llama-70b|deepseek-r1-70b"
)
while IFS= read -r m; do
    [ -z "$m" ] && continue
    RUNS+=("lmstudio|${m}|${m//\//_}")
done <<< "$MODEL_LIST"

# Challenge directories.  Tier filtering is done via --tiers 1 (reads the
# "tier" field inside each challenge JSON), so we can use a catch-all pattern.
CHALLENGE_SETS=(
    "data/tasks/official/challenges/json|tt-official-ch*.json|official"
    "data/tasks/challenges_1comp|tt-official-ch*.json|1comp"
    "data/tasks/challenges_2comp|tt-official-ch*.json|2comp"
)

ARGS_COMMON="--task-type agentic_synthesis --save-report --max-turns 25 --tiers 1"

# ── Run benchmark ────────────────────────────────────────────────────────────
ALL_REPORTS=()

for run in "${RUNS[@]}"; do
    IFS='|' read -r provider model label <<< "$run"

    banner "Provider: $provider  |  Model: $label"

    for cset in "${CHALLENGE_SETS[@]}"; do
        IFS='|' read -r challenges_dir pattern cset_label <<< "$cset"

        OUT_DIR="$RESULTS_DIR/${label}/${cset_label}"
        mkdir -p "$OUT_DIR"

        echo -e "  Running: ${cset_label} (pattern: $pattern)"

        set +e
        uv run tt-benchmark \
            --provider "$provider" \
            --model "$model" \
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

        # Collect report files
        for f in "$OUT_DIR"/benchmark_*.json; do
            if [ -f "$f" ]; then
                ALL_REPORTS+=("$f")
            fi
        done
    done
done

# ── Aggregate results ────────────────────────────────────────────────────────
banner "Aggregating results"

python3 << 'PYEOF'
import json, sys
from pathlib import Path
from collections import defaultdict

results_dir = Path("benchmark_results/tier1_comparison")
reports = sorted(results_dir.rglob("benchmark_*.json"))

if not reports:
    print("No reports found!")
    sys.exit(0)

summary = {"experiment": "tier1_agentic_comparison", "models": {}}

for rp in reports:
    with open(rp) as f:
        data = json.load(f)

    # Derive model label from path: .../tier1_comparison/<label>/<challenge_set>/benchmark_*.json
    parts = rp.relative_to(results_dir).parts
    model_label = parts[0]       # e.g. deepseek-v4-pro
    challenge_set = parts[1]     # e.g. official

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

    # Also collect per-task details for richer analysis
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

# Compute success rates
for model_label, mdata in summary["models"].items():
    for cset, csdata in mdata["by_set"].items():
        csdata["success_rate"] = round(csdata["successful"] / csdata["total"] * 100, 1) if csdata["total"] else 0.0
    o = mdata["overall"]
    o["success_rate"] = round(o["successful"] / o["total"] * 100, 1) if o["total"] else 0.0

with open(results_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ── Print results table ─────────────────────────────────────────────────────
print()
print("=" * 80)
print("  TIER 1 AGENTIC SYNTHESIS — RESULTS")
print("=" * 80)
print()

# Header
header = f"{'Model':<22} {'Set':<10} {'Success':>8} {'Total':>6} {'Rate':>8}"
print(header)
print("-" * len(header))

for model_label, mdata in sorted(summary["models"].items()):
    for cset in ["official", "1comp", "2comp"]:
        csdata = mdata["by_set"].get(cset)
        if csdata:
            print(f"{model_label:<22} {cset:<10} {csdata['successful']:>6} / {csdata['total']:>4}  {csdata['success_rate']:>6.1f}%")
    o = mdata["overall"]
    print(f"{'':22} {'OVERALL':<10} {o['successful']:>6} / {o['total']:>4}  {o['success_rate']:>6.1f}%")
    print()

print("=" * 80)

# ── Per-task matrix ─────────────────────────────────────────────────────────
print()
print("PER-TASK MATRIX (✔ = solved, ✘ = failed)")
print("-" * 55)

# Collect all task IDs
all_task_ids = sorted(set(
    t["task_id"] for mdata in summary["models"].values() for t in mdata.get("tasks", [])
))

# Build lookup: model -> task_id -> success
matrix = defaultdict(lambda: defaultdict(lambda: "—"))
for model_label, mdata in summary["models"].items():
    for t in mdata.get("tasks", []):
        matrix[model_label][t["task_id"]] = "✔" if t["success"] else "✘"

# Print
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
