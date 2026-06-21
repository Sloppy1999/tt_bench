#!/usr/bin/env bash
# =============================================================================
# Tier 1 Agentic Synthesis — Local Models Only (LMStudio)
# =============================================================================
# Runs gemma-4-e4b and qwen3.6-27b via LMStudio across official, 1comp,
# and 2comp challenge sets (tier 1 only, filtered via --tiers 1).
#
# Usage:
#   chmod +x scripts/run_local_benchmark.sh
#   ./scripts/run_local_benchmark.sh
#
# Prerequisites:
#   - LMStudio running at http://localhost:1234 with both models loaded
#   - uv installed
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="benchmark_results/local_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

banner() { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "  ${RED}✘${NC} $1"; }

# ── Pre-flight ──────────────────────────────────────────────────────────────
banner "Pre-flight"

if curl -s --max-time 3 http://localhost:1234/v1/models > /dev/null 2>&1; then
    MODELS=$(curl -s http://localhost:1234/v1/models | python3 -c "import sys,json; print(', '.join(m['id'] for m in json.load(sys.stdin)['data']))")
    ok "LMStudio reachable — models: $MODELS"
else
    fail "LMStudio not reachable at localhost:1234"
    exit 1
fi

rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# ── Discover local models ───────────────────────────────────────────────────
# Query LMStudio /v1/models and filter out embedding models.
MODEL_LIST=$(curl -s http://localhost:1234/v1/models | python3 -c "
import sys, json
data = json.load(sys.stdin)
llms = [m['id'] for m in data.get('data', []) if 'embed' not in m.get('id','').lower()]
for m in llms:
    print(m)
")

if [ -z "$MODEL_LIST" ]; then
    fail "No LLM models found in LMStudio"
    exit 1
fi

ok "Will benchmark: $(echo "$MODEL_LIST" | tr '\n' ' ')"

# ── Run config ──────────────────────────────────────────────────────────────
IFS=$'\n' read -d '' -ra RUNS <<< "$MODEL_LIST" || true

CHALLENGE_SETS=(
    "data/tasks/official/challenges/json|official"
    "data/tasks/challenges_1comp|1comp"
    "data/tasks/challenges_2comp|2comp"
)

ARGS_COMMON="--provider lmstudio --task-type agentic_synthesis --save-report --max-turns 25 --tiers 1"

# ── Run ─────────────────────────────────────────────────────────────────────
ALL_REPORTS=()

while IFS= read -r model; do
    [ -z "$model" ] && continue
    banner "Model: $model"

    for cset in "${CHALLENGE_SETS[@]}"; do
        IFS='|' read -r challenges_dir cset_label <<< "$cset"

        OUT_DIR="$RESULTS_DIR/${model//\//_}/${cset_label}"
        mkdir -p "$OUT_DIR"

        echo -e "  Running: ${cset_label}"

        set +e
        uv run tt-benchmark \
            --model "$model" \
            --challenges-dir "$challenges_dir" \
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
done <<< "$MODEL_LIST"

# ── Summary ─────────────────────────────────────────────────────────────────
banner "Results"

python3 << 'PYEOF'
import json, sys
from pathlib import Path
from collections import defaultdict

results_dir = Path("benchmark_results/local_comparison")
reports = sorted(results_dir.rglob("benchmark_*.json"))

if not reports:
    print("No reports found!")
    sys.exit(0)

models = defaultdict(lambda: {"overall": {"total": 0, "successful": 0, "failed": 0}, "by_set": {}, "tasks": []})

for rp in reports:
    with open(rp) as f:
        data = json.load(f)
    parts = rp.relative_to(results_dir).parts
    model_label = parts[0]
    cset = parts[1]

    m = models[model_label]
    m["by_set"][cset] = {
        "total": data.get("total_tasks", 0),
        "successful": data.get("successful", 0),
        "failed": data.get("failed", 0),
    }
    m["overall"]["total"] += data.get("total_tasks", 0)
    m["overall"]["successful"] += data.get("successful", 0)
    m["overall"]["failed"] += data.get("failed", 0)
    for task in data.get("results", []):
        m["tasks"].append({
            "task_id": task.get("task_id"),
            "challenge_set": cset,
            "success": task.get("success"),
            "tool_calls": len(task.get("predicted", {}).get("tool_calls", [])),
        })

for ml, mdata in models.items():
    for cs, csd in mdata["by_set"].items():
        csd["rate"] = round(csd["successful"] / csd["total"] * 100, 1) if csd["total"] else 0.0
    o = mdata["overall"]
    o["rate"] = round(o["successful"] / o["total"] * 100, 1) if o["total"] else 0.0

# Print table
print()
print("=" * 70)
print("  TIER 1 AGENTIC SYNTHESIS — LOCAL MODELS")
print("=" * 70)
print()
print(f"{'Model':<24} {'Set':<10} {'Solved':>8} {'Rate':>8}")
print("-" * 52)

for ml in sorted(models):
    mdata = models[ml]
    for cset in ["official", "1comp", "2comp"]:
        csd = mdata["by_set"].get(cset)
        if csd:
            print(f"{ml:<24} {cset:<10} {csd['successful']:>3} / {csd['total']:<3}  {csd['rate']:>6.1f}%")
    o = mdata["overall"]
    print(f"{'':24} {'OVERALL':<10} {o['successful']:>3} / {o['total']:<3}  {o['rate']:>6.1f}%")
    print()

# Per-task matrix
print("=" * 70)
print("PER-TASK MATRIX")
print()

all_tasks = sorted(set(t["task_id"] for m in models.values() for t in m["tasks"]))
col_w = max(len(t) for t in all_tasks)
print(f"  {'Challenge':<{col_w}}  " + "  ".join(f"{ml:<24}" for ml in sorted(models)))
print("  " + "-" * (col_w + 2 + sum(26 for _ in models)))

matrix = defaultdict(lambda: defaultdict(str))
for ml, mdata in models.items():
    for t in mdata["tasks"]:
        matrix[t["task_id"]][ml] = "\u2713" if t["success"] else "\u2717"

for tid in all_tasks:
    vals = [matrix[tid].get(ml, "—") for ml in sorted(models)]
    print(f"  {tid:<{col_w}}  " + "  ".join(f"{v:<24}" for v in vals))

print()

# Save summary
with open(results_dir / "summary.json", "w") as f:
    json.dump({
        "experiment": "tier1_local_comparison",
        "models": {ml: {
            "overall": m["overall"],
            "by_set": m["by_set"],
        } for ml, m in models.items()}
    }, f, indent=2)

print(f"Summary saved to: {results_dir / 'summary.json'}")
PYEOF

banner "Done"
