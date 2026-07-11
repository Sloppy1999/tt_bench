#!/usr/bin/env bash
# =============================================================================
# Aggregate JURECA Benchmark Results
# =============================================================================
# Collects all benchmark_*.json files from jureca_tier1/*/ and produces a
# summary with per-model, per-challenge-set success rates.
#
# Usage:
#   bash jureca/aggregate_results.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULTS_DIR="$PROJECT_DIR/benchmark_results/jureca_tier1"
SUMMARY_FILE="$RESULTS_DIR/summary.json"
CSV_FILE="$RESULTS_DIR/summary.csv"

BOLD='\033[1m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

banner() { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC} $1"; }
fail()  { echo -e "  ${RED}✘${NC} $1"; }

banner "Aggregating results from $RESULTS_DIR"

# Ensure venv is available for the Python aggregation
VENV_DIR="$HOME/.venv-ttbench"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

cd "$PROJECT_DIR"

python3 << 'PYEOF'
import json
from pathlib import Path
from collections import defaultdict

results_dir = Path("benchmark_results/jureca_tier1")
reports = sorted(results_dir.rglob("benchmark_*.json"))

if not reports:
    print("No reports found in benchmark_results/jureca_tier1/")
    print("Wait for Slurm jobs to finish, then re-run this script.")
    exit(0)

print(f"Found {len(reports)} report files\n")

summary = {"experiment": "jureca_tier1_vllm", "models": {}}

for rp in reports:
    with open(rp) as f:
        data = json.load(f)

    # Extract model label from path: .../jureca_tier1/<model_label>/<set>/benchmark_*.json
    parts = rp.relative_to(results_dir).parts
    if len(parts) < 3:
        continue
    model_label = parts[0]       # e.g. qwen2.5-coder-7b
    challenge_set = parts[1]     # e.g. official

    if model_label not in summary["models"]:
        summary["models"][model_label] = {
            "provider": data.get("provider", ""),
            "model": data.get("model", ""),
            "by_set": {},
            "overall": {"total": 0, "successful": 0, "failed": 0},
            "tasks": [],
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

    # Per-task details
    for task in data.get("results", []):
        entry["tasks"].append({
            "task_id": task.get("task_id"),
            "challenge_set": challenge_set,
            "success": task.get("success"),
            "component_score": task.get("component_score"),
            "turns": len(task.get("predicted", {}).get("tool_calls", [])),
        })

# Compute success rates
for model_label, mdata in summary["models"].items():
    for cset, csdata in mdata["by_set"].items():
        csdata["success_rate"] = round(csdata["successful"] / csdata["total"] * 100, 1) if csdata["total"] else 0.0
    o = mdata["overall"]
    o["success_rate"] = round(o["successful"] / o["total"] * 100, 1) if o["total"] else 0.0

# ── Print results table ─────────────────────────────────────────────────────
print()
print("=" * 90)
print("  JURECA H100 — TIER 1 AGENTIC SYNTHESIS RESULTS")
print("=" * 90)
print()

header = f"  {'Model':<28} {'Set':<12} {'Solved':>8} {'Total':>6} {'Rate':>8}  {'Component':>10}"
print(header)
print("  " + "-" * len(header))

for model_label in sorted(summary["models"].keys()):
    mdata = summary["models"][model_label]
    for cset in ["official", "1comp", "2comp"]:
        csdata = mdata["by_set"].get(cset)
        if csdata:
            avg_comp = ""
            comp_tasks = [
                t["component_score"] for t in mdata["tasks"]
                if t["challenge_set"] == cset and t["component_score"] is not None
            ]
            if comp_tasks:
                avg_comp = f"{sum(comp_tasks) / len(comp_tasks):.2f}"
            print(f"  {model_label:<28} {cset:<12} {csdata['successful']:>6} / {csdata['total']:>4}  {csdata['success_rate']:>6.1f}%  {avg_comp:>10}")
    o = mdata["overall"]
    print(f"  {'':28} {'OVERALL':<12} {o['successful']:>6} / {o['total']:>4}  {o['success_rate']:>6.1f}%")
    print()

# ── Per-task matrix ─────────────────────────────────────────────────────────
print("=" * 90)
print("  PER-TASK MATRIX (\u2713 = solved, \u2717 = failed)")
print()

all_task_ids = sorted(set(
    t["task_id"] for mdata in summary["models"].values() for t in mdata.get("tasks", [])
))

matrix = defaultdict(lambda: defaultdict(str))
for model_label, mdata in summary["models"].items():
    for t in mdata.get("tasks", []):
        matrix[model_label][t["task_id"]] = "\u2713" if t["success"] else "\u2717"

model_names = sorted(summary["models"].keys())
col_w = max(len(t) for t in all_task_ids) if all_task_ids else 15
row_fmt = f"  {{:<{col_w}}}  " + "  ".join(f"{{:<{len(m)+2}}}" for m in model_names)

print(row_fmt.format("Challenge", *model_names))
print("  " + "-" * (col_w + 2 + sum(len(m) + 4 for m in model_names)))

for tid in all_task_ids:
    vals = [matrix[m].get(tid, "—") for m in model_names]
    print(row_fmt.format(tid, *vals))

print()
print("=" * 90)

# ── Save summary JSON ───────────────────────────────────────────────────────
with open(results_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved: {results_dir / 'summary.json'}")

# ── Save CSV for spreadsheet analysis ───────────────────────────────────────
with open(results_dir / "summary.csv", "w") as f:
    f.write("model,challenge_set,total,successful,failed,success_rate\n")
    for model_label in sorted(summary["models"].keys()):
        mdata = summary["models"][model_label]
        for cset in ["official", "1comp", "2comp"]:
            csdata = mdata["by_set"].get(cset)
            if csdata:
                f.write(f"{model_label},{cset},{csdata['total']},{csdata['successful']},{csdata['failed']},{csdata['success_rate']}%\n")
        o = mdata["overall"]
        f.write(f"{model_label},OVERALL,{o['total']},{o['successful']},{o['failed']},{o['success_rate']}%\n")

print(f"CSV saved:    {results_dir / 'summary.csv'}")

# ── Per-model per-task CSV ───────────────────────────────────────────────────
with open(results_dir / "per_task.csv", "w") as f:
    f.write("model," + ",".join(all_task_ids) + "\n")
    for model_label in sorted(summary["models"].keys()):
        vals = [matrix[model_label].get(tid, "") for tid in all_task_ids]
        f.write(f"{model_label}," + ",".join(vals) + "\n")

print(f"Per-task CSV: {results_dir / 'per_task.csv'}")
PYEOF

echo ""
ok "Aggregation complete"
