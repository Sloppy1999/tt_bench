#!/usr/bin/env bash
# =============================================================================
# Aggregate JURECA Benchmark Results
# =============================================================================
# Collects results from jureca_tier1/<model>/<set>/ and produces a per-model,
# per-challenge-set summary. For each set it prefers the aggregate
# benchmark_*.json report, and falls back to reconstructing from the
# per_task/*.json files (written incrementally) when a set has no aggregate
# report yet — e.g. a job that timed out mid-run. Covers official/1comp/2comp
# and the large scaled set.
#
# Usage:
#   bash jureca/aggregate_results.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-detect project directory (consistent with submit_all.sh)
if [ -n "${PROJECT_DIR:-}" ]; then
    :  # already set
elif [ -d "/p/scratch/westai0070/$USER/tt-bench" ]; then
    PROJECT_DIR="/p/scratch/westai0070/$USER/tt-bench"
elif [ -d "$SCRIPT_DIR/.." ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

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

# Display order; any other discovered sets are appended after these.
SET_ORDER = ["official", "1comp", "2comp", "scaled"]
# scaled has ~1000 tasks — counted in summaries/CSV but kept out of the printed
# per-task matrix (which would otherwise be unreadable).
MATRIX_SETS = {"official", "1comp", "2comp"}


def load_set_results(set_dir):
    """Return (results_list, provider, model) for one <model>/<set> dir.

    Prefer the newest aggregate benchmark_*.json; if none exists (e.g. the job
    timed out mid-set), reconstruct from the per_task/*.json files written
    incrementally. Returns (None, "", "") if the set has no data at all.
    """
    reports = sorted(set_dir.glob("benchmark_*.json"))
    if reports:
        with open(reports[-1]) as f:            # newest by timestamped name
            data = json.load(f)
        return data.get("results", []), data.get("provider", ""), data.get("model", "")
    per_task = set_dir / "per_task"
    if per_task.is_dir():
        results = []
        for pf in sorted(per_task.glob("*.json")):
            try:
                with open(pf) as f:
                    results.append(json.load(f))
            except Exception:
                pass
        if results:
            return results, "", "(reconstructed from per_task/)"
    return None, "", ""


if not results_dir.is_dir():
    print(f"No results directory yet: {results_dir}")
    print("Wait for Slurm jobs to finish, then re-run this script.")
    raise SystemExit(0)

summary = {"experiment": "jureca_tier1_vllm", "models": {}}
found = 0

for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
    model_label = model_dir.name
    for set_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        challenge_set = set_dir.name
        results, provider, model = load_set_results(set_dir)
        if not results:
            continue
        found += 1
        has_report = bool(list(set_dir.glob("benchmark_*.json")))

        if model_label not in summary["models"]:
            summary["models"][model_label] = {
                "provider": provider, "model": model,
                "by_set": {}, "overall": {"total": 0, "successful": 0, "failed": 0},
                "tasks": [],
            }
        entry = summary["models"][model_label]
        if provider and not entry["provider"]:
            entry["provider"] = provider
        if model and not entry["model"]:
            entry["model"] = model

        total = len(results)
        successful = sum(1 for t in results if t.get("success") is True)
        entry["by_set"][challenge_set] = {
            "total": total, "successful": successful, "failed": total - successful,
            "source": "report" if has_report else "per_task",
        }
        entry["overall"]["total"] += total
        entry["overall"]["successful"] += successful
        entry["overall"]["failed"] += total - successful

        for task in results:
            entry["tasks"].append({
                "task_id": task.get("task_id"),
                "challenge_set": challenge_set,
                "success": task.get("success"),
                "component_score": task.get("component_score"),
                "tool_calls": task.get("metrics", {}).get("tool_calls_count", 0),
            })

if not found:
    print("No reports found in benchmark_results/jureca_tier1/")
    print("Wait for Slurm jobs to finish, then re-run this script.")
    raise SystemExit(0)

print(f"Found data for {found} model/set combination(s)\n")


def ordered_sets(sets):
    return [s for s in SET_ORDER if s in sets] + sorted(s for s in sets if s not in SET_ORDER)


for mdata in summary["models"].values():
    for csdata in mdata["by_set"].values():
        csdata["success_rate"] = round(csdata["successful"] / csdata["total"] * 100, 1) if csdata["total"] else 0.0
    o = mdata["overall"]
    o["success_rate"] = round(o["successful"] / o["total"] * 100, 1) if o["total"] else 0.0

# ── Results table ────────────────────────────────────────────────────────────
print("=" * 90)
print("  JURECA H100 — TIER 1 AGENTIC SYNTHESIS RESULTS")
print("=" * 90 + "\n")
header = f"  {'Model':<28} {'Set':<12} {'Solved':>8} {'Total':>6} {'Rate':>8}  {'Component':>10}"
print(header)
print("  " + "-" * len(header))

for model_label in sorted(summary["models"]):
    mdata = summary["models"][model_label]
    for cset in ordered_sets(mdata["by_set"]):
        csdata = mdata["by_set"][cset]
        comp = [t["component_score"] for t in mdata["tasks"]
                if t["challenge_set"] == cset and t["component_score"] is not None]
        avg_comp = f"{sum(comp) / len(comp):.2f}" if comp else ""
        tag = "" if csdata["source"] == "report" else " *partial"
        print(f"  {model_label:<28} {cset:<12} {csdata['successful']:>6} / {csdata['total']:>4}  "
              f"{csdata['success_rate']:>6.1f}%  {avg_comp:>10}{tag}")
    o = mdata["overall"]
    print(f"  {'':28} {'OVERALL':<12} {o['successful']:>6} / {o['total']:>4}  {o['success_rate']:>6.1f}%\n")

print("  * partial = reconstructed from per_task/ (set has no aggregate report yet)\n")

# ── Per-task matrix (small sets only) ────────────────────────────────────────
print("=" * 90)
print("  PER-TASK MATRIX (solved=1 / failed=0) — official/1comp/2comp only\n")
all_task_ids = sorted(set(
    t["task_id"] for mdata in summary["models"].values()
    for t in mdata["tasks"] if t["challenge_set"] in MATRIX_SETS
))
matrix = defaultdict(dict)
for model_label, mdata in summary["models"].items():
    for t in mdata["tasks"]:
        if t["challenge_set"] in MATRIX_SETS:
            matrix[model_label][t["task_id"]] = "1" if t["success"] else "0"

model_names = sorted(summary["models"])
if all_task_ids and model_names:
    col_w = max(len(t) for t in all_task_ids)
    row_fmt = f"  {{:<{col_w}}}  " + "  ".join(f"{{:<{len(m)+2}}}" for m in model_names)
    print(row_fmt.format("Challenge", *model_names))
    print("  " + "-" * (col_w + 2 + sum(len(m) + 4 for m in model_names)))
    for tid in all_task_ids:
        print(row_fmt.format(tid, *[matrix[m].get(tid, "-") for m in model_names]))
print("\n" + "=" * 90)

# ── Save summary JSON ────────────────────────────────────────────────────────
with open(results_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved: {results_dir / 'summary.json'}")

# ── summary.csv (all sets incl. scaled) ──────────────────────────────────────
with open(results_dir / "summary.csv", "w") as f:
    f.write("model,challenge_set,total,successful,failed,success_rate,source\n")
    for model_label in sorted(summary["models"]):
        mdata = summary["models"][model_label]
        for cset in ordered_sets(mdata["by_set"]):
            c = mdata["by_set"][cset]
            f.write(f"{model_label},{cset},{c['total']},{c['successful']},{c['failed']},{c['success_rate']}%,{c['source']}\n")
        o = mdata["overall"]
        f.write(f"{model_label},OVERALL,{o['total']},{o['successful']},{o['failed']},{o['success_rate']}%,\n")
print(f"CSV saved:    {results_dir / 'summary.csv'}")

# ── per_task.csv (long format — scales to the ~1000-task scaled set) ──────────
with open(results_dir / "per_task.csv", "w") as f:
    f.write("model,challenge_set,task_id,success,component_score,tool_calls\n")
    for model_label in sorted(summary["models"]):
        for t in summary["models"][model_label]["tasks"]:
            f.write(f"{model_label},{t['challenge_set']},{t['task_id']},"
                    f"{t['success']},{t['component_score']},{t['tool_calls']}\n")
print(f"Per-task CSV: {results_dir / 'per_task.csv'}  (long format)")
PYEOF

echo ""
ok "Aggregation complete"
