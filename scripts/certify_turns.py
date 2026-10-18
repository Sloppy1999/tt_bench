#!/usr/bin/env python3
"""Certify the stored `turns` field before trusting a turn-budget sweep.

Plan step P1 (HARNESS_ABLATION_PLAN.md section 7). docs/results_analysis.md
section 9.2 records that `turns` was once ``len(tool_calls)``, which is not a
turn count: a model emitting several tool calls in one assistant message
inflates it past the budget, and DeepSeek logged 213, 225, 276 and 444 against a
limit of 25. It is now ``len(turn_logprobs)`` — one entry per API call — with a
`turns_source` field saying which definition produced it.

That field is the entire mechanism story of the turn-budget arm: "budget
exhaustion" means `turns == budget`, and the plan's per-model predictions are
derived from the resulting ceiling fractions. So it has to be certified per model
BEFORE the sweep runs, not explained afterwards.

The independent estimate comes from the transcript, which stores a `turn` index
per tool call:

    derived_min = max(entry["turn"]) + 1

This is a LOWER BOUND on the number of API calls: a final turn that emitted
content rather than a tool call leaves no transcript entry. Where the task also
recorded a non-empty `llm_response`, that final turn happened, and derived_max
adds it. A recorded value inside [derived_min, derived_max] is consistent.

Usage:
    python scripts/certify_turns.py benchmark_results/jureca_tier1
    python scripts/certify_turns.py benchmark_results/jureca_tier1 --set scaled_1comp
    python scripts/certify_turns.py benchmark_results/jureca_tier1 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "jureca"))
from inspect_results import parse_set_label  # noqa: E402


def load_tasks(set_dir: Path) -> list[dict] | None:
    """Agentic task records for one run directory, or None if there are none.

    Prefers per_task/, which survives a Slurm time-out: docs/results_analysis.md
    section 8 records two jobs that produced a complete-looking aggregate and one
    that produced none at all, and the aggregate is only written after the last
    task finishes.
    """
    per_task = set_dir / "per_task"
    if per_task.is_dir():
        tasks = []
        for f in sorted(per_task.glob("*.json")):
            try:
                rec = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            if rec.get("task_type") == "agentic_synthesis":
                tasks.append(rec)
        if tasks:
            return tasks

    reports = sorted(set_dir.glob("benchmark_*.json"))
    if not reports:
        return None
    report = json.loads(reports[-1].read_text())
    tasks = [
        t for t in (report.get("results") or [])
        if t.get("task_type") == "agentic_synthesis"
    ]
    return tasks or None


def derive(task: dict) -> tuple[int, int]:
    """(derived_min, derived_max) API calls for one agentic task."""
    transcript = (task.get("predicted") or {}).get("transcript") or []
    turns = [e.get("turn") for e in transcript if isinstance(e.get("turn"), int)]
    derived_min = (max(turns) + 1) if turns else 0
    ended_with_content = bool((task.get("llm_response") or "").strip())
    return derived_min, derived_min + (1 if ended_with_content else 0)


def audit(tasks: list[dict], set_label: str) -> dict:
    _base, budget, _temp, _arm, _sample = parse_set_label(set_label)

    total = consistent = missing = 0
    sources: dict[str, int] = {}
    over_budget = 0
    ceil_recorded = ceil_derived = 0
    failures = 0
    worst: list[tuple[int, int, int, str]] = []

    for task in tasks:
        total += 1
        met = task.get("metrics") or {}
        recorded = met.get("turns")
        sources[met.get("turns_source") or "unrecorded"] = (
            sources.get(met.get("turns_source") or "unrecorded", 0) + 1
        )
        if not isinstance(recorded, int):
            missing += 1
            continue

        low, high = derive(task)
        if low <= recorded <= high:
            consistent += 1
        else:
            worst.append((recorded, low, high, task.get("task_id", "?")))
        if recorded > budget:
            over_budget += 1

        if not task.get("success"):
            failures += 1
            ceil_recorded += recorded >= budget
            ceil_derived += high >= budget

    worst.sort(key=lambda row: -abs(row[0] - row[2]))
    return {
        "set": set_label,
        "budget": budget,
        "n": total,
        "consistent": consistent,
        "missing": missing,
        "inconsistent": total - consistent - missing,
        "turns_source": dict(sorted(sources.items())),
        "over_budget": over_budget,
        "failures": failures,
        "ceil_recorded": ceil_recorded,
        "ceil_derived": ceil_derived,
        "examples": worst[:3],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--model", help="only this model directory")
    ap.add_argument("--set", dest="set_label", help="only set labels with this base name")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.results_dir.is_dir():
        sys.exit(f"No such results directory: {args.results_dir}")

    rows: dict[str, list[dict]] = {}
    for model_dir in sorted(p for p in args.results_dir.iterdir() if p.is_dir()):
        if args.model and model_dir.name != args.model:
            continue
        for set_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if args.set_label and parse_set_label(set_dir.name)[0] != args.set_label:
                continue
            tasks = load_tasks(set_dir)
            if tasks is None:
                continue
            rows.setdefault(model_dir.name, []).append(audit(tasks, set_dir.name))

    if not rows:
        sys.exit(f"No reports found under {args.results_dir}")

    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        print()
        return 0

    head = (
        f"{'model':<24} {'set':<22} {'n':>5} {'consist':>8} {'incons':>7} "
        f"{'>budget':>8} {'ceil rec':>9} {'ceil der':>9}"
    )
    print(head)
    print("-" * len(head))
    suspect = []
    for model, entries in rows.items():
        for i, r in enumerate(entries):
            pct = 100.0 * r["consistent"] / r["n"] if r["n"] else 0.0
            flag = "" if pct >= 99.0 else "  <-- SUSPECT"
            if flag:
                suspect.append((model, r))
            fails = r["failures"] or 1
            print(
                f"{model if i == 0 else '':<24} {r['set']:<22} {r['n']:>5} "
                f"{pct:>7.1f}% {r['inconsistent']:>7} {r['over_budget']:>8} "
                f"{100.0 * r['ceil_recorded'] / fails:>8.1f}% "
                f"{100.0 * r['ceil_derived'] / fails:>8.1f}%{flag}"
            )
        print()

    print("'consist' is the share of tasks whose recorded turns fall inside the")
    print("range derived from the transcript. 'ceil rec' and 'ceil der' are the")
    print("share of FAILURES on the turn ceiling under each — the statistic the")
    print("plan's per-model predictions are built on. Where they disagree, the")
    print("prediction table must be rebuilt from 'ceil der' before the sweep runs.")
    print("'>budget' counts tasks recording more turns than the budget allows;")
    print("any nonzero value is the len(tool_calls) bug of results_analysis 9.2.")

    if suspect:
        print("\nSuspect cells:")
        for model, r in suspect:
            print(f"  {model} / {r['set']}: sources={r['turns_source']}")
            for recorded, low, high, task_id in r["examples"]:
                print(f"    {task_id}: recorded {recorded}, transcript implies {low}-{high}")
        print("\nDo not quote ceiling fractions for these cells from the stored")
        print("field. Either re-run them or use the derived column.")
        return 1
    print("\nAll cells consistent — the stored turn counts support the sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
