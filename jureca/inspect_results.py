#!/usr/bin/env python3
"""Summarise TT-Bench benchmark reports into a comparison table.

Walks ``benchmark_results/<experiment>/<model>/<set>/benchmark_*.json`` and prints
one row per model/set with the success rate, the turn distribution split by
outcome, component accuracy, and the dominant failure family.

Two deliberate choices worth knowing about:

* **Success rate is recomputed** from the per-task ``success`` flags rather than
  read from the report's ``success_rate`` field. That field appears twice in a
  report with different units (0.8 at the top level, 80.0 under ``per_tier``),
  so anything reading it by name can be off by 100x depending on which one it
  finds first. Counting the tasks has no such ambiguity.

* **Turns are reported separately for successes and failures.** On the tier-1
  runs the two populations are bimodal — successes finish in 4-7 turns while
  failures sit exactly on the turn ceiling — and a single mean hides precisely
  the thing worth seeing.

Usage:
    python jureca/inspect_results.py
    python jureca/inspect_results.py --model gemma-4-31b-it --errors
    python jureca/inspect_results.py --json > summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# Turn budget encoded in a set directory name by run_benchmark.sbatch: a sweep
# writes "1comp_t50" so runs at different budgets never share a directory.
SET_BUDGET_RE = re.compile(r"_t(\d+)$")
DEFAULT_BUDGET = 25

# Board rows grouped into bands, shared with scripts/analyze_tier1_experiment.py
# so the per-model and cross-model figures cannot disagree. Two of the three
# consumers used to stop at y=9 while a third went to y>=10; the scaled sets are
# 13x13 and 15x15, so tasks with components below row 9 fell into no band.
#
# A task counts towards every band it has a ground-truth component in, so the
# bands deliberately do NOT sum to the task total — a two-component solution
# spanning Top and Deep appears in both.
ZONES = [
    ("Top (y=0-3)", 0, 3),
    ("Mid (y=4-6)", 4, 6),
    ("Bot (y=7-9)", 7, 9),
    ("Deep (y>=10)", 10, 99),
]

# Ordering for the common sets; anything unrecognised is appended alphabetically.
SET_ORDER = ["official", "1comp", "2comp", "scaled", "scaled_1comp", "scaled_2comp"]


def normalise_error(msg: str) -> str:
    """Collapse an error string into a comparable family.

    "Illegal free fall: marble 1 traversed empty cell (8, 8)" and the same
    message with (2, 6) are one failure mode, not two. Numbers and quoted
    inventory lists carry the instance detail, so they are replaced by
    placeholders to leave the mode itself.
    """
    if not msg:
        return ""
    fam = re.sub(r"\[[^\]]*\]", "[...]", msg)
    fam = re.sub(r"\(\s*\d+\s*,\s*\d+\s*\)", "(N, N)", fam)
    fam = re.sub(r"\d+", "N", fam)
    return fam.strip()


def set_sort_key(label: str) -> tuple[int, int, str]:
    base = SET_BUDGET_RE.sub("", label)
    budget = int(m.group(1)) if (m := SET_BUDGET_RE.search(label)) else DEFAULT_BUDGET
    rank = SET_ORDER.index(base) if base in SET_ORDER else len(SET_ORDER)
    return (rank, budget, label)


def latest_report(set_dir: Path) -> Path | None:
    """Newest report in a set directory.

    Re-running the same set into the same directory leaves several timestamped
    reports; sorting by name works because the filenames are ISO timestamps.
    """
    reports = sorted(set_dir.glob("benchmark_*.json"))
    return reports[-1] if reports else None


def summarise(report_path: Path, set_label: str) -> dict:
    with report_path.open() as fh:
        report = json.load(fh)

    results = report.get("results") or []
    budget = int(m.group(1)) if (m := SET_BUDGET_RE.search(set_label)) else DEFAULT_BUDGET

    ok_turns: list[int] = []
    fail_turns: list[int] = []
    comp_correct = comp_placed = comp_gt = 0
    error_families: dict[str, int] = {}
    turn_errors: dict[str, int] = {}
    turn_error_tasks: dict[str, int] = {}
    tokens: list[int] = []
    latencies: list[int] = []
    zone_ok = {z[0]: 0 for z in ZONES}
    zone_total = {z[0]: 0 for z in ZONES}

    tool_calls_total = 0

    for task in results:
        met = task.get("metrics") or {}
        tool_calls_total += met.get("tool_calls_count") or 0
        turns = met.get("turns")
        if isinstance(turns, int):
            (ok_turns if task.get("success") else fail_turns).append(turns)

        comp_correct += met.get("component_correct") or 0
        comp_placed += met.get("component_placed") or 0
        comp_gt += met.get("component_gt") or 0

        # The top-level `error` is the OUTCOME: why the task was scored a failure.
        if fam := normalise_error(task.get("error") or ""):
            error_families[fam] = error_families.get(fam, 0) + 1

        # Per-tool-call errors are a different thing entirely: rejected actions
        # the agent recovered from. Inventory violations live here, not in the
        # outcome — counting raw error strings across a whole report conflates
        # the two and overstates them as a terminal cause. Tracked per task as
        # well as per occurrence, because "300 violations" spread over 5 tasks
        # and over 300 tasks are very different claims.
        seen_here: set[str] = set()
        for step in (task.get("predicted") or {}).get("transcript") or []:
            if sfam := normalise_error(step.get("error") or ""):
                turn_errors[sfam] = turn_errors.get(sfam, 0) + 1
                seen_here.add(sfam)
        for sfam in seen_here:
            turn_error_tasks[sfam] = turn_error_tasks.get(sfam, 0) + 1

        # Which board bands this task's ground-truth solution touches.
        ys = [c.get("y") for c in
              ((task.get("expected") or {}).get("solution") or {}).get("placed_components") or []]
        ys = [y for y in ys if isinstance(y, int)]
        for name, lo, hi in ZONES:
            if any(lo <= y <= hi for y in ys):
                zone_total[name] += 1
                if task.get("success"):
                    zone_ok[name] += 1

        if isinstance(task.get("tokens_used"), int):
            tokens.append(task["tokens_used"])
        if isinstance(task.get("latency_ms"), int):
            latencies.append(task["latency_ms"])

    total = len(results)
    successful = sum(1 for t in results if t.get("success") is True)

    return {
        "set": set_label,
        "budget": budget,
        "report": str(report_path),
        "timestamp": report.get("timestamp"),
        "model_id": report.get("model"),
        "provider": report.get("provider"),
        "total": total,
        "successful": successful,
        # Recomputed, not read from the report — see the module docstring.
        "success_rate_pct": (100.0 * successful / total) if total else 0.0,
        "turns_ok": sorted(ok_turns),
        "turns_fail": sorted(fail_turns),
        # A failure sitting exactly on the budget did not converge; it ran out.
        "fail_at_ceiling": sum(1 for t in fail_turns if t >= budget),
        "component_correct": comp_correct,
        "component_placed": comp_placed,
        "component_gt": comp_gt,
        "error_families": dict(sorted(error_families.items(), key=lambda kv: -kv[1])),
        # Rejected actions during the episode, not the reason the task failed.
        "turn_errors": dict(sorted(turn_errors.items(), key=lambda kv: -kv[1])),
        "turn_error_tasks": dict(sorted(turn_error_tasks.items(), key=lambda kv: -kv[1])),
        "zone_ok": zone_ok,
        "zone_total": zone_total,
        "tool_calls_total": tool_calls_total,
        "tokens_total": sum(tokens),
        "latency_ms_median": int(statistics.median(latencies)) if latencies else None,
    }


def discover(root: Path, model_filter: str | None, set_filter: str | None) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = {}
    if not root.is_dir():
        sys.exit(f"No such results directory: {root}")

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if model_filter and model_dir.name != model_filter:
            continue
        rows = []
        for set_dir in sorted((p for p in model_dir.iterdir() if p.is_dir()), key=lambda p: set_sort_key(p.name)):
            if set_filter and SET_BUDGET_RE.sub("", set_dir.name) != set_filter:
                continue
            if (report := latest_report(set_dir)) is None:
                continue
            rows.append(summarise(report, set_dir.name))
        if rows:
            by_model[model_dir.name] = rows
    return by_model


def fmt_turns(turns: list[int]) -> str:
    if not turns:
        return "-"
    if len(turns) == 1:
        return str(turns[0])
    return f"{turns[0]}/{int(statistics.median(turns))}/{turns[-1]}"


def render(by_model: dict[str, list[dict]], show_errors: bool, show_turn_errors: bool = False) -> None:
    if not by_model:
        print("No reports found. Has a job finished writing to benchmark_results/?")
        return

    head = (
        f"{'model':<24} {'set':<14} {'budget':>6} {'n':>5} {'ok':>4} {'rate':>7} "
        f"{'turns ok':>12} {'turns fail':>12} {'ceil':>5} {'comp':>10} "
        f"{'tok/task':>9} {'lat_med':>8}"
    )
    print(head)
    print("-" * len(head))

    for model, rows in by_model.items():
        for i, r in enumerate(rows):
            comp = f"{r['component_correct']}/{r['component_gt']}"
            # Throughput, because nobody was watching it: qwen2.5-coder-7b spent
            # 11.5h on a set two other models finished in ~2h, and the only signal
            # was the eventual TIMEOUT. Tokens per task and median latency make
            # that visible in the first minutes of a run instead of the 12th hour.
            tok = r["tokens_total"] // r["total"] if r["total"] else 0
            lat = r["latency_ms_median"]
            print(
                f"{model if i == 0 else '':<24} "
                f"{SET_BUDGET_RE.sub('', r['set']):<14} "
                f"{r['budget']:>6} "
                f"{r['total']:>5} {r['successful']:>4} "
                f"{r['success_rate_pct']:>6.1f}% "
                f"{fmt_turns(r['turns_ok']):>12} "
                f"{fmt_turns(r['turns_fail']):>12} "
                f"{r['fail_at_ceiling']:>5} "
                f"{comp:>10} "
                f"{tok:>9,} "
                f"{(f'{lat / 1000:.1f}s' if lat else '-'):>8}"
            )
        print()

    print("turns columns are min/median/max. 'ceil' counts failures sitting on the")
    print("turn budget — those ran out of turns rather than converging on a wrong answer.")
    print("'comp' is components placed correctly / required by ground truth.")
    print("'tok/task' is mean tokens per task and 'lat_med' the median task latency —")
    print("watch these early: a model generating 5x the tokens of another will not")
    print("finish a 1000-task set in the same walltime, and TIMEOUT is a late signal.")

    if show_errors:
        for model, rows in by_model.items():
            for r in rows:
                if not r["error_families"]:
                    continue
                print(f"\n{model} / {r['set']} — terminal failure families:")
                for fam, count in r["error_families"].items():
                    print(f"  {count:>3}x  {fam}")

    if show_turn_errors:
        print("\n" + "=" * 78)
        print("REJECTED ACTIONS DURING EPISODES — not the reason any task failed.")
        print("'tasks' is how many tasks hit the family at least once, out of n.")
        print("=" * 78)
        for model, rows in by_model.items():
            for r in rows:
                if not r["turn_errors"]:
                    continue
                print(f"\n{model} / {r['set']}  (n={r['total']}):")
                for fam, count in r["turn_errors"].items():
                    tasks = r["turn_error_tasks"].get(fam, 0)
                    pct = 100.0 * tasks / r["total"] if r["total"] else 0
                    print(f"  {count:>5}x over {tasks:>4} task(s) ({pct:4.1f}%)  {fam}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--results-dir",
        default="benchmark_results/jureca_tier1",
        help="experiment directory holding <model>/<set>/ (default: %(default)s)",
    )
    ap.add_argument("--model", help="only this model directory")
    ap.add_argument("--set", dest="set_label", help="only this set label, ignoring any _tN suffix")
    ap.add_argument("--errors", action="store_true", help="also list terminal failure families per model/set")
    ap.add_argument(
        "--turn-errors",
        action="store_true",
        help="list actions the executor rejected mid-episode (inventory violations, "
        "out-of-bounds placements). These are recovered-from events, NOT the reason "
        "a task was scored a failure — reporting them as failure causes overstates them.",
    )
    ap.add_argument("--json", action="store_true", help="emit the summary as JSON instead of a table")
    args = ap.parse_args()

    by_model = discover(Path(args.results_dir), args.model, args.set_label)

    if args.json:
        json.dump(by_model, sys.stdout, indent=2)
        print()
    else:
        render(by_model, args.errors, args.turn_errors)


if __name__ == "__main__":
    main()
