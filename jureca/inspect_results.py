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

# run_benchmark.sbatch encodes the run's configuration in the set directory name
# so that varying anything cannot overwrite the baseline:
#   1comp_t50              turn budget 50
#   scaled_T0.7            temperature 0.7
#   scaled_2comp_T0.7_s3   temperature 0.7, repetition 3
# Suffixes are stripped right-to-left in the order they are appended.
SET_SAMPLE_RE = re.compile(r"_s(\d+)$")
SET_TEMP_RE = re.compile(r"_T([0-9.]+)$")
SET_BUDGET_RE = re.compile(r"_t(\d+)$")
DEFAULT_BUDGET = 25
DEFAULT_TEMPERATURE = "0.0"

# The directory name is a WEAK source for the temperature: an unsuffixed
# directory means greedy only for runs made after the harness could set the
# temperature at all. Before 7920908/745cbce (both 2026-07-28) neither the
# runner nor the sbatch had a --temperature flag, so LLMConfig's default of 0.7
# applied and the run still landed in an unsuffixed directory. Reports written
# since then carry a "sampling" block and are read from it directly; older ones
# are attributed to 0.7 and flagged, because a silent 0.0 there would relabel
# every pre-cutover measurement as greedy.
# First day on which an unsuffixed directory reliably means greedy. Both cutover
# commits landed DURING 2026-07-28, so runs from that day sit on either side of
# them and are attributed to 0.7 with a '?' rather than silently resolved.
GREEDY_DEFAULT_SINCE = "2026-07-29"
LEGACY_TEMPERATURE = "0.7"


def resolve_temperature(report: dict, set_label: str) -> tuple[str, str]:
    """(temperature, how it was determined) for one loaded report."""
    if m := SET_TEMP_RE.search(set_label):
        return m.group(1), "dirname"
    recorded = (report.get("sampling") or {}).get("temperature")
    if recorded is not None:
        return f"{float(recorded):.1f}", "report"
    if str(report.get("timestamp", ""))[:10] < GREEDY_DEFAULT_SINCE:
        return LEGACY_TEMPERATURE, "pre-cutover default"
    return DEFAULT_TEMPERATURE, "dirname"


def parse_set_label(label: str) -> tuple[str, int, str, int | None]:
    """Split a set directory name into (base, turn budget, temperature, sample)."""
    rest = label
    sample = None
    if m := SET_SAMPLE_RE.search(rest):
        sample = int(m.group(1))
        rest = SET_SAMPLE_RE.sub("", rest)
    temperature = DEFAULT_TEMPERATURE
    if m := SET_TEMP_RE.search(rest):
        temperature = m.group(1)
        rest = SET_TEMP_RE.sub("", rest)
    budget = DEFAULT_BUDGET
    if m := SET_BUDGET_RE.search(rest):
        budget = int(m.group(1))
        rest = SET_BUDGET_RE.sub("", rest)
    return rest, budget, temperature, sample

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
    base, budget, _temp, _sample = parse_set_label(label)
    rank = SET_ORDER.index(base) if base in SET_ORDER else len(SET_ORDER)
    return (rank, budget, label)


def latest_report(set_dir: Path) -> Path | None:
    """Newest report in a set directory.

    Re-running the same set into the same directory leaves several timestamped
    reports; sorting by name works because the filenames are ISO timestamps.
    """
    reports = sorted(set_dir.glob("benchmark_*.json"))
    return reports[-1] if reports else None


def report_temperature(report_path: Path, set_label: str) -> str:
    """The decoding temperature a report was produced at.

    Read from the report when it records one, else attributed by date: nothing
    could set the temperature before GREEDY_DEFAULT_SINCE, so LLMConfig's 0.7
    applied. An explicit _T suffix on the directory always wins.
    """
    try:
        with report_path.open() as fh:
            report = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return DEFAULT_TEMPERATURE
    return resolve_temperature(report, set_label)[0]


def latest_per_regime(set_dir: Path) -> list[Path]:
    """Newest report per decoding temperature in a set directory.

    Greedy carries no directory suffix, so a greedy re-run lands in the same
    directory as a pre-cutover 0.7 run and `latest_report` would hide the older
    regime behind the newer one — silently dropping the temperature comparison
    the results chapter is built on. Keyed by regime, both survive.
    """
    newest: dict[str, Path] = {}
    for report in sorted(set_dir.glob("benchmark_*.json")):
        newest[report_temperature(report, set_dir.name)] = report
    return list(newest.values())


def summarise(report_path: Path, set_label: str) -> dict:
    with report_path.open() as fh:
        report = json.load(fh)

    results = report.get("results") or []
    base_set, budget, temperature, sample = parse_set_label(set_label)

    # Resolve the temperature against the report itself wherever possible: an
    # unsuffixed directory is an absence of evidence, not evidence of greedy.
    temperature, temp_source = resolve_temperature(report, set_label)

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
        "base_set": base_set,
        "temperature": temperature,
        "temperature_source": temp_source,
        "sample": sample,
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
            for report in latest_per_regime(set_dir):
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

    # 'temp' is not decoration: greedy leaves no directory suffix, so one set
    # directory can hold both a greedy and a 0.7 run and the two rows would
    # otherwise read as duplicates of the same measurement.
    head = (
        f"{'model':<24} {'set':<14} {'temp':>5} {'budget':>6} {'n':>5} {'ok':>4} {'rate':>7} "
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
                f"{(r['temperature'] + ('?' if r['temperature_source'] == 'pre-cutover default' else '')):>5} "
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

    print("'temp' marked '?' is attributed, not recorded: the run predates the flag")
    print("that sets the temperature, so LLMConfig's 0.7 applied and the directory")
    print("carries no suffix to say so. Runs since then record it in the report.")
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


def render_variability(by_model: dict[str, list[dict]]) -> None:
    """Aggregate repetitions of the same configuration into mean +/- sd.

    Groups on (model, base set, turn budget, temperature) so repetitions of one
    configuration collapse into one row, while a different temperature or turn
    budget stays a separate row — those are different experiments, not samples of
    the same one.

    The spread reported is the standard deviation ACROSS RUNS. It answers a
    different question from the binomial interval on a single run: that one asks
    how precisely 432 tasks estimate a model's true rate, this one asks how much
    the whole pipeline moves when nothing is changed. Both belong in a results
    chapter and neither substitutes for the other.
    """
    groups: dict[tuple, list[dict]] = {}
    for model, rows in by_model.items():
        for r in rows:
            if r.get("sample") is None:
                continue
            groups.setdefault((model, r["base_set"], r["budget"], r["temperature"]), []).append(r)

    if not groups:
        print("No repeated runs found. Submit them with:")
        print("  bash jureca/submit_all.sh --samples 5 --temperature 0.7 ...")
        print("A run without --samples writes no _sN suffix and cannot be grouped.")
        return

    head = (
        f"{'model':<24} {'set':<14} {'T':>5} {'runs':>5} {'n':>6} "
        f"{'mean':>7} {'sd':>6} {'min':>7} {'max':>7} {'range':>7}"
    )
    print(head)
    print("-" * len(head))
    for (model, base, budget, temp), rows in sorted(groups.items()):
        rates = sorted(r["success_rate_pct"] for r in rows)
        ns = {r["total"] for r in rows}
        sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
        n_label = str(ns.pop()) if len(ns) == 1 else "mixed"
        print(
            f"{model:<24} {base:<14} {temp:>5} {len(rates):>5} {n_label:>6} "
            f"{statistics.mean(rates):>6.1f}% {sd:>5.1f} "
            f"{rates[0]:>6.1f}% {rates[-1]:>6.1f}% {rates[-1] - rates[0]:>6.1f}"
        )
        if len(ns) > 0:
            print(f"  ! {model}/{base}: repetitions cover different task counts",
                  file=sys.stderr)
    print()
    print("sd is the spread ACROSS RUNS of one configuration — how much the pipeline")
    print("moves when nothing is changed. It is not the binomial interval on a single")
    print("run, which measures how precisely n tasks estimate the model's true rate.")


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
        "--variability",
        action="store_true",
        help="aggregate repeated runs (the _sN directories written by --samples) "
        "into mean and standard deviation per configuration",
    )
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
    elif args.variability:
        render_variability(by_model)
    else:
        render(by_model, args.errors, args.turn_errors)


if __name__ == "__main__":
    main()
