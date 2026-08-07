#!/usr/bin/env python3
"""Count truncated generations per model and decoding regime.

A turn that ends with finish_reason=length was cut off mid-generation: the model
hit max_tokens (32768) before it finished speaking. That is not the same failure
as running out of turns, and the two are easy to confuse — a run can look like
it "converged early" on the turn metric while every turn it did take was
silently amputated.

The count is not in the reports. client.py detects the condition and emits a
logger.warning, and nothing carries it any further, so the only record is the
benchmark log. This reads those logs.

    python jureca/count_truncations.py
    python jureca/count_truncations.py --set scaled_2comp
    python jureca/count_truncations.py --all-jobs

Attribution caveat, stated because it bounds what the number means: the harness
runs tasks concurrently and the log interleaves them, so a truncation warning
cannot be tied back to the task that produced it. These are counts of truncated
TURNS, not of affected tasks. The turn numbers in the warnings are per-task, so
their distribution still says where in a task's life the cut happens.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_results import (  # noqa: E402
    DEFAULT_BUDGET,
    parse_set_label,
    resolve_temperature,
    set_sort_key,
)

# bench_<model>_<set label>_<job id>.log — model names are hyphenated and set
# labels are not, so the first field is the model and the last is the job id.
LOG_RE = re.compile(r"^bench_(?P<rest>.+)_(?P<job>\d+|manual)\.log$")
TRUNC_RE = re.compile(r"turn (\d+): response truncated \(finish_reason=length\)")
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T]")
STARTED_RE = re.compile(r"Running agentic synthesis task:")
DONE_RE = re.compile(r"Completed .* → \d+ result")


def scan(path: Path) -> dict | None:
    """One log file: model, set, regime, truncations, task counts."""
    m = LOG_RE.match(path.name)
    if not m:
        return None
    fields = m.group("rest").split("_")
    if len(fields) < 2:
        return None
    model, label = fields[0], "_".join(fields[1:])

    run_date = ""
    trunc_turns: list[int] = []
    started = done = 0
    try:
        with path.open(errors="replace") as fh:
            for line in fh:
                if not run_date and (d := DATE_RE.match(line)):
                    run_date = d.group(1)
                if t := TRUNC_RE.search(line):
                    trunc_turns.append(int(t.group(1)))
                elif STARTED_RE.search(line):
                    started += 1
                elif DONE_RE.search(line):
                    done += 1
    except OSError:
        return None

    # resolve_temperature needs only these two keys; the log's first timestamp
    # stands in for the report's, which is what the date attribution keys on.
    temp, temp_source = resolve_temperature({"timestamp": run_date}, label)
    base, budget, _t, sample = parse_set_label(label)
    return {
        "log": path,
        "model": model,
        "set": label,
        "base_set": base,
        "budget": budget,
        "sample": sample,
        "date": run_date,
        "temperature": temp,
        "temperature_source": temp_source,
        "job": m.group("job"),
        "truncations": len(trunc_turns),
        "trunc_turns": trunc_turns,
        "started": started,
        "completed": done,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", default="slurm_logs", type=Path)
    ap.add_argument("--set", dest="set_filter", help="restrict to one base set label")
    ap.add_argument("--model", dest="model_filter")
    ap.add_argument("--all-jobs", action="store_true",
                    help="list every log; by default only the newest job per "
                         "(model, set, regime) is kept, matching inspect_results")
    args = ap.parse_args()

    if not args.logs_dir.is_dir():
        sys.exit(f"No such log directory: {args.logs_dir}")

    runs = [r for p in sorted(args.logs_dir.glob("bench_*.log")) if (r := scan(p))]
    if args.set_filter:
        runs = [r for r in runs if r["base_set"] == args.set_filter]
    if args.model_filter:
        runs = [r for r in runs if r["model"] == args.model_filter]
    if not runs:
        sys.exit("No benchmark logs matched.")

    if not args.all_jobs:
        newest: dict[tuple, dict] = {}
        for r in runs:
            key = (r["model"], r["set"], r["temperature"], r["sample"])
            prev = newest.get(key)
            if prev is None or (r["date"], r["job"]) >= (prev["date"], prev["job"]):
                newest[key] = r
        runs = list(newest.values())

    runs.sort(key=lambda r: (r["model"], set_sort_key(r["set"]), r["temperature"]))

    head = (f"{'model':<24} {'set':<14} {'temp':>5} {'tasks':>6} {'trunc':>7} "
            f"{'per task':>9} {'turn min/med/max':>18} {'job':>10}")
    print(head)
    print("-" * len(head))

    by_regime: dict[tuple, list[dict]] = defaultdict(list)
    last_model = None
    for r in runs:
        tasks = r["completed"] or r["started"]
        per = f"{r['truncations'] / tasks:.2f}" if tasks else "-"
        turns = r["trunc_turns"]
        spread = (f"{min(turns)}/{int(statistics.median(turns))}/{max(turns)}"
                  if turns else "-")
        temp = r["temperature"] + ("?" if r["temperature_source"] == "pre-cutover default" else "")
        # Rebuild the label without the _T suffix: it has its own column, and
        # repeating it overflows this one and reads as two different sets.
        # Rebuilt rather than stripped because _T is not always last (a repeated
        # run ends _T0.7_s3) and an anchored sub would miss it there.
        label = (r["base_set"]
                 + (f"_t{r['budget']}" if r["budget"] != DEFAULT_BUDGET else "")
                 + (f"_s{r['sample']}" if r["sample"] else ""))
        print(f"{r['model'] if r['model'] != last_model else '':<24} "
              f"{label:<14} {temp:>5} {tasks:>6} {r['truncations']:>7} "
              f"{per:>9} {spread:>18} {r['job']:>10}")
        last_model = r["model"]
        by_regime[(r["base_set"], r["temperature"])].append(r)

    print("\nTruncated TURNS, not tasks: the log interleaves concurrent tasks, so a")
    print("warning cannot be tied back to the task that raised it. 'tasks' counts")
    print("completions in that log, so 'per task' is a rate, not a proportion.")
    print("A run still in flight reports what it has produced so far.")

    print("\nBy regime (mean truncations per task across models):")
    for (base, temp) in sorted(by_regime, key=lambda k: (set_sort_key(k[0]), k[1])):
        group = by_regime[(base, temp)]
        rates = [r["truncations"] / t for r in group if (t := r["completed"] or r["started"])]
        if rates:
            print(f"  {base:<14} T={temp:<5} {statistics.mean(rates):>6.2f}  "
                  f"({len(rates)} model(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
