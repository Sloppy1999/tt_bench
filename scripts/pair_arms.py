#!/usr/bin/env python3
"""Paired comparison of two harness-ablation arms on the same task set.

Implements the analysis plan in HARNESS_ABLATION_PLAN.md section 4. The unit is
the TASK, paired across arms, because the run-to-run spread of this pipeline is
0.2-0.6 pp while the binomial interval at n=432 is +/-4.7 pp
(docs/results_analysis.md section 7.2): repetitions buy almost nothing and
pairing buys a factor of ~2.5 in sensitivity.

Two properties matter more than the p-value:

* **The task sets must be identical.** A differing --pattern or --tiers between
  arms silently changes what is being compared, and an intersection would hide
  it. This script exits non-zero on a mismatch rather than comparing the overlap.

* **Successes are split by route.** A gain that lives entirely in the
  best-placement fallback or the early-stop path is a scoring-credit shift, not
  a capability gain, and must not be reported as one. `fb-auto` is especially
  exposed: its injected simulation sets the same flags an agent-initiated call
  would.

Usage:
    python scripts/pair_arms.py BASELINE_DIR ARM_DIR
    python scripts/pair_arms.py .../scaled_1comp_s1 .../scaled_1comp_afb-auto
    python scripts/pair_arms.py BASE ARM --json
    python scripts/pair_arms.py BASE ARM1 ARM2 ...      # one row per arm

Each directory is a run directory containing per_task/ (written whenever
--save-report is passed), or the per_task/ directory itself.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def per_task_dir(path: Path) -> Path:
    if (path / "per_task").is_dir():
        return path / "per_task"
    if path.name == "per_task" and path.is_dir():
        return path
    sys.exit(f"No per_task/ directory under {path}. Was the run saved with --save-report?")


def load_arm(path: Path) -> dict[str, dict]:
    """task_id -> result, for agentic_synthesis tasks only."""
    tasks: dict[str, dict] = {}
    for f in sorted(per_task_dir(path).glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  ! unreadable, skipped: {f.name}", file=sys.stderr)
            continue
        if rec.get("task_type") != "agentic_synthesis":
            continue
        task_id = rec.get("task_id")
        if task_id is None:
            continue
        if task_id in tasks:
            sys.exit(f"Duplicate task_id {task_id!r} in {path}")
        tasks[task_id] = rec
    if not tasks:
        sys.exit(f"No agentic_synthesis results in {path}")
    return tasks


# ---------------------------------------------------------------------------
# Outcome classification (plan section 4.4)
# ---------------------------------------------------------------------------


def turns_of(rec: dict) -> int | None:
    met = rec.get("metrics") or {}
    turns = met.get("turns")
    return turns if isinstance(turns, int) else None


def success_route(rec: dict) -> str:
    """How a successful task was credited.

    This is the difference between a model that solved the puzzle and a harness
    that gave it credit: `fallback` means the agent built a correct board and
    then removed it, `early-stop` that the loop ended the episode itself on a
    verified board, `ceiling-valid` that the budget expired on a board that
    happened to validate.

    The runner records it directly (``metrics.credit_route``) because `fallback`
    cannot be recovered from the stored record: it keeps the credited board, not
    the executor's final one. Runs made before that field existed fall back to a
    heuristic that can see everything except `fallback`.
    """
    met = rec.get("metrics") or {}
    recorded = met.get("credit_route")
    if recorded:
        return recorded

    raw = (rec.get("llm_response") or "").strip()
    parsed = None
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, dict) and parsed.get("solution_found") is True:
        return "early-stop?"
    if raw:
        return "submitted?"
    return "ceiling-valid?"


def failure_route(rec: dict, budget: int) -> str:
    turns = turns_of(rec)
    error = (rec.get("error") or "").lower()
    if "max turns" in error or (turns is not None and budget and turns >= budget):
        return "ceiling"
    if "timeout" in error or "exception" in error or "empty response" in error:
        return "harness"
    if (rec.get("llm_response") or "").strip():
        return "converged-early"
    return "no-answer-early"


def classify(rec: dict, budget: int) -> str:
    return (
        f"ok:{success_route(rec)}"
        if rec.get("success")
        else f"fail:{failure_route(rec, budget)}"
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_diff_ci(b: int, c: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """95% CI for the paired difference in proportions (Agresti-Min).

    Uses the Wald interval with the Agresti-Caffo style +z^2/2 adjustment spread
    over the discordant cells, which keeps coverage sane when b+c is small — the
    common case for an arm that changes few tasks.
    """
    adj = z * z / 2.0
    nn = n + 2 * adj
    bb, cc = b + adj, c + adj
    diff = (bb - cc) / nn
    var = (bb + cc - (bb - cc) ** 2 / nn) / (nn * nn)
    half = z * math.sqrt(max(var, 0.0))
    return (100 * (diff - half), 100 * (diff + half))


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(base: dict[str, dict], arm: dict[str, dict], budget: int, labels: tuple[str, str]) -> dict:
    base_ids, arm_ids = set(base), set(arm)
    if base_ids != arm_ids:
        only_base = sorted(base_ids - arm_ids)[:5]
        only_arm = sorted(arm_ids - base_ids)[:5]
        sys.exit(
            f"Task sets differ: {labels[0]} has {len(base_ids)}, {labels[1]} has "
            f"{len(arm_ids)}, {len(base_ids & arm_ids)} shared.\n"
            f"  only in {labels[0]}: {only_base}\n"
            f"  only in {labels[1]}: {only_arm}\n"
            "Comparing the intersection would hide a --pattern or --tiers "
            "mismatch between the arms. Re-run the arm on the same task list."
        )

    ids = sorted(base_ids)
    a = d = b = c = 0  # both ok / both fail / base fail->arm ok / base ok->arm fail
    flipped_to_ok, flipped_to_fail = [], []
    for task_id in ids:
        ok_base = bool(base[task_id].get("success"))
        ok_arm = bool(arm[task_id].get("success"))
        if ok_base and ok_arm:
            a += 1
        elif not ok_base and not ok_arm:
            d += 1
        elif ok_arm:
            b += 1
            flipped_to_ok.append(task_id)
        else:
            c += 1
            flipped_to_fail.append(task_id)

    n = len(ids)
    lo, hi = paired_diff_ci(b, c, n)

    def routes(source: dict[str, dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task_id in ids:
            counts[classify(source[task_id], budget)] = (
                counts.get(classify(source[task_id], budget), 0) + 1
            )
        return dict(sorted(counts.items()))

    def numeric(source: dict[str, dict], key: str) -> list[float]:
        out = []
        for task_id in ids:
            value = source[task_id].get(key)
            if isinstance(value, (int, float)):
                out.append(float(value))
        return out

    def turns(source: dict[str, dict]) -> list[float]:
        return [
            float(t)
            for t in (turns_of(source[task_id]) for task_id in ids)
            if t is not None
        ]

    base_tokens, arm_tokens = numeric(base, "tokens_used"), numeric(arm, "tokens_used")
    base_ok, arm_ok = a + c, a + b

    return {
        "baseline": labels[0],
        "arm": labels[1],
        "n": n,
        "budget": budget,
        "baseline_ok": base_ok,
        "arm_ok": arm_ok,
        "baseline_rate_pct": 100.0 * base_ok / n,
        "arm_rate_pct": 100.0 * arm_ok / n,
        "delta_pp": 100.0 * (arm_ok - base_ok) / n,
        "ci95_pp": [lo, hi],
        "both_ok": a,
        "both_fail": d,
        "gained": b,
        "lost": c,
        "discordant_pct": 100.0 * (b + c) / n,
        "mcnemar_p": mcnemar_exact_p(b, c),
        "baseline_routes": routes(base),
        "arm_routes": routes(arm),
        "baseline_tokens_mean": mean_or_none(base_tokens),
        "arm_tokens_mean": mean_or_none(arm_tokens),
        "baseline_turns_mean": mean_or_none(turns(base)),
        "arm_turns_mean": mean_or_none(turns(arm)),
        "gained_task_ids": flipped_to_ok,
        "lost_task_ids": flipped_to_fail,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(result: dict) -> None:
    n = result["n"]
    print(f"\n{result['baseline']}  ->  {result['arm']}   (n = {n}, budget {result['budget']})")
    print("-" * 78)
    print(f"  baseline  {result['baseline_ok']:>5} / {n}   {result['baseline_rate_pct']:>6.2f} %")
    print(f"  arm       {result['arm_ok']:>5} / {n}   {result['arm_rate_pct']:>6.2f} %")
    lo, hi = result["ci95_pp"]
    print(
        f"  delta     {result['delta_pp']:>+6.2f} pp   95% CI [{lo:+.2f}, {hi:+.2f}]   "
        f"McNemar exact p = {result['mcnemar_p']:.4g}"
    )
    print()
    print("  paired outcomes                    arm fail   arm ok")
    print(f"    baseline fail                    {result['both_fail']:>8}   {result['gained']:>6}")
    print(f"    baseline ok                      {result['lost']:>8}   {result['both_ok']:>6}")
    print(
        f"  discordant {result['gained'] + result['lost']} of {n} "
        f"({result['discordant_pct']:.1f} %)  — the design's sensitivity depends on this"
    )

    print("\n  outcome routes (plan section 4.4)")
    keys = sorted(set(result["baseline_routes"]) | set(result["arm_routes"]))
    print(f"    {'route':<22} {'baseline':>9} {'arm':>9} {'delta':>7}")
    for key in keys:
        base_n = result["baseline_routes"].get(key, 0)
        arm_n = result["arm_routes"].get(key, 0)
        print(f"    {key:<22} {base_n:>9} {arm_n:>9} {arm_n - base_n:>+7}")

    print("\n  cost")
    for label, base_key, arm_key in (
        ("tokens / task", "baseline_tokens_mean", "arm_tokens_mean"),
        ("turns / task", "baseline_turns_mean", "arm_turns_mean"),
    ):
        base_v, arm_v = result[base_key], result[arm_key]
        if base_v is None or arm_v is None:
            continue
        ratio = arm_v / base_v if base_v else float("nan")
        print(f"    {label:<22} {base_v:>9,.0f} {arm_v:>9,.0f} {ratio:>6.2f}x")
    if result["arm_tokens_mean"] and result["baseline_tokens_mean"]:
        for who, ok, tokens in (
            ("baseline", result["baseline_ok"], result["baseline_tokens_mean"]),
            ("arm", result["arm_ok"], result["arm_tokens_mean"]),
        ):
            per = 1e5 * ok / (tokens * n) if tokens else 0
            print(f"    successes / 100k tok  {who:<10} {per:>18.2f}")

    # A gain that lives entirely in the routes the agent did not drive is a
    # scoring-credit shift. Flag it rather than leaving it to be noticed.
    shifted = sum(
        result["arm_routes"].get(f"ok:{route}", 0)
        - result["baseline_routes"].get(f"ok:{route}", 0)
        for route in ("fallback", "ceiling-valid", "early-stop")
    )
    if result["gained"] and shifted >= result["gained"] - result["lost"] > 0:
        print(
            "\n  ! the net gain is fully accounted for by the fallback, "
            "ceiling-valid and\n    early-stop routes — credit the agent did not "
            "drive. Read this as a scoring\n    shift, not a capability gain "
            "(HARNESS_ABLATION_PLAN.md section 4.4)."
        )
    if any(r.endswith("?") for r in result["arm_routes"]) or any(
        r.endswith("?") for r in result["baseline_routes"]
    ):
        print(
            "\n  ! routes marked '?' are inferred: the run predates "
            "metrics.credit_route,\n    so the best-placement fallback is "
            "indistinguishable from a submitted answer."
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Paired McNemar comparison of harness-ablation arms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("baseline", type=Path, help="baseline run directory (or its per_task/)")
    ap.add_argument("arms", type=Path, nargs="+", help="one or more arm run directories")
    ap.add_argument(
        "--budget",
        type=int,
        default=25,
        help="turn budget the runs used, for the ceiling split (default: 25). "
        "A turn-budget arm must be given ITS budget, not the baseline's.",
    )
    ap.add_argument("--json", action="store_true", help="emit results as JSON")
    args = ap.parse_args()

    base = load_arm(args.baseline)
    results = [
        compare(base, load_arm(arm), args.budget, (args.baseline.name, arm.name))
        for arm in args.arms
    ]

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
        return 0

    for result in results:
        render(result)
    print()
    print("p is a two-sided exact McNemar test on the discordant pairs only; the")
    print("concordant tasks carry no information about the difference. At n=432 and")
    print("~8% discordance the design resolves about 3.8 pp at 80% power, 4.8 pp")
    print("after Holm over seven arms — a flat cell means 'no effect above that',")
    print("never 'no effect'. See HARNESS_ABLATION_PLAN.md section 4.3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
