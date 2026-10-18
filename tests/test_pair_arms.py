"""Coverage for the paired arm comparison (HARNESS_ABLATION_PLAN.md section 4).

This script produces the numbers the ablation's conclusions rest on, so the
statistics are checked against scipy and the two guards that protect the paired
design — identical task sets, and credit routes read rather than inferred — are
checked directly.
"""

import json

import pytest
from scipy.stats import binomtest

from scripts.pair_arms import (
    classify,
    compare,
    load_arm,
    mcnemar_exact_p,
    paired_diff_ci,
    success_route,
)


def _task(task_id, *, success, route=None, turns=5, tokens=1000, error="", response="x"):
    metrics = {"turns": turns}
    if route is not None:
        metrics["credit_route"] = route
    return {
        "task_id": task_id,
        "task_type": "agentic_synthesis",
        "success": success,
        "llm_response": response,
        "predicted": {"final_solution": []},
        "metrics": metrics,
        "error": error,
        "tokens_used": tokens,
    }


def _write(directory, records):
    per_task = directory / "per_task"
    per_task.mkdir(parents=True, exist_ok=True)
    for rec in records:
        (per_task / f"{rec['task_id']}__agentic_synthesis.json").write_text(json.dumps(rec))
    return directory


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "b,c", [(0, 0), (5, 0), (0, 5), (10, 5), (3, 7), (25, 12), (40, 40), (60, 31), (100, 70)]
)
def test_mcnemar_matches_scipy(b, c):
    expected = binomtest(b, b + c, 0.5, alternative="two-sided").pvalue if b + c else 1.0
    assert mcnemar_exact_p(b, c) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("b,c", [(0, 0), (30, 10), (10, 30), (20, 20), (5, 1)])
def test_paired_ci_brackets_the_point_estimate(b, c):
    lo, hi = paired_diff_ci(b, c, 432)
    delta = 100.0 * (b - c) / 432
    assert lo <= delta <= hi
    assert (lo + hi) / 2 == pytest.approx(delta, abs=0.35)


def test_ci_width_reflects_only_the_discordant_pairs():
    """More concordant tasks tighten the interval; more discordant widen it."""
    narrow = paired_diff_ci(5, 5, 432)
    wide = paired_diff_ci(50, 50, 432)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


def test_recorded_credit_route_is_preferred_over_inference():
    rec = _task("t", success=True, route="fallback", response="")
    assert success_route(rec) == "fallback"


def test_inferred_routes_are_marked_as_inferred():
    """A run predating metrics.credit_route cannot show the fallback at all."""
    early = _task("t", success=True, response='{"content": "", "solution_found": true}')
    submitted = _task("t", success=True, response='{"final_solution": []}')
    expired = _task("t", success=True, response="")

    assert success_route(early) == "early-stop?"
    assert success_route(submitted) == "submitted?"
    assert success_route(expired) == "ceiling-valid?"


def test_failures_split_by_regime():
    ceiling = _task("t", success=False, turns=25, error="Max turns (25) reached", response="")
    converged = _task("t", success=False, turns=9, error="Illegal free fall", response="{}")
    silent = _task("t", success=False, turns=9, error="Illegal free fall", response="")
    broken = _task("t", success=False, turns=3, error="Exception: boom", response="")

    assert classify(ceiling, 25) == "fail:ceiling"
    assert classify(converged, 25) == "fail:converged-early"
    assert classify(silent, 25) == "fail:no-answer-early"
    assert classify(broken, 25) == "fail:harness"


def test_ceiling_is_relative_to_the_arms_own_budget():
    """A t40 arm's failure at turn 25 is not budget exhaustion."""
    rec = _task("t", success=False, turns=25, error="Illegal free fall", response="{}")
    assert classify(rec, 25) == "fail:ceiling"
    assert classify(rec, 40) == "fail:converged-early"


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_identical_runs_give_no_difference(tmp_path):
    records = [_task(f"t{i}", success=i % 3 == 0) for i in range(12)]
    base = _write(tmp_path / "base", records)
    arm = _write(tmp_path / "arm", records)

    result = compare(load_arm(base), load_arm(arm), 25, ("base", "arm"))

    assert result["delta_pp"] == 0
    assert result["mcnemar_p"] == 1.0
    assert result["gained"] == result["lost"] == 0


def test_discordant_pairs_are_counted_in_the_right_cells(tmp_path):
    base = _write(
        tmp_path / "base",
        [_task("a", success=True), _task("b", success=True), _task("c", success=False),
         _task("d", success=False)],
    )
    arm = _write(
        tmp_path / "arm",
        [_task("a", success=True), _task("b", success=False), _task("c", success=True),
         _task("d", success=False)],
    )

    result = compare(load_arm(base), load_arm(arm), 25, ("base", "arm"))

    assert (result["both_ok"], result["lost"], result["gained"], result["both_fail"]) == (1, 1, 1, 1)
    assert result["delta_pp"] == 0
    assert result["gained_task_ids"] == ["c"]
    assert result["lost_task_ids"] == ["b"]


def test_a_differing_task_set_aborts_rather_than_intersecting(tmp_path):
    """A --pattern or --tiers mismatch must not be silently absorbed."""
    base = _write(tmp_path / "base", [_task(f"t{i}", success=True) for i in range(5)])
    arm = _write(tmp_path / "arm", [_task(f"t{i}", success=True) for i in range(4)])

    with pytest.raises(SystemExit) as excinfo:
        compare(load_arm(base), load_arm(arm), 25, ("base", "arm"))

    assert "Task sets differ" in str(excinfo.value)


def test_understanding_results_are_ignored(tmp_path):
    """Only the agentic loop has a harness to ablate."""
    records = [_task("a", success=True)]
    other = dict(_task("a", success=False), task_type="understanding")
    other["task_id"] = "a-q1"
    base = _write(tmp_path / "base", records + [other])

    assert set(load_arm(base)) == {"a"}


def test_cost_columns_come_from_the_runs(tmp_path):
    base = _write(tmp_path / "base", [_task("a", success=True, tokens=100, turns=4)])
    arm = _write(tmp_path / "arm", [_task("a", success=True, tokens=300, turns=12)])

    result = compare(load_arm(base), load_arm(arm), 25, ("base", "arm"))

    assert result["baseline_tokens_mean"] == 100
    assert result["arm_tokens_mean"] == 300
    assert result["baseline_turns_mean"] == 4
    assert result["arm_turns_mean"] == 12
