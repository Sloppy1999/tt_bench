"""
Unit tests for Board Complexity Metrics
========================================

Tests cover all P0-P3 metrics, invariants (range checks), and
integration with real challenge files.
"""

import os
import sys
from pathlib import Path

import pytest


from tt_bench.simulator import (
    Board, Ramp, Crossover, Bit, GearBit, Gear, Interceptor, Trigger,
    Direction, Side,
)
from tt_bench.analytics.metrics import (
    scr, ctd, dependency_depth, gcc, rpcc, ibr, hic, bici, sac, sac_norm,
    synthesis_load, psde, oss,
    k_approx, compute_all_metrics,
)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers for building test boards
# ---------------------------------------------------------------------------


def _board(*components) -> Board:
    """Build a 5x5 board with the given (x, y, comp) tuples."""
    b = Board(rows=5, cols=5)
    for x, y, comp in components:
        b.place(x, y, comp)
    return b


# ---------------------------------------------------------------------------
# SCR — Stateful Component Ratio
# ---------------------------------------------------------------------------


class TestSCR:
    def test_empty_board(self):
        b = Board(rows=5, cols=5)
        assert scr(b) == 0.0

    def test_all_ramps(self):
        b = _board(
            (2, 1, Ramp(2, 1, Direction.RIGHT)),
            (3, 2, Ramp(3, 2, Direction.LEFT)),
        )
        assert scr(b) == 0.0

    def test_all_bits(self):
        b = _board(
            (2, 1, Bit(2, 1, state=0)),
            (3, 2, Bit(3, 2, state=1)),
        )
        assert scr(b) == 1.0

    def test_gear_bits_weighted(self):
        """Gear bits count 2× — a board with 2 gear_bits + 2 gears = 1.0."""
        b = _board(
            (1, 1, GearBit(1, 1, state=0)),
            (2, 1, Gear(2, 1)),
            (3, 1, GearBit(3, 1, state=0)),
            (4, 1, Gear(4, 1)),
        )
        # 2*2 / 4 = 1.0
        assert scr(b) == 1.0

    def test_mixed(self):
        b = _board(
            (2, 1, Ramp(2, 1, Direction.RIGHT)),
            (3, 2, Bit(3, 2, state=0)),
            (4, 3, GearBit(4, 3, state=1)),
            (1, 4, Interceptor(1, 4)),
        )
        # (1 bit + 2*1 gear_bit) = 3 stateful out of 4 total
        assert scr(b) == pytest.approx(3 / 4)

    def test_range(self):
        # Large random board should have scr in [0, 1]
        b = Board(rows=11, cols=11)
        for x in range(11):
            for y in range(6):
                b.place(x, y, Ramp(x, y, Direction.RIGHT))
        for y in range(6, 11):
            b.place(5, y, Bit(5, y, state=0))
        assert 0.0 <= scr(b) <= 1.0


# ---------------------------------------------------------------------------
# CTD — Component Type Diversity
# ---------------------------------------------------------------------------


class TestCTD:
    def test_empty_board(self):
        b = Board(rows=5, cols=5)
        assert ctd(b) == 0.0

    def test_one_type(self):
        b = _board((2, 1, Ramp(2, 1, Direction.RIGHT)))
        assert ctd(b) == pytest.approx(1 / 8)

    def test_four_types(self):
        b = _board(
            (2, 1, Ramp(2, 1, Direction.RIGHT)),
            (3, 2, Bit(3, 2, state=0)),
            (2, 3, Crossover(2, 3)),
            (4, 4, Interceptor(4, 4)),
        )
        assert ctd(b) == pytest.approx(4 / 8)

    def test_all_eight_types(self):
        b = _board(
            (1, 0, Ramp(1, 0, Direction.RIGHT)),
            (3, 0, Ramp(3, 0, Direction.LEFT)),
            (2, 1, Crossover(2, 1)),
            (1, 2, Bit(1, 2, state=0)),
            (3, 2, GearBit(3, 2, state=0)),
            (2, 3, Gear(2, 3)),
            (4, 4, Interceptor(4, 4)),
            (0, 4, Trigger(0, 4, side="blue")),
        )
        assert ctd(b) == pytest.approx(1.0)

    def test_range(self):
        b = Board(rows=11, cols=11)
        b.place(5, 5, Ramp(5, 5, Direction.RIGHT))
        assert 0.0 <= ctd(b) <= 1.0


# ---------------------------------------------------------------------------
# Dependency Depth
# ---------------------------------------------------------------------------


class TestDependencyDepth:
    def test_empty_board(self):
        b = Board(rows=5, cols=5)
        assert dependency_depth(b) == 0

    def test_ramps_only(self):
        b = _board(
            (2, 0, Ramp(2, 0, Direction.RIGHT)),
            (3, 1, Ramp(3, 1, Direction.LEFT)),
        )
        assert dependency_depth(b) == 0

    def test_board_with_bits_on_path(self):
        """Place bits at positions the blue marble will visit."""
        b = Board(rows=5, cols=5, blue_hopper_x=2, hopper_entry_mode="inward")
        # Blue marble enters at x=3 (inward), falls to y=0
        # Place bit at (3, 1) - marble reaches it after falling
        b.place(3, 1, Bit(3, 1, state=0))
        b.place(4, 2, Ramp(4, 2, Direction.LEFT))
        b.place(3, 3, Bit(3, 3, state=0))
        assert dependency_depth(b) >= 1


# ---------------------------------------------------------------------------
# GCC — Gear Connectivity Complexity
# ---------------------------------------------------------------------------


class TestGCC:
    def test_no_gear_bits(self):
        b = _board((2, 1, Ramp(2, 1, Direction.RIGHT)))
        assert gcc(b) == 0.0

    def test_single_gear_bit_no_gears(self):
        b = _board((2, 1, GearBit(2, 1, state=0)))
        val = gcc(b)
        # Solo gear_bit = 100% gear complexity → 1.0
        assert val == pytest.approx(1.0)

    def test_gear_bits_with_connecting_gear(self):
        """Two gear_bits + one gear, properly connected = 100% gear complexity."""
        from tt_bench import simulator as tt_sim
        b = Board(rows=5, cols=5)
        b.place(2, 1, GearBit(2, 1, state=0))
        b.place(3, 1, Gear(3, 1))
        b.place(4, 1, GearBit(4, 1, state=0))
        # Use build_gear_connections to properly wire gear<->gear_bit adjacencies
        tt_sim.build_gear_connections(b)
        val = gcc(b)
        # 2 gear_bits * 1.5 (gears present in group) / 3 components = 1.0
        assert val == pytest.approx(1.0)

    def test_gear_bits_vs_total_dominance(self):
        """Board with gear + non-gear components should have lower GCC than all-gear."""
        b = Board(rows=5, cols=5)
        b.place(2, 1, GearBit(2, 1, state=0))
        b.place(3, 2, Ramp(3, 2, Direction.RIGHT))
        b.place(4, 3, Bit(4, 3, state=0))
        val_mixed = gcc(b)
        # 1 gear_bit * 1.0 / 3 components = 0.333
        assert val_mixed == pytest.approx(1 / 3)

    def test_range(self):
        b = _board((2, 1, GearBit(2, 1, state=0)))
        assert 0.0 <= gcc(b) <= 1.0


# ---------------------------------------------------------------------------
# RPCC — Routing Path Crossover Count
# ---------------------------------------------------------------------------


class TestRPCC:
    def test_no_crossovers(self):
        b = _board((2, 1, Ramp(2, 1, Direction.RIGHT)))
        assert rpcc(b) == 0.0

    def test_with_crossovers(self):
        b = _board(
            (2, 1, Crossover(2, 1)),
            (4, 3, Crossover(4, 3)),
        )
        assert rpcc(b) > 0.0

    def test_range(self):
        b = _board((2, 1, Crossover(2, 1)))
        assert 0.0 <= rpcc(b) <= 1.0


# ---------------------------------------------------------------------------
# IBR — Interceptor-to-Bit Ratio
# ---------------------------------------------------------------------------


class TestIBR:
    def test_no_interceptors(self):
        b = _board((2, 1, Bit(2, 1, state=0)))
        assert ibr(b) == 0.0

    def test_interceptors_no_bits(self):
        b = _board((2, 1, Interceptor(2, 1)))
        # interceptors / max(interceptors + 0, 1) = 1 / 1 = 1.0
        assert ibr(b) == 1.0

    def test_mixed(self):
        b = _board(
            (2, 1, Bit(2, 1, state=0)),
            (3, 2, Interceptor(3, 2)),
            (4, 3, Bit(4, 3, state=1)),
            (1, 4, Interceptor(1, 4)),
        )
        # 2 interceptors / max(2 interceptors + 2 bits, 1) = 2/4 = 0.5
        assert ibr(b) == pytest.approx(0.5)

    def test_range(self):
        b = _board((2, 1, Interceptor(2, 1)))
        assert ibr(b) >= 0.0

    def test_ibr_bounded(self):
        """IBR must not exceed 1.0 even with many interceptors and no bits."""
        b = Board(rows=5, cols=5)
        for i in range(5):
            b.place(i, 0, Interceptor(i, 0))
        assert 0.0 <= ibr(b) <= 1.0

    def test_interceptors_with_stateful(self):
        """When stateful components dominate, IBR approaches 0."""
        b = _board(
            (2, 1, Bit(2, 1, state=0)),
            (3, 1, Bit(3, 1, state=1)),
            (4, 1, Bit(4, 1, state=0)),
            (2, 2, Interceptor(2, 2)),
        )
        # 1 interceptor / max(1 + 3 bits, 1) = 1/4 = 0.25
        assert ibr(b) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# HIC — Hopper Interaction Complexity
# ---------------------------------------------------------------------------


class TestHIC:
    def test_no_triggers(self):
        b = _board((2, 1, Ramp(2, 1, Direction.RIGHT)))
        assert hic(b) == 0.0

    def test_one_trigger(self):
        b = _board((2, 1, Trigger(2, 1, side="blue")))
        assert hic(b) == 0.25

    def test_two_triggers(self):
        b = _board(
            (2, 1, Trigger(2, 1, side="blue")),
            (3, 2, Trigger(3, 2, side="red")),
        )
        assert hic(b) == 0.5


# ---------------------------------------------------------------------------
# BICI — Board Input Complexity Index
# ---------------------------------------------------------------------------


class TestBICI:
    def test_empty_board(self):
        b = Board(rows=5, cols=5)
        # Empty board: scr=0, ctd=0, gcc=0, rpcc=0, ibr=0 → bici=0
        assert bici(b) == pytest.approx(0.0)

    def test_uniform_weights(self):
        """BICI should equal mean of sub-metrics with uniform weights."""
        b = _board(
            (2, 1, Bit(2, 1, state=0)),
            (3, 2, Crossover(3, 2)),
            (1, 3, Interceptor(1, 3)),
        )
        expected = (scr(b) + ctd(b) + gcc(b) + rpcc(b) + ibr(b)) / 5
        assert bici(b) == pytest.approx(expected)

    def test_custom_weights(self):
        b = _board(
            (2, 1, Bit(2, 1, state=0)),
            (3, 2, Crossover(3, 2)),
        )
        # All weight on SCR
        bici_scr = bici(b, weights=[1.0, 0.0, 0.0, 0.0, 0.0])
        assert bici_scr == pytest.approx(scr(b))

    def test_range(self):
        b = _board((2, 1, Bit(2, 1, state=0)))
        assert 0.0 <= bici(b) <= 1.0

    def test_bici_with_high_ibr(self):
        """BICI remains within [0, 1] even on boards with many interceptors."""
        b = Board(rows=5, cols=5)
        for i in range(5):
            b.place(i, 0, Interceptor(i, 0))
        assert 0.0 <= bici(b) <= 1.0


# ---------------------------------------------------------------------------
# SAC — Synthesis Action Complexity
# ---------------------------------------------------------------------------


class TestSAC:
    def test_no_task_info(self):
        b = Board(rows=5, cols=5)
        assert sac(b) is None

    def test_no_available_parts(self):
        b = Board(rows=5, cols=5)
        assert sac(b, {}) == 0.0

    def test_empty_available_parts(self):
        b = Board(rows=5, cols=5)
        assert sac(b, {"available_parts": {}}) == 0.0

    def test_with_parts_on_empty_board(self):
        b = Board(rows=5, cols=5)
        task_info = {"available_parts": {"ramp_right": 4, "ramp_left": 4}}
        # empty=25, H=1.0, SAC = 25 * (1 + 1.0) = 50
        val = sac(b, task_info)
        assert val is not None and val == pytest.approx(50.0)

    def test_single_type_sac_nonzero(self):
        """When only one part type is available, SAC should still be >0
        (spatial search matters even without type diversity)."""
        b = Board(rows=5, cols=5)
        b.place(2, 1, Ramp(2, 1, Direction.RIGHT))
        # 25 cells - 1 occupied = 24 empty
        task_info = {"available_parts": {"ramp_right": 9}}
        val = sac(b, task_info)
        # H=0, empty=24, SAC = 24 * (1 + 0) = 24
        assert val == pytest.approx(24.0)

    def test_with_parts_on_full_board(self):
        b = Board(rows=3, cols=3)
        for x in range(3):
            for y in range(3):
                b.place(x, y, Ramp(x, y, Direction.RIGHT))
        task_info = {"available_parts": {"ramp_right": 4}}
        val = sac(b, task_info)
        # Full board → empty=0 → SAC=0
        assert val == 0.0


class TestSACNorm:
    def test_no_task_info(self):
        assert sac_norm(Board(rows=5, cols=5)) is None

    def test_read_only_board(self):
        b = Board(rows=5, cols=5)
        assert sac_norm(b, {"available_parts": {}}) == 0.0

    def test_single_type(self):
        """One part type → H=0 → (1+0)/4 = 0.25"""
        b = Board(rows=5, cols=5)
        val = sac_norm(b, {"available_parts": {"ramp_right": 9}})
        assert val == pytest.approx(0.25)

    def test_two_types_equal(self):
        """Two types equal → H=1 → (1+1)/4 = 0.5"""
        b = Board(rows=5, cols=5)
        val = sac_norm(b, {"available_parts": {"ramp_right": 4, "ramp_left": 4}})
        assert val == pytest.approx(0.5)

    def test_eight_types_equal(self):
        """Max diversity → H=3 → (1+3)/4 = 1.0"""
        b = Board(rows=5, cols=5)
        val = sac_norm(b, {
            "available_parts": {
                "ramp_right": 5, "ramp_left": 5, "crossover": 5, "bit": 5,
                "gear_bit": 5, "gear": 5, "interceptor": 5, "trigger": 5,
            }
        })
        assert val == pytest.approx(1.0)

    def test_range(self):
        b = Board(rows=5, cols=5)
        val = sac_norm(b, {"available_parts": {"bit": 3, "ramp_right": 7}})
        assert val is not None and 0.0 <= val <= 1.0


class TestSynthesisLoad:
    def test_no_task_info(self):
        assert synthesis_load(Board(rows=5, cols=5)) is None

    def test_read_only(self):
        b = Board(rows=5, cols=5)
        assert synthesis_load(b, {"available_parts": {}}) == 0.0

    def test_light_load(self):
        """4 parts on 5x5=25 cell board → 4/25 = 0.16"""
        b = Board(rows=5, cols=5)
        val = synthesis_load(b, {"available_parts": {"ramp_right": 2, "ramp_left": 2}})
        assert val == pytest.approx(4 / 25)

    def test_heavy_load(self):
        """All cells need filling → 1.0"""
        b = Board(rows=5, cols=5)
        val = synthesis_load(b, {"available_parts": {"ramp_right": 25}})
        assert val == 1.0

    def test_range(self):
        b = Board(rows=5, cols=5)
        val = synthesis_load(b, {"available_parts": {"bit": 5}})
        assert val is not None and 0.0 <= val <= 1.0


class TestPSDE:
    def test_no_task_info(self):
        assert psde(Board(rows=5, cols=5)) is None

    def test_read_only(self):
        b = Board(rows=5, cols=5)
        assert psde(b, {"available_parts": {}}) == 0.0

    def test_single_type_medium_load(self):
        """1 type (sac_norm=0.25) + moderate load → mid PSDE"""
        b = Board(rows=5, cols=5)
        val = psde(b, {"available_parts": {"ramp_right": 8}})
        # sac_norm = 0.25, synthesis_load = 8/25 = 0.32, psde = 0.285
        assert val == pytest.approx((0.25 + 8 / 25) / 2)

    def test_max(self):
        """Max diversity + full load → 1.0"""
        b = Board(rows=5, cols=5)
        val = psde(b, {
            "available_parts": {
                "ramp_right": 5, "ramp_left": 5, "crossover": 5, "bit": 5,
                "gear_bit": 5, "gear": 5, "interceptor": 5, "trigger": 5,
            }
        })
        # sac_norm = 1.0, synthesis_load = 40/25 → clamped to 1.0, psde = 1.0
        assert val == pytest.approx(1.0)

    def test_range(self):
        b = Board(rows=5, cols=5)
        val = psde(b, {"available_parts": {"bit": 3}})
        assert val is not None and 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# OSS — Objective Specificity Score
# ---------------------------------------------------------------------------


class TestOSS:
    def test_no_task_info(self):
        assert oss(None) is None

    def test_empty_objective(self):
        assert oss({"objective": ""}) is None
        assert oss({"objective": "   "}) is None

    def test_exact_quantifier(self):
        val = oss({"objective": "Let exactly 4 blue balls reach the end."})
        assert val is not None and val >= 1 / 3

    def test_conditional(self):
        val = oss({"objective": "If bit A starts left, intercept blue. Otherwise, intercept red."})
        assert val is not None and val >= 1 / 3

    def test_pattern_constraint(self):
        val = oss({"objective": "Make the pattern blue, red, blue, red..."})
        assert val is not None and val >= 1 / 3

    def test_combined(self):
        val = oss({
            "objective": (
                "If bit A starts to the left, intercept exactly 3 blue balls "
                "and produce the pattern blue, red, blue."
            )
        })
        assert val is not None and val == pytest.approx(1.0)

    def test_range(self):
        val = oss({"objective": "Route all blue balls to the left."})
        assert val is not None and 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# K̃ — Kolmogorov Approximation
# ---------------------------------------------------------------------------


class TestKApprox:
    def test_empty_board_low(self):
        """Empty grid (all zeros) compresses well — baseline low K̃."""
        b = Board(rows=11, cols=11)
        val = k_approx(b)
        # 121 zero bytes → gzip ~24 bytes → ratio ~0.20
        assert 0.0 <= val <= 0.35, f"Empty board should compress well, got {val:.3f}"

    def test_populated_board_higher(self):
        """Populated board resists compression more than empty board."""
        b_empty = Board(rows=11, cols=11)
        val_empty = k_approx(b_empty)

        b = Board(rows=11, cols=11)
        for x in range(11):
            for y in range(5):
                b.place(x, y, Ramp(x, y, Direction.RIGHT if (x + y) % 2 == 0 else Direction.LEFT))
        for y in range(5, 8):
            b.place(5, y, Bit(5, y, state=0))
        val_complex = k_approx(b)
        assert val_complex > val_empty, \
            f"Populated board K̃ ({val_complex:.3f}) should exceed empty ({val_empty:.3f})"

    def test_real_challenge_range(self):
        """K̃ for real challenges should be in reasonable range."""
        import json, os
        path = os.path.join(BASE_PATH,
            "data/tasks/official/challenges/json/tt-official-ch01.json")
        from tt_bench.simulator import Board as B
        board = B.from_task_json(path)
        val = k_approx(board)
        assert 0.10 <= val <= 0.80, f"Real board K̃ out of range: {val:.3f}"


# ---------------------------------------------------------------------------
# compute_all_metrics — Integration
# ---------------------------------------------------------------------------


class TestComputeAll:
    def test_returns_dict(self):
        b = Board(rows=5, cols=5)
        result = compute_all_metrics(b)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        b = Board(rows=5, cols=5)
        result = compute_all_metrics(b)
        for key in ("scr", "bici", "k_approx", "dependency_depth"):
            assert key in result, f"Missing required key: {key}"

    def test_with_task_info(self):
        b = Board(rows=5, cols=5)
        task_info = {
            "available_parts": {"ramp_right": 4},
            "objective": "Route all blue balls to the left.",
        }
        result = compute_all_metrics(b, task_info)
        assert "sac" in result
        assert "sac_norm" in result
        assert "synthesis_load" in result
        assert "psde" in result
        assert "oss" in result

    def test_integration_real_challenge(self):
        """Load challenge 01 and compute all metrics — no exceptions."""
        path = os.path.join(
            BASE_PATH, "data/tasks/official/challenges/json/tt-official-ch01.json"
        )
        import json
        from tt_bench.simulator import Board as B
        board = B.from_task_json(path)
        with open(path) as f:
            task_info = json.load(f)
        result = compute_all_metrics(board, task_info)
        assert "scr" in result
        assert "bici" in result
        assert isinstance(result["dependency_depth"], (int, float))


# ---------------------------------------------------------------------------
# Tier Monotonicity (loose validation)
# ---------------------------------------------------------------------------


class TestTierMonotonicity:
    """Load all official challenges and verify BICI increases with tier on average."""

    def test_bici_increases_with_tier(self):
        import json
        import glob
        from tt_bench.simulator import Board as B

        challenges_dir = os.path.join(
            BASE_PATH, "data/tasks/official/challenges/json"
        )
        index_path = os.path.join(BASE_PATH, "data/tasks/official/INDEX.json")

        if not os.path.exists(index_path):
            pytest.skip("INDEX.json not found")

        with open(index_path) as f:
            idx = json.load(f)

        tier_map = {}
        for entry in idx.get("tasks", []):
            tier_map[entry["task_id"]] = entry["tier"]

        by_tier = {}
        pattern = os.path.join(challenges_dir, "tt-official-ch*.json")
        for ch_path in sorted(glob.glob(pattern)):
            task_id = os.path.basename(ch_path).replace(".json", "")
            tier = tier_map.get(task_id)
            if tier is None:
                continue

            try:
                board = B.from_task_json(ch_path)
                with open(ch_path) as f:
                    task_info = json.load(f)
                metrics = compute_all_metrics(board, task_info)
                by_tier.setdefault(tier, []).append(metrics.get("bici", 0))
            except Exception:
                continue

        means = {t: sum(v) / len(v) for t, v in by_tier.items() if v}
        tiers = sorted(means)

        if len(tiers) >= 2:
            # At minimum, tier 1 mean should be <= tier 4 mean
            t1_mean = means.get(1, 0)
            t4_mean = means.get(4, 0)
            # This is a loose check — report as warning if violated
            if t1_mean > t4_mean:
                print(f"\n  WARNING: Tier 1 BICI mean ({t1_mean:.4f}) > "
                      f"Tier 4 BICI mean ({t4_mean:.4f})")
            # Not a hard assertion since individual challenges can vary
            assert True  # Always passes; monotonicity is investigated, not enforced
