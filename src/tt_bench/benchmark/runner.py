#!/usr/bin/env python3
"""Turing Tumble Benchmark Runner.

Evaluates LLMs on:
1. Procedural Understanding: Given solution, predict/explain behavior
2. Agentic Synthesis: Iteratively build and verify solutions using tools
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tt_bench.simulator.targets import validate_targets
from tt_bench.llm import client as llm_client_
from tt_bench.tools import executor as tool_executor_

from tt_bench import simulator as tt_sim
from tt_bench.analytics import metrics as complexity_metrics
from tt_bench.benchmark.prompts import (
    AGENTIC_PROMPT_TEMPLATE,
    AGENTIC_SYSTEM_PROMPT,
    AGENTIC_SYSTEM_PROMPTS,
    COMPONENT_RULES,
    UNDERSTANDING_PROMPT_TEMPLATE,
    UNDERSTANDING_SYSTEM_PROMPT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TaskResult:
    """Result of running a single task."""

    task_id: str
    task_type: str  # "understanding" or "agentic_synthesis"
    success: Optional[bool]
    llm_response: str
    predicted: Dict[str, Any]
    expected: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: int = 0
    tokens_used: int = 0
    component_score: Optional[float] = None
    """Jaccard component-placement accuracy against ground-truth (0--1).
    Only populated for agentic_synthesis tasks."""
    logprobs: Any = None
    """Token-level log probabilities from the LLM.  For understanding tasks
    this is a single list of per-token logprob dicts.  For agentic tasks it is
    a list of per-turn logprob lists.  ``None`` when not captured."""


@dataclass
class BenchmarkReport:
    """Aggregate benchmark results."""

    timestamp: str
    model: str
    provider: str
    total_tasks: int
    successful: int
    failed: int
    task_results: List[TaskResult]
    per_tier: Dict[int, Dict[str, int]] = field(default_factory=dict)
    error: Optional[str] = None


# ============================================================================

# ============================================================================
# Benchmark Runner
# ============================================================================


class TuringTumbleBenchmark:
    """Main benchmark runner."""

    def __init__(
        self,
        llm_client: llm_client_.LLMClient,
        challenges_dir: Path,
        output_dir: Path,
        print_board: bool = False,
        max_turns: int = 25,
        max_tokens: int = 32768,
        compute_complexity: bool = False,
        declare_zero_parts: bool = False,
        observable_inventory: bool = False,
        auto_simulate: bool = False,
        revision_style: str = "off",
        revision_rounds: int = 0,
        prompt_variant: str = "urged",
        executor_factory=None,
    ):
        self.llm = llm_client
        # Ablation switches, both off by default so results stay comparable with
        # earlier runs. See --declare-zero-parts and --observable-inventory.
        self.declare_zero_parts = declare_zero_parts
        self.observable_inventory = observable_inventory
        # Harness-ablation arms (HARNESS_ABLATION_PLAN.md). All default to the
        # baseline, which is pinned by tests/test_harness_arms.py.
        self.auto_simulate = auto_simulate
        self.revision_style = revision_style
        self.revision_rounds = revision_rounds
        self.prompt_variant = prompt_variant
        self.executor_factory = executor_factory or tool_executor_.create_executor_from_task

        if prompt_variant not in AGENTIC_SYSTEM_PROMPTS:
            raise ValueError(
                f"Unknown prompt_variant {prompt_variant!r}; "
                f"expected one of {sorted(AGENTIC_SYSTEM_PROMPTS)}"
            )
        if revision_style not in ("off", "retry", "structured"):
            raise ValueError(
                f"Unknown revision_style {revision_style!r}; "
                "expected off, retry or structured"
            )
        # The revision hook lives in VLLMClient.generate_with_tools. Every other
        # client absorbs the keyword into **kwargs and ignores it, which would
        # run a revision arm as the baseline and report it under the arm's
        # label. Fail here rather than after a 12-hour job.
        if revision_style != "off" and not isinstance(self.llm, llm_client_.VLLMClient):
            raise ValueError(
                f"revision_style={revision_style!r} needs the vllm provider; "
                f"{type(self.llm).__name__} ignores the revision hook and would "
                "silently produce baseline results under an arm label."
            )
        self.challenges_dir = challenges_dir
        self.output_dir = output_dir
        self.print_board = print_board
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.compute_complexity = compute_complexity
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._workers: int = 1

        # Results storage
        self.results: List[TaskResult] = []

        # Load questions from the questions folder
        self.questions_dir = challenges_dir.parent.parent / "questions"
        self._questions_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _run_single_challenge(
        self, task_path: Path, task_types: List[str],
    ) -> List[TaskResult]:
        """Run all task types for a single challenge file (thread-safe)."""
        chunk: List[TaskResult] = []
        if "understanding" in task_types:
            chunk.extend(self.run_understanding_task(task_path))
        if "agentic_synthesis" in task_types:
            chunk.append(self.run_agentic_task(task_path))
        return chunk

    def load_questions(self, task_id: str) -> List[Dict[str, Any]]:
        """Load questions from the questions JSON file for a task."""
        if task_id in self._questions_cache:
            return self._questions_cache[task_id]

        questions_file = self.questions_dir / f"{task_id}_questions.json"
        if not questions_file.exists():
            return []

        try:
            with open(questions_file) as f:
                data = json.load(f)
            questions = data.get("questions", [])
            self._questions_cache[task_id] = questions
            return questions
        except Exception as e:
            logger.warning(f"Failed to load questions from {questions_file}: {e}")
            return []

    def load_task(self, task_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load a task from JSON file."""
        with open(task_path) as f:
            data = json.load(f)

        task_id = data.get("task_id", task_path.stem)
        if task_path.stem.startswith("tt-official-") and task_id != task_path.stem:
            # Some practice-variant JSON files inherited the base challenge's
            # task_id. Use the unique filename stem for official tasks so
            # reports do not collapse variants into duplicate IDs.
            task_id = task_path.stem

        task_info = {
            "task_id": task_id,
            "tier": data.get("tier", 1),
            "objective": data.get("objective", ""),
            "board": data.get("board", {}),
            "available_parts": data.get("available_parts", {}),
            "solution": data.get("solution", {}),
            "expected_output": data.get("expected_output", {}),
            "required_output": data.get("required_output"),
            "trials": data.get("trials"),
            "registers": data.get("registers"),
            "input_sequence": data.get(
                "input_sequence", ["blue"]
            ),
        }

        return task_info, data

    @staticmethod
    def _task_hopper_entry_mode(board_data: Dict[str, Any]) -> str:
        """Return the hopper-entry convention for this task.

        Official challenge JSONs use ``inward`` mode: a marble from the blue
        hopper at column x enters one column to the right (x+1), and a red
        marble enters one column to the left (x-1).  This matches the
        simulator's ``from_task_json`` so prompts, tool simulations, and
        scoring stay aligned.
        """
        return board_data.get("hopper_entry_mode", "inward")

    @staticmethod
    def _normalize_input_sequence(input_seq: Any) -> List[str]:
        """Normalize input_sequence from JSON/string into a list of side names."""
        if isinstance(input_seq, str):
            return [s.strip() for s in input_seq.split(",") if s.strip()]
        if isinstance(input_seq, list):
            return [str(s).strip() for s in input_seq if str(s).strip()]
        return ["blue"]

    @staticmethod
    def _normalize_placement(placement: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize direct and agentic placement shapes to simulator shape."""
        component_type = (
            placement.get("component")
            or placement.get("component_type")
            or placement.get("type")
        )
        x = placement.get("x", placement.get("col"))
        y = placement.get("y", placement.get("row"))
        if component_type is None or x is None or y is None:
            raise ValueError(f"Invalid placement shape: {placement}")

        normalized = {"type": component_type, "x": int(x), "y": int(y)}
        if "state" in placement:
            normalized["state"] = placement["state"]
        return normalized

    def _normalize_placements(self, placements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._normalize_placement(p) for p in placements]

    @staticmethod
    def _compute_component_score(
        placements: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
    ) -> Tuple[float, int, int, int]:
        """Jaccard-like component-placement accuracy against ground truth.

        A placement *matches* when (x, y, type, state) are identical.
        The score is :math:`|correct| / |placements \\cup ground\\ truth|`
        — penalising both missing ground-truth components and extra, wrong
        placements equally.  Returns 1.0 when both lists are empty (vacuously
        correct).

        Returns
        -------
        score : float  0--1
        correct : int  number of placements matching ground truth
        placed_count : int  |placements|
        gt_count : int  |ground_truth|
        """
        def _key(comp: Dict[str, Any]) -> tuple:
            return (
                int(comp.get("x", comp.get("col", 0))),
                int(comp.get("y", comp.get("row", 0))),
                str(comp.get("type", comp.get("component_type", ""))),
                int(comp.get("state", 0)),
            )

        placed_keys = {_key(p) for p in placements}
        gt_keys = {_key(g) for g in ground_truth}

        correct = len(placed_keys & gt_keys)
        union = len(placed_keys | gt_keys)

        if union == 0:
            return 1.0, 0, 0, 0

        score = correct / union
        return score, correct, len(placements), len(ground_truth)

    def _build_board(
        self,
        task_info: Dict[str, Any],
        *,
        include_solution: bool = False,
        placements: Optional[List[Dict[str, Any]]] = None,
    ) -> tt_sim.Board:
        """Build a simulator board from task JSON and optional placements."""
        board_data = task_info["board"]
        blue_h = board_data.get("ball_hoppers", {}).get("blue", {})
        red_h = board_data.get("ball_hoppers", {}).get("red", {})
        levers = board_data.get("trigger_levers", {})
        board = tt_sim.Board(
            rows=board_data.get("height", 11),
            cols=board_data.get("width", 11),
            blue_hopper_x=blue_h.get("x", 2),
            red_hopper_x=red_h.get("x", 8),
            blue_hopper_count=blue_h.get("count", 8),
            red_hopper_count=red_h.get("count", 8),
            hopper_entry_mode=self._task_hopper_entry_mode(board_data),
            left_catcher_x=levers.get("left", {}).get("x"),
            right_catcher_x=levers.get("right", {}).get("x"),
        )
        board.editable_bit_states = {tuple(p) for p in board_data.get('editable_bit_states', [])}
        for comp_dict in board_data.get("fixed_components", []):
            comp = tt_sim.Component.from_dict(comp_dict)
            board.place(comp.x, comp.y, comp)
        if include_solution:
            for comp_dict in task_info.get("solution", {}).get("placed_components", []):
                comp = tt_sim.Component.from_dict(comp_dict)
                board.place_solution_component(comp)
        for comp_dict in placements or []:
            comp = tt_sim.Component.from_dict(self._normalize_placement(comp_dict))
            board.place_solution_component(comp)
        tt_sim.build_gear_connections(board)
        return board

    def _board_for_prompt(
        self, task_info: Dict[str, Any], *, include_solution: bool
    ) -> tt_sim.Board:
        """Build a Board reflecting exactly what the LLM should reason over.

        For agentic tasks we show only ``fixed_components``; for
        understanding we also place the reference solution.
        """
        return self._build_board(task_info, include_solution=include_solution)

    @staticmethod
    def _format_board_json(board: tt_sim.Board) -> str:
        return json.dumps(board.to_llm_dict(), indent=2)

    @staticmethod
    def _format_available_parts(avail: Dict[str, int], declare_zero: bool = False) -> str:
        """Render the inventory for the prompt.

        ABLATION (opt-in via declare_zero): by default types with a count of 0
        are omitted, so the agent must infer unavailability from ABSENCE. Language
        models handle explicit negation far better than omission, and the most
        common rejected action observed is placing a type whose count is zero.
        Listing them explicitly separates "did not know" from "knew and ignored".
        """
        if declare_zero:
            lines = [f"  - {part}: {count}" for part, count in avail.items()]
        else:
            lines = [f"  - {part}: {count}" for part, count in avail.items() if count > 0]
        if 'ramp' in avail:
            lines.append('  ramp is a shared pool: ramp_left and ramp_right each consume one ramp.')
        return "\n".join(lines) if lines else "  (none)"

    def _print_board(
        self,
        task_info: Dict[str, Any],
        *,
        include_solution: bool,
        task_type: str,
    ) -> None:
        """Print an ASCII board snapshot for the current task when enabled."""
        if not self.print_board:
            return

        board = self._board_for_prompt(task_info, include_solution=include_solution)
        print("\n" + "=" * 70)
        print(f"Task: {task_info['task_id']} | Type: {task_type}")
        print("=" * 70)
        print(board.render())

    def build_understanding_prompt(
        self,
        task_info: Dict[str, Any],
        question_type: str,
        question: str,
        answer_format: str,
    ) -> str:
        """Build an understanding prompt from task info."""
        board = self._board_for_prompt(task_info, include_solution=True)
        return UNDERSTANDING_PROMPT_TEMPLATE.format(
            board_json=self._format_board_json(board),
            COMPONENT_RULES=COMPONENT_RULES,
            question_type=question_type,
            question=question,
            answer_format=answer_format,
        )

    def build_agentic_prompt(self, task_info: Dict[str, Any]) -> str:
        """Build an agentic synthesis prompt with tools."""
        board = self._board_for_prompt(task_info, include_solution=False)
        state_hint = ''
        if board.editable_bit_states:
            state_hint = ('\nChoose initial states for the fixed bits listed in editable_bit_states. '
                          'Call place_component with the same bit type and coordinates to set its state; '
                          'this consumes no part and does not move the fixed bit.')
        return AGENTIC_PROMPT_TEMPLATE.format(
            board_json=self._format_board_json(board),
            available_parts=self._format_available_parts(
                task_info["available_parts"], getattr(self, "declare_zero_parts", False)
            ),
            target_behavior=task_info["objective"] + state_hint + self._format_trials(task_info),
            COMPONENT_RULES=COMPONENT_RULES,
        )

    @staticmethod
    def _format_trials(task_info: Dict[str, Any]) -> str:
        """Render the trial table that a trial-scored goal is judged on.

        The objective sentence alone cannot state it. "Count the blue balls in
        register A" is scored over several runs with different ball counts, and
        "reverse each bit regardless of how it starts" over several starting
        configurations -- so the model has to be told which runs it will be
        judged on, or the task is unanswerable rather than hard.
        """
        trials = task_info.get("trials") or []
        if not trials:
            return ""

        def direction(state: int) -> str:
            return "right" if state else "left"

        lines = [
            "",
            "",
            "This task is scored over several trials. Each one runs on a fresh board "
            "with the hoppers loaded and the bits pointed as stated, and every trial "
            "must pass. A board that trips its own trigger lever keeps running until "
            "its hopper is empty, so the load sets how many balls a trial delivers.",
        ]
        for name, bits in (task_info.get("registers") or {}).items():
            lines.append(
                f"Register {name} is read from {', '.join(bits)} "
                f"as a binary number, most significant bit first."
            )
        lines.append("")

        for index, trial in enumerate(trials):
            given = []
            hoppers = trial.get("hoppers")
            if hoppers:
                given.append(
                    "load " + ", ".join(f"{c} x{n}" for c, n in sorted(hoppers.items()))
                )
            sequence = trial.get("input_sequence", task_info.get("input_sequence"))
            if sequence:
                counts = Counter(sequence)
                given.append(
                    "release " + ", ".join(f"{n} x{c}" for n, c in sorted(counts.items()))
                )
            initial = trial.get("initial_bit_states") or {}
            if initial:
                given.append(
                    "starting "
                    + ", ".join(f"{k} {direction(v)}" for k, v in sorted(initial.items()))
                )

            expect = trial.get("expect") or {}
            wanted = [
                f"register {n} = {v}" for n, v in (expect.get("registers") or {}).items()
            ]
            wanted += [
                f"{k} points {direction(v)}"
                for k, v in (expect.get("final_bit_states") or {}).items()
            ]
            wanted += [
                f"{f} = {expect[f]}"
                for f in ("left_catcher", "right_catcher", "intercepted")
                if f in expect
            ]
            if expect.get("intercepted_at"):
                x, y = expect["intercepted_at"]
                wanted.append(f"ball caught by the interceptor at ({x},{y})")
            if expect.get('intercepted_colours'):
                wanted.append('intercepted colours: ' + ', '.join(expect['intercepted_colours']))
            if expect.get("required_output"):
                wanted.append("output " + ", ".join(expect["required_output"]))
            if expect.get("final_marble_state"):
                wanted.append("catchers " + ", ".join(expect["final_marble_state"]))

            lines.append(
                f"  trial {trial.get('name', index)}: "
                f"{'; '.join(given)} -> {'; '.join(wanted)}"
            )
        return "\n".join(lines)

    def _validate_available_parts(
        self,
        task_info: Dict[str, Any],
        placements: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Ensure predicted placements do not exceed the available inventory."""
        available = task_info.get("available_parts", {}) or {}

        # Some legacy official encodings have declared inventories that do not
        # cover their own reference solution (for example practice variants
        # where ramp orientation counts were entered incorrectly). In that case,
        # inventory cannot be used as a reliable hard-fail criterion for this
        # task; functional legality remains authoritative.
        reference = self._normalize_placements(
            task_info.get("solution", {}).get("placed_components", [])
        )
        from tt_bench.simulator.inventory import used_inventory
        reference_used = used_inventory(reference, available, task_info.get('board'))
        reference_exceeds_inventory = any(
            count > available.get(component_type, 0)
            for component_type, count in reference_used.items()
        )
        if reference_exceeds_inventory:
            return True, "Inventory check skipped: reference solution exceeds declared inventory"

        used = used_inventory(placements, available, task_info.get('board'))
        for component_type, count in used.items():
            allowed = available.get(component_type, 0)
            if count > allowed:
                return (
                    False,
                    f"Used {count} {component_type} part(s), but only {allowed} available",
                )
        return True, "Parts inventory respected"

    @staticmethod
    def _caught_colour_sequence(results: List[tt_sim.MarbleResult]) -> List[str]:
        """Map catcher hits to the benchmark's final marble sequence.

        An interceptor is a terminal position like the catchers, so it has to
        appear in the sequence: ground truth records it as "intercepted", and
        dropping it here would make every interceptor task unsolvable.
        """
        colours: List[str] = []
        for result in results:
            if result.caught_by == "left_catcher":
                colours.append("blue")
            elif result.caught_by == "right_catcher":
                colours.append("red")
            elif result.caught_by and "interceptor" in str(result.caught_by):
                colours.append("intercepted")
        return colours

    @classmethod
    def _detect_free_fall(
        cls, board: tt_sim.Board, results: List[tt_sim.MarbleResult]
    ) -> Tuple[bool, str]:
        """Detect illegal in-board movement through empty cells.

        The simulator can physically continue a marble through empty cells, but
        Turing Tumble puzzle solutions are only legal when a marble lands on a
        component at every in-board step after it enters from the hopper.

        Reports the first offending cell only, which is what scoring needs.
        ``_free_fall_cells`` shares the predicate and returns all of them, so the
        revision critique cannot disagree with the rejection about what is legal.
        """
        cells = cls._free_fall_cells(board, results)
        if cells:
            marble_idx, cell = cells[0]
            return True, f"marble {marble_idx} traversed empty cell {cell}"
        return False, ""

    @staticmethod
    def _free_fall_cells(
        board: tt_sim.Board, results: List[tt_sim.MarbleResult]
    ) -> List[Tuple[int, Tuple[int, int]]]:
        """Every (marble index, cell) pair that traverses an empty in-board cell."""
        offending: List[Tuple[int, Tuple[int, int]]] = []
        for marble_idx, result in enumerate(results, start=1):
            path = result.path or []
            for path_idx, curr in enumerate(path[1:], start=1):
                prev = path[path_idx - 1]
                x, y = curr

                # The hopper-to-board transition may enter an empty coordinate;
                # subsequent in-board motion may not.
                if prev[1] < 0 and y >= 0:
                    continue

                next_pos = path[path_idx + 1] if path_idx + 1 < len(path) else None
                if (
                    y == board.rows - 1
                    and next_pos is not None
                    and next_pos[1] >= board.rows
                    and board.catcher_at(x) is not None
                ):
                    # The final coordinate just above a trigger lever is a
                    # catcher approach slot in several official encodings, not
                    # an illegal mid-board gap.
                    continue

                if 0 <= x < board.cols and 0 <= y < board.rows and curr not in board.components:
                    offending.append((marble_idx, tuple(curr)))
        return offending

    def _validate_simulation_results(
        self,
        board: tt_sim.Board,
        task_info: Dict[str, Any],
        results: List[tt_sim.MarbleResult],
    ) -> Tuple[bool, str]:
        """Validate a completed simulator run against task ground truth."""
        has_free_fall, free_fall_msg = self._detect_free_fall(board, results)
        if has_free_fall:
            return False, f"Illegal free fall: {free_fall_msg}"

        lost = [
            r
            for r in results
            if r.caught_by is None
            and r.termination_reason not in ("no_blue_balls", "no_red_balls")
        ]
        if lost:
            reasons = Counter(r.termination_reason or "unknown" for r in lost)
            summary = ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items()))
            return False, f"{len(lost)} marble(s) did not reach a valid catcher ({summary})"

        return validate_targets(task_info, board, results)

    def validate_synthesis(
        self,
        task_info: Dict[str, Any],
        placements: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Validate a synthesis solution by running it in the simulator."""
        try:
            normalized = self._normalize_placements(placements)
            inventory_ok, inventory_msg = self._validate_available_parts(task_info, normalized)
            if not inventory_ok:
                return False, inventory_msg

            board = self._build_board(task_info, placements=normalized)
            return self._validate_against_expected(
                board,
                task_info.get("expected_output", {}),
                task_info,
            )

        except Exception as e:
            return False, f"Validation error: {e}"

    # Harness arm `rev-retry`: the compute-matched control for rev-structured.
    # It continues the loop on exactly the same trigger but says nothing
    # task-specific, so rev-structured minus rev-retry isolates the grounded
    # content of the critique from the mere fact of not stopping.
    REVISION_RETRY_MESSAGE = (
        "Your submitted solution was rejected. Try again."
    )

    # A 15x15 board with 8 marbles can free-fall through dozens of cells; naming
    # all of them would crowd out the rest of the critique.
    REVISION_MAX_FREE_FALL_CELLS = 12

    def build_revision_critique(
        self,
        task_info: Dict[str, Any],
        placements: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Simulator-grounded diagnosis of a rejected solution, or None if valid.

        Used by the `rev-structured` arm. The verdict and the headline reason come
        from ``validate_synthesis`` itself rather than from a parallel set of
        checks: a critique that described a different rule from the one that
        rejected the answer would be feeding the model misinformation, and the
        arm would measure that instead of structured revision.

        ``placements`` must be the executor's board (``get_placed_components()``).
        ``VLLMClient`` returns the final answer as ``{"content": "<raw text>"}``
        and never parses a ``final_solution`` field, so validating a parsed
        answer field would reject every vLLM submission regardless of content.
        """
        is_valid, reason = self.validate_synthesis(task_info, placements)
        if is_valid:
            return None

        lines = [
            "Your submitted solution was REJECTED by the scorer.",
            f"Reason: {reason}",
        ]

        normalized = self._normalize_placements(placements)
        try:
            board = self._build_board(task_info, placements=normalized)
            input_seq = self._normalize_input_sequence(
                task_info.get("input_sequence", ["blue"])
            )
            results = board.run(input_seq)
        except Exception as e:
            lines.append(f"The board could not be simulated: {e}")
            return "\n".join(lines)

        # Group by unique cell, as the run_simulation tool already does: eight
        # marbles down the same path are one gap to fill, not eight, and listing
        # the repeat crowds out the rest of the critique.
        by_cell: Dict[Tuple[int, int], int] = {}
        for _marble, cell in self._free_fall_cells(board, results):
            by_cell[cell] = by_cell.get(cell, 0) + 1
        if by_cell:
            ordered = sorted(by_cell.items())
            shown = ordered[: self.REVISION_MAX_FREE_FALL_CELLS]
            listed = ", ".join(
                f"{cell} (traversed by {count} marble(s))" for cell, count in shown
            )
            omitted = len(ordered) - len(shown)
            lines.append("")
            lines.append(
                f"Empty cells a marble traversed ({len(ordered)} distinct; every one "
                f"must hold a component): {listed}"
                + (f", and {omitted} more" if omitted else "")
            )

        counts = Counter(r.caught_by for r in results)
        lines.append("")
        lines.append("Observed for input_sequence " + json.dumps(input_seq) + ":")
        lines.append(
            f"  catchers: left={counts['left_catcher']} right={counts['right_catcher']} "
            f"interceptor={counts['interceptor']}"
        )
        lines.append(
            "  catcher sequence: "
            + json.dumps(self._caught_colour_sequence(results))
        )
        lines.append("  final bit states: " + json.dumps(board.get_all_states()))

        declared = {
            "solution.final_marble_state": task_info.get("solution", {}).get(
                "final_marble_state"
            ),
            "required_output": task_info.get("required_output"),
            "expected_output": task_info.get("expected_output") or None,
        }
        stated = {k: v for k, v in declared.items() if v}
        if stated:
            lines.append("")
            lines.append("Declared targets:")
            for key, value in stated.items():
                lines.append(f"  {key}: {json.dumps(value)}")

        available = task_info.get("available_parts", {}) or {}
        if available:
            from tt_bench.simulator.inventory import used_inventory
            used = used_inventory(normalized, available, task_info.get('board'))
            remaining = {k: available[k] - used.get(k, 0) for k in sorted(available)}
            lines.append("")
            lines.append("Remaining inventory: " + json.dumps(remaining))

        lines.append("")
        lines.append(
            "Components you have placed: "
            + (json.dumps(normalized) if normalized else "none")
        )
        lines.append("")
        lines.append(
            "Fix the board with the tools, verify with run_simulation, and submit "
            "again. Do not repeat the rejected solution."
        )
        return "\n".join(lines)

    def _build_revision_hook(
        self,
        task_info: Dict[str, Any],
        executor: "tool_executor_.TuringTumbleToolExecutor",
        is_unsolvable: bool,
    ) -> Tuple[Optional[Any], List[int]]:
        """(callback, mutable round counter) for the revision arms, or (None, [0]).

        Both arms fire on exactly the same trigger — a submitted answer that
        ``build_revision_critique`` rejects — and differ only in the message.
        That is what makes rev-structured minus rev-retry a measurement of the
        critique's content rather than of when the loop continues.
        """
        counter = [0]
        if self.revision_style == "off" or is_unsolvable:
            # Undefined on unsolvable variants: there the correct answer is a
            # refusal, and "fix the board and submit again" argues against it.
            return None, counter

        def hook(content: str) -> Optional[str]:
            # A declared unsolvability is a different answer type, not a
            # rejected board; scoring handles it separately.
            if self._model_declares_unsolvable({"content": content}):
                return None
            critique = self.build_revision_critique(
                task_info, executor.get_placed_components()
            )
            if critique is None:
                return None
            counter[0] += 1
            if self.revision_style == "structured":
                return critique
            return self.REVISION_RETRY_MESSAGE

        return hook, counter

    def _validate_against_expected(
        self, board: tt_sim.Board, expected: Dict[str, Any], task_info: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate using explicit expected_output declaration."""
        # ``expected`` is retained for API compatibility. All declarations in
        # task_info are checked together by the shared target contract.
        input_seq = self._normalize_input_sequence(task_info.get("input_sequence", ["blue"]))
        results = board.run(input_seq)
        return self._validate_simulation_results(board, task_info, results)

    @staticmethod
    def _model_declares_unsolvable(final_result: Any) -> bool:
        """Check whether the model declared the task unsolvable.

        Parses the ``final_result`` dict returned by ``generate_with_tools``
        for a ``success`` field set to ``False``.  Handles both flat dicts
        (most models) and ``{"content": "<json>"}`` wrappers (some providers).
        """
        if not isinstance(final_result, dict):
            return False
        for candidate in (final_result,):
            if "success" in candidate:
                return candidate.get("success") is False
        content = final_result.get("content", "")
        if isinstance(content, str) and content.strip():
            try:
                inner = json.loads(content)
                if isinstance(inner, dict) and "success" in inner:
                    return inner.get("success") is False
            except (json.JSONDecodeError, TypeError):
                pass
        return False

    def run_understanding_task(self, task_path: Path) -> List[TaskResult]:
        """Run procedural understanding tasks for a challenge.

        Generates multiple question types:
        - execution_trace: Predict state after N marbles
        - component_role: Explain what a component does
        - counterfactual: Predict behavior if something changed
        - abstraction: Describe the overall computation
        """
        results = []
        start_time = time.time()

        try:
            task_info, _ = self.load_task(task_path)
            task_id = task_info["task_id"]

            self._print_board(
                task_info,
                include_solution=True,
                task_type="understanding",
            )

            # Build the board with solution
            board_data = task_info["board"]
            solution = task_info.get("solution", {}).get("placed_components", [])
            board = self._build_board(task_info, include_solution=True)

            # Compute complexity metrics if requested
            cx_metrics: Dict[str, float] = {}
            if self.compute_complexity:
                try:
                    cx_metrics = complexity_metrics.compute_all_metrics(board, task_info)
                except Exception as e:
                    logger.warning(f"Complexity metrics failed for {task_id}: {e}")

            # Get component list for prompts
            all_components = board_data.get("fixed_components", []) + solution

            # Load questions from the questions folder
            questions = self.load_questions(task_id)

            if not questions:
                logger.warning(f"No questions found for {task_id}, skipping understanding tasks")
                return results

            logger.info(f"Running understanding tasks for: {task_id} ({len(questions)} questions)")

            for question_index, q in enumerate(questions, start=1):
                qid, q_type, question, expected_answer = self._normalise_question(
                    q, question_index
                )

                if not question:
                    continue

                try:
                    # Build prompt
                    prompt = self.build_understanding_prompt(
                        task_info=task_info,
                        question_type=q_type,
                        question=question,
                        answer_format=self._get_answer_format(q_type),
                    )

                    # Query LLM
                    predicted, error, usage, logprobs = self.llm.generate_json(
                        prompt=prompt,
                        system_prompt=UNDERSTANDING_SYSTEM_PROMPT,
                    )
                    total_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

                    # Validate by running actual simulation or comparing to expected answer
                    predicted = predicted or {}
                    if not isinstance(predicted, dict):
                        predicted = {"answer": predicted}
                    validation_result = self._validate_understanding(
                        board, q_type, question, predicted, expected_answer=expected_answer
                    )

                    results.append(
                        TaskResult(
                            task_id=f"{task_id}_{qid}",
                            task_type="understanding",
                            success=validation_result["correct"],
                            llm_response=str(predicted.get("answer", "")),
                            predicted=predicted,
                            expected=validation_result["expected"],
                            metrics={
                                "trace_accuracy": validation_result.get(
                                    "trace_accuracy", 0.0
                                ),
                                "state_precision": validation_result.get(
                                    "state_precision", 0.0
                                ),
                                **cx_metrics,
                            },
                            error=validation_result.get("error", error),
                            latency_ms=int((time.time() - start_time) * 1000),
                            tokens_used=total_tokens,
                            logprobs=logprobs,
                        )
                    )

                except Exception as e:
                    logger.warning(f"Error in understanding task {q_type}: {e}")
                    results.append(
                        TaskResult(
                            task_id=f"{task_id}_{q_type}",
                            task_type="understanding",
                            success=False,
                            llm_response="",
                            predicted={},
                            expected={},
                            error=str(e),
                            latency_ms=int((time.time() - start_time) * 1000),
                        )
                    )

        except Exception as e:
            logger.exception(f"Error loading understanding task {task_path}")
            results.append(
                TaskResult(
                    task_id=task_path.stem,
                    task_type="understanding",
                    success=False,
                    llm_response="",
                    predicted={},
                    expected={},
                    error=str(e),
                    latency_ms=int((time.time() - start_time) * 1000),
                )
            )

        return results

    @staticmethod
    def _normalise_question(
        question_data: Dict[str, Any], index: int
    ) -> Tuple[str, str, str, Any]:
        """Normalize all question schemas present in the official corpus."""
        qid = (
            question_data.get("qid")
            or question_data.get("question_id")
            or question_data.get("id")
            or f"q{index}"
        )
        question_type = question_data.get("type") or "unknown"
        question = question_data.get("question", "")
        expected = question_data.get(
            "answer", question_data.get("expected_answer", "")
        )
        return str(qid), str(question_type), str(question), expected

    def run_agentic_task(self, task_path: Path) -> TaskResult:
        """Run an agentic synthesis task using function calling.

        The LLM uses tools to iteratively build and verify a solution.
        """
        start_time = time.time()

        try:
            task_info, raw_data = self.load_task(task_path)
            task_id = task_info["task_id"]
            board_data = task_info["board"]

            self._print_board(
                task_info,
                include_solution=False,
                task_type="agentic_synthesis",
            )

            logger.info(f"Running agentic synthesis task: {task_id}")

            # Compute complexity metrics if requested (on initial board, pre-solution)
            cx_metrics: Dict[str, float] = {}
            if self.compute_complexity:
                try:
                    init_board = self._build_board(task_info, include_solution=False)
                    cx_metrics = complexity_metrics.compute_all_metrics(init_board, task_info)
                except Exception as e:
                    logger.warning(f"Complexity metrics failed for {task_id}: {e}")

            # Create tool executor with fixed components
            fixed = board_data.get("fixed_components", [])
            available_parts = task_info.get("available_parts", {})
            executor = self.executor_factory(
                board_data,
                fixed,
                available_parts=available_parts,
                target_sequence=self._normalize_input_sequence(
                    task_info.get("input_sequence", ["blue"])
                ),
                expose_inventory=getattr(self, "observable_inventory", False),
                auto_simulate=self.auto_simulate,
                target_final_state=task_info.get("solution", {}).get(
                    "final_marble_state"
                ),
                expected_output=task_info.get("expected_output", {}),
                required_output=task_info.get("required_output"),
                trials=task_info.get("trials"),
                registers=task_info.get("registers"),
            )

            # Build prompt
            prompt = self.build_agentic_prompt(task_info)

            # ── Determine if this is an unsolvable variant ──────────────────
            # Read before the loop: the revision arms must not badger a model
            # off a correct unsolvability declaration.
            task_meta = raw_data.get("_meta", {}) if isinstance(raw_data, dict) else {}
            is_unsolvable = task_meta.get("variant_type") == "unsolvable"

            revision_hook, revision_counter = self._build_revision_hook(
                task_info, executor, is_unsolvable
            )

            # Run agent with tools
            final_result, error, tool_calls, tool_results, usage, turn_logprobs = self.llm.generate_with_tools(
                    prompt=prompt,
                    tools=llm_client_.turing_tumble_tools(
                        board_data.get("height", 11),
                        board_data.get("width", 11),
                    ),
                    tool_executor=executor,
                    system_prompt=AGENTIC_SYSTEM_PROMPTS[self.prompt_variant],
                    max_turns=self.max_turns,
                    max_tokens=self.max_tokens,
                    **(
                        {
                            "on_final_answer": revision_hook,
                            "revision_rounds": self.revision_rounds,
                        }
                        if revision_hook is not None
                        else {}
                    ),
                )

            placed = executor.get_placed_components()
            solution_used = placed
            unsolvable_detected: Optional[bool] = None
            # How a success was CREDITED (plan section 4.4). Only the first two
            # mean the agent submitted a solution it believed in; `ceiling-valid`
            # means the budget expired on a board that happened to validate, and
            # `fallback` that the agent built a correct board and then removed
            # it. An arm whose gain sits in the last two shifted scoring credit
            # rather than capability, and cannot be inferred after the fact —
            # the per-task record does not keep the executor's final board
            # alongside the credited one.
            credit_route: Optional[str] = None

            if is_unsolvable:
                # For unsolvable tasks, success = model correctly declares
                # unsolvability rather than trying to build an impossible
                # solution.  validate_synthesis always returns False for
                # these boards, so the normal scoring path cannot reward a
                # correct answer.
                declared_unsolvable = self._model_declares_unsolvable(final_result)
                if declared_unsolvable:
                    is_valid = True
                    msg = "Correctly identified task as unsolvable"
                    credit_route = "unsolvable-declared"
                else:
                    is_valid = False
                    msg = "Model did not recognize task as unsolvable"
                unsolvable_detected = declared_unsolvable
            else:
                # ── Normal validation for well-posed tasks ────────────────
                is_valid, msg = False, "No solution found"

                # Validate if the LLM submitted a final_answer *or* if it
                # placed components before running out of turns.  This catches
                # the common case where a model finds the right board but
                # exhausts its turn budget before emitting a final_solution.
                if final_result or placed:
                    is_valid, msg = self.validate_synthesis(task_info, placed)
                    if is_valid:
                        if isinstance(final_result, dict) and final_result.get(
                            "solution_found"
                        ):
                            credit_route = "early-stop"
                        elif final_result is not None:
                            credit_route = "submitted"
                        else:
                            credit_route = "ceiling-valid"

                # Fall back to the best board state recorded during successful
                # simulation runs — handles the case where the LLM places a
                # correct component, verifies it, then removes it.
                if not is_valid:
                    best = executor.get_best_placement()
                    if best and best != placed:
                        is_valid, msg = self.validate_synthesis(task_info, best)
                        if is_valid:
                            solution_used = best
                            credit_route = "fallback"

            # Compute component-level accuracy against ground truth.
            # For unsolvable tasks where the model correctly refused, the
            # component score is not meaningful: the right answer is to place
            # nothing, so every score would read 0.0 regardless of correctness.
            gt_placements = task_info.get("solution", {}).get("placed_components", [])
            if is_unsolvable and unsolvable_detected:
                comp_score = None
                comp_correct = 0
                comp_placed = 0
                comp_gt = len(gt_placements)
            else:
                comp_score, comp_correct, comp_placed, comp_gt = self._compute_component_score(
                    solution_used, gt_placements
                )

            transcript = []
            for tc, tr in zip(tool_calls, tool_results):
                transcript.append(
                    {
                        "turn": getattr(tc, "turn_index", 0),
                        "assistant_text": getattr(tc, "assistant_text", "") or "",
                        "tool_name": tc.name,
                        "arguments": tc.arguments,
                        "result": tr.result,
                        "error": tr.error,
                    }
                )

            return TaskResult(
                task_id=task_id,
                task_type="agentic_synthesis",
                success=is_valid,
                llm_response=json.dumps(final_result) if final_result is not None else "",
                predicted={
                    "final_solution": solution_used,
                    "tool_calls": [
                        {"name": tc.name, "args": tc.arguments} for tc in tool_calls
                    ],
                    "transcript": transcript,
                },
                expected={
                    "solution": task_info.get("solution", {}),
                    "_meta": task_meta,
                },
                metrics={
                    "valid": float(is_valid),
                    "tool_calls_count": len(tool_calls),
                    # `turns` was len(tool_calls), which is not turns. A model
                    # emitting several tool calls in one assistant message inflates
                    # it past max_turns — DeepSeek recorded 444 against a budget of
                    # 25, while models emitting one call per turn happened to match
                    # and made the field look correct. turn_logprobs carries one
                    # entry per API call (see client.py "turn_logprobs is a list of
                    # per-turn logprob lists"), i.e. one per agent-loop iteration,
                    # and every client populates it. Fall back to the old value
                    # rather than reporting 0, because 0 turns is the signature the
                    # analysis tooling uses for "this run generated nothing".
                    "turns": len(turn_logprobs) if turn_logprobs else len(tool_calls),
                    "turns_source": "api_calls" if turn_logprobs else "tool_calls",
                    # Rejected answers fed back to the model. 0 under the
                    # baseline; under a revision arm it separates tasks the arm
                    # actually touched from those it never fired on.
                    "revisions": revision_counter[0],
                    "credit_route": credit_route,
                    "component_score": comp_score,
                    "component_correct": comp_correct,
                    "component_placed": comp_placed,
                    "component_gt": comp_gt,
                    "unsolvable_detected": float(unsolvable_detected) if unsolvable_detected is not None else None,
                    **cx_metrics,
                },
                error=msg if not is_valid else error,
                latency_ms=int((time.time() - start_time) * 1000),
                tokens_used=usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
                component_score=comp_score,
                logprobs=turn_logprobs,
            )

        except Exception as e:
            logger.exception(f"Error in agentic task {task_path}")
            return TaskResult(
                task_id=task_path.stem,
                task_type="agentic_synthesis",
                success=False,
                llm_response="",
                predicted={},
                expected={},
                error=str(e),
                latency_ms=int((time.time() - start_time) * 1000),
            )

    def _find_component_position(self, components: List[Dict]) -> Tuple[int, int]:
        """Find position of first interesting component (bit/gear)."""
        for comp in components:
            comp_type = comp.get("type", "")
            if "bit" in comp_type:
                return comp.get("x", 0), comp.get("y", 0)
        return 3, 3  # Default

    def _get_answer_format(self, question_type: str) -> str:
        """Get expected answer format for a question type."""
        formats = {
            "execution_trace": '{"final_destination": "left_catcher" or "right_catcher", "reasoning": "step by step..."}',
            "ball_path": "Describe the complete path of the first blue ball, listing every component it interacts with in order.",
            "output_sequence": "In what order do balls exit the machine (blue exit, red exit, or intercepted)?",
            "trigger_sequence": "Which lever does the first blue ball trigger and what color ball is released next?",
            "component_count": "Provide a number (e.g., '8 components')",
            "parts_count": "Provide the requested count as a number.",
            "numeric": "Provide the requested value as a number.",
            "calculation": "Provide the calculated value and a brief justification.",
            "select": "Return the selected option exactly.",
            "state_trace": "Report the requested states in order.",
            "state_tracking": "Report the requested states in order.",
            "final_state": "Report the final state.",
            "pattern": "Report the complete requested pattern.",
            "configuration": "Describe the requested configuration precisely.",
            "alternating_mechanism": "Describe the alternating behavior precisely.",
            "logic_analysis": "Give the result and a concise causal explanation.",
            "conceptual": "Give a concise conceptual answer.",
            "concept": "Give a concise conceptual answer.",
            "component_role": "This component [functions as...]",
            "abstraction": "This board performs [computation type]",
        }
        return formats.get(question_type or "", "Provide a clear answer.")

    @staticmethod
    def _prediction_text(predicted: Any) -> str:
        """Extract the answer field without letting reasoning mask a wrong answer."""
        if isinstance(predicted, dict):
            if "answer" in predicted:
                value = predicted["answer"]
                return value if isinstance(value, str) else json.dumps(value)
            values = [
                value for key, value in predicted.items()
                if key not in {"reasoning", "explanation"}
            ]
            return " ".join(
                value if isinstance(value, str) else json.dumps(value)
                for value in values
            )
        return predicted if isinstance(predicted, str) else json.dumps(predicted)

    @staticmethod
    def _normalise_answer_text(value: Any) -> str:
        text = str(value).lower().replace("_", " ")
        return " ".join(re.findall(r"[a-z0-9+-]+", text))

    @classmethod
    def _text_answer_matches(cls, predicted: str, expected: str) -> bool:
        """Compare free-form answers while allowing concise paraphrases."""
        pred_norm = cls._normalise_answer_text(predicted)
        exp_norm = cls._normalise_answer_text(expected)
        if not pred_norm or not exp_norm:
            return False
        if pred_norm == exp_norm or exp_norm in pred_norm:
            return True

        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "in", "is", "it", "of", "on", "the", "then", "to",
            "with", "will",
        }
        expected_tokens = {
            token for token in exp_norm.split() if token not in stopwords
        }
        predicted_tokens = set(pred_norm.split())
        if not expected_tokens:
            return False
        return len(expected_tokens & predicted_tokens) / len(expected_tokens) >= 0.7

    def _validate_understanding(
        self, board: tt_sim.Board, question_type: str, question: str, predicted: Dict[str, Any],
        expected_answer: str = ""
    ) -> Dict[str, Any]:
        """Validate understanding answer against actual simulation or expected answer."""
        result = {"correct": False, "expected": {}, "error": None}

        try:
            trace_types = {
                "execution_trace", "ball_path", "output_sequence",
                "trigger_sequence", "state_trace", "state_tracking",
                "final_state", "pattern",
            }
            numeric_types = {"component_count", "parts_count", "numeric", "calculation"}

            if question_type in trace_types:
                if expected_answer:
                    predicted_text = self._prediction_text(predicted).lower()
                    exp_lower = expected_answer.lower()

                    if self._normalise_answer_text(
                        predicted_text
                    ) == self._normalise_answer_text(expected_answer):
                        result["correct"] = True
                        result["expected"] = {"answer": expected_answer}
                        return result

                    outcome_checks = []

                    if "left side" in exp_lower or "left exit" in exp_lower or "(left)" in exp_lower or "left lever" in exp_lower or "left_catcher" in exp_lower:
                        outcome_checks.append("left")
                    if "right side" in exp_lower or "right exit" in exp_lower or "(right)" in exp_lower or "right lever" in exp_lower or "right_catcher" in exp_lower:
                        outcome_checks.append("right")
                    if "blue exit" in exp_lower or "blue (left)" in exp_lower or "blue ball" in exp_lower and "trigger" not in exp_lower:
                        outcome_checks.append("blue")
                    if "red exit" in exp_lower or "red (right)" in exp_lower or "red ball" in exp_lower:
                        outcome_checks.append("red")
                    if "intercept" in exp_lower:
                        outcome_checks.append("intercept")

                    if "triggers the right lever" in exp_lower:
                        outcome_checks.append("right lever")
                    if "triggers the left lever" in exp_lower:
                        outcome_checks.append("left lever")
                    if "releasing a red" in exp_lower or "red ball is released" in exp_lower:
                        outcome_checks.append("red released")
                    if "releasing a blue" in exp_lower or "blue ball is released" in exp_lower:
                        outcome_checks.append("blue released")

                    coord_pattern = r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
                    expected_coords = [
                        (int(x), int(y))
                        for x, y in re.findall(coord_pattern, expected_answer)
                    ]
                    predicted_coords = [
                        (int(x), int(y))
                        for x, y in re.findall(coord_pattern, predicted_text)
                    ]
                    coords_match = not expected_coords or predicted_coords == expected_coords
                    markers_match = (
                        all(kw in predicted_text for kw in outcome_checks)
                        if outcome_checks else self._text_answer_matches(predicted_text, expected_answer)
                    )
                    matched = coords_match and markers_match
                    result["correct"] = matched
                    result["expected"] = {"answer": expected_answer}
                else:
                    board.reset()
                    sim_result = board.release_marble(tt_sim.Side.BLUE)
                    expected_catcher = sim_result.caught_by
                    if expected_catcher:
                        pred_text = " ".join(v for v in predicted.values() if isinstance(v, str)).lower()
                        result["correct"] = expected_catcher in pred_text
                    result["expected"] = {
                        "caught_by": sim_result.caught_by,
                        "path": sim_result.path,
                        "final_states": sim_result.final_state,
                    }

            elif question_type in numeric_types:
                if expected_answer:
                    pred_str = self._prediction_text(predicted)
                    pred_candidates = re.findall(r'-?\d+(?:\.\d+)?', pred_str)
                    exp_candidates = re.findall(r'-?\d+(?:\.\d+)?', str(expected_answer))

                    if not exp_candidates:
                        result["correct"] = self._text_answer_matches(
                            pred_str, str(expected_answer)
                        )
                    elif not pred_candidates:
                        result["correct"] = False
                    else:
                        exp_has_total = re.search(r'(?:total|of|are)\s+(\d+)', expected_answer)
                        pred_has_total = re.search(r'(?:total|of|are)\s+(\d+)', pred_str)

                        if exp_has_total and pred_has_total:
                            result["correct"] = exp_has_total.group(1) == pred_has_total.group(1)
                        elif exp_has_total:
                            result["correct"] = exp_has_total.group(1) == pred_candidates[0]
                        elif pred_has_total:
                            result["correct"] = pred_has_total.group(1) == exp_candidates[0]
                        else:
                            result["correct"] = pred_candidates[0] == exp_candidates[0]
                    result["expected"] = {"answer": expected_answer}
                else:
                    result["correct"] = None

            elif question_type == "select":
                if expected_answer:
                    predicted_text = self._normalise_answer_text(
                        self._prediction_text(predicted)
                    )
                    expected_text = self._normalise_answer_text(expected_answer)
                    result["correct"] = (
                        predicted_text == expected_text
                        or predicted_text.startswith(f"{expected_text} ")
                    )
                    result["expected"] = {"answer": expected_answer}
                else:
                    result["correct"] = None

            else:
                if expected_answer:
                    result["correct"] = self._text_answer_matches(
                        self._prediction_text(predicted), str(expected_answer)
                    )
                    result["expected"] = {"answer": expected_answer}
                else:
                    result["correct"] = None
                    result["expected"] = {"type": "manual review"}
                    result["note"] = f"{question_type} requires manual review"

        except Exception as e:
            result["error"] = str(e)
            result["correct"] = False

        return result

    def run_benchmark(
        self,
        pattern: str = "tt-official-ch*.json",
        max_tasks: Optional[int] = None,
        task_types: Optional[List[str]] = None,
        tiers: Optional[List[int]] = None,
    ) -> BenchmarkReport:
        """Run the full benchmark.

        Task types:
        - "understanding": Answer questions about board behavior
        - "agentic_synthesis": Use tools to build and verify solution iteratively

        Tiers:
        - Filter challenges by tier number (e.g., [1, 2]).
        - None means all tiers.
        """
        task_types = task_types or ["understanding", "agentic_synthesis"]

        # Validate challenges directory exists before globbing
        if not self.challenges_dir.is_dir():
            logger.error(
                f"Challenges directory does not exist: {self.challenges_dir}\n"
                f"  Pass --challenges-dir with an explicit path, or run from the repo root."
            )
            return BenchmarkReport(
                timestamp=datetime.now().isoformat(),
                model=self.llm.config.model,
                provider=self.llm.config.provider,
                total_tasks=0,
                successful=0,
                failed=0,
                task_results=[],
                error=f"challenges_dir not found: {self.challenges_dir}",
            )

        # Find challenge files
        challenge_files = sorted(self.challenges_dir.glob(pattern))

        # Filter by tier if specified
        if tiers is not None:
            tier_set = set(tiers)
            filtered = []
            for cf in challenge_files:
                try:
                    with open(cf) as f:
                        data = json.load(f)
                    file_tier = data.get("tier", 1)
                    if file_tier in tier_set:
                        filtered.append(cf)
                except Exception:
                    pass  # skip unreadable files
            challenge_files = filtered
            logger.info(
                "Tier filter %s → %d challenge(s)", sorted(tier_set), len(challenge_files)
            )

        # Truncate only after filtering, or a tier-limited run samples from the
        # wrong tier and can come back empty.
        if max_tasks:
            challenge_files = challenge_files[:max_tasks]

        logger.info(f"Found {len(challenge_files)} challenge files")

        task_types = task_types or ["understanding", "agentic_synthesis"]
        workers = getattr(self, "_workers", 1)

        # ── sequential path (default) ──────────────────────────────────
        save_per_task = getattr(self, "_save_per_task", False)

        if workers <= 1:
            for task_path in challenge_files:
                if "understanding" in task_types:
                    results = self.run_understanding_task(task_path)
                    self.results.extend(results)
                    if save_per_task:
                        for r in results:
                            self._save_task_result(r)
                if "agentic_synthesis" in task_types:
                    result = self.run_agentic_task(task_path)
                    self.results.append(result)
                    if save_per_task:
                        self._save_task_result(result)
        else:
            # ── parallel path ──────────────────────────────────────────
            logger.info(f"Using {workers} parallel workers")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures: dict = {}
                for task_path in challenge_files:
                    fut = pool.submit(
                        self._run_single_challenge, task_path, task_types,
                    )
                    futures[fut] = task_path

                for fut in as_completed(futures):
                    tp = futures[fut]
                    try:
                        chunk = fut.result()
                        self.results.extend(chunk)
                        if save_per_task:
                            for r in chunk:
                                self._save_task_result(r)
                        logger.info(
                            "Completed %s → %d result(s)", tp.name, len(chunk),
                        )
                    except Exception as exc:
                        logger.exception("Task %s failed: %s", tp.name, exc)

        successful = sum(1 for r in self.results if r.success is True)
        failed = len(self.results) - successful

        # Per-tier aggregation
        per_tier: Dict[int, Dict[str, int]] = {}
        task_tier_map: Dict[str, int] = {}
        for cf in challenge_files:
            try:
                with open(cf) as f:
                    data = json.load(f)
                task_id = data.get("task_id", cf.stem)
                if cf.stem.startswith("tt-official-") and task_id != cf.stem:
                    task_id = cf.stem
                task_tier_map[task_id] = data.get("tier", 1)
            except Exception:
                pass

        for r in self.results:
            tier = task_tier_map.get(r.task_id, 1)
            if tier not in per_tier:
                per_tier[tier] = {"total": 0, "successful": 0, "failed": 0}
            per_tier[tier]["total"] += 1
            if r.success is True:
                per_tier[tier]["successful"] += 1
            else:
                per_tier[tier]["failed"] += 1

        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            model=self.llm.config.model,
            provider=self.llm.config.provider,
            total_tasks=len(self.results),
            successful=successful,
            failed=failed,
            per_tier=per_tier,
            task_results=self.results,
        )

    @staticmethod
    def _result_to_dict(r: "TaskResult") -> Dict[str, Any]:
        """Serialize a single TaskResult to the JSON shape used in reports."""
        return {
            "task_id": r.task_id,
            "task_type": r.task_type,
            "success": r.success,
            "component_score": r.component_score,
            "llm_response": r.llm_response,
            "predicted": r.predicted,
            "expected": r.expected,
            "metrics": r.metrics,
            "error": r.error,
            "latency_ms": r.latency_ms,
            "tokens_used": r.tokens_used,
            "logprobs": r.logprobs,
        }

    def _save_task_result(self, result: "TaskResult") -> None:
        """Persist one task result immediately, as it is evaluated.

        Writes ``<output_dir>/per_task/<task_id>__<task_type>.json`` so a
        crash or Slurm time-out mid-run does not discard already-completed
        challenges — important for the large ``scaled`` set, where the
        aggregate report is only written after every task finishes.
        """
        per_task_dir = self.output_dir / "per_task"
        try:
            per_task_dir.mkdir(parents=True, exist_ok=True)
            safe_id = re.sub(
                r"[^A-Za-z0-9._-]", "_", f"{result.task_id}__{result.task_type}"
            )
            with open(per_task_dir / f"{safe_id}.json", "w") as f:
                json.dump(self._result_to_dict(result), f, indent=2)
        except Exception as exc:  # never let a save failure abort the run
            logger.warning(
                "Could not save per-task result for %s: %s", result.task_id, exc
            )

    def save_report(self, report: BenchmarkReport, filename: Optional[str] = None):
        """Save benchmark report to JSON."""
        if filename is None:
            filename = f"benchmark_{report.timestamp}.json"

        output_path = self.output_dir / filename

        # Convert results to serializable format
        results_data = [self._result_to_dict(r) for r in report.task_results]

        # Build per-tier summary with rates
        per_tier_json = {}
        for tier, stats in sorted(report.per_tier.items()):
            total = stats["total"]
            per_tier_json[str(tier)] = {
                "total": total,
                "successful": stats["successful"],
                "failed": stats["failed"],
                "success_rate": round(stats["successful"] / total * 100, 1) if total > 0 else 0,
            }

        # Record the decoding configuration IN the report. Until now the only
        # trace of it was the output directory name, which the sbatch suffixes
        # with _T<temp> for anything but greedy — so a plain directory was read
        # as greedy even for the runs made before that suffix existed (before
        # 2026-07-28 nothing could set the temperature and LLMConfig's 0.7
        # applied). Reports from two decoding regimes then land in one
        # directory, distinguishable only by their timestamps.
        data = {
            "timestamp": report.timestamp,
            "model": report.model,
            "provider": report.provider,
            "sampling": {
                "temperature": getattr(self.llm.config, "temperature", None),
                "seed": getattr(self.llm.config, "seed", None),
                "max_turns": self.max_turns,
            },
            # Harness-ablation arm settings. Recorded here for the same reason
            # the decoding configuration is: a directory name is not provenance,
            # and an arm mislabelled by a stale environment variable is
            # indistinguishable from a real effect once the run is over.
            "harness": {
                "prompt_variant": self.prompt_variant,
                "auto_simulate": self.auto_simulate,
                "revision_style": self.revision_style,
                "revision_rounds": self.revision_rounds,
                "declare_zero_parts": self.declare_zero_parts,
                "observable_inventory": self.observable_inventory,
                "max_tokens": self.max_tokens,
            },
            "total_tasks": report.total_tasks,
            "successful": report.successful,
            "failed": report.failed,
            "success_rate": report.successful / report.total_tasks
            if report.total_tasks > 0
            else 0,
            "per_tier": per_tier_json,
            "results": results_data,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Report saved to {output_path}")
        return output_path


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Turing Tumble Benchmark Runner")

    # LLM options
    parser.add_argument(
        "--provider", default="mock", choices=["openai", "anthropic", "ollama", "lmstudio", "vllm", "deepseek", "cloud", "mock"]
    )
    parser.add_argument("--model", default="gpt-4")
    parser.add_argument("--api-key", type=str, help="API key (or set env var)")
    parser.add_argument("--base-url", type=str, help="API base URL")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed forwarded to the provider. Only meaningful with "
        "--temperature > 0: greedy decoding is already deterministic, so repeated "
        "runs at temperature 0 measure vLLM's batching non-determinism rather than "
        "sampling variability, and a seed does not control that.",
    )
    parser.add_argument(
        "--declare-zero-parts",
        action="store_true",
        help="ABLATION: list part types with count 0 in the prompt instead of "
        "omitting them, so unavailability is stated rather than implied.",
    )
    parser.add_argument(
        "--observable-inventory",
        action="store_true",
        help="ABLATION: report the REMAINING inventory from get_board_state. By "
        "default it is declared once in the initial prompt and never again, so the "
        "agent must track consumption from memory across up to 25 turns.",
    )
    parser.add_argument(
        "--auto-simulate",
        action="store_true",
        help="ARM fb-auto: attach the target-sequence simulation to every "
        "successful placement or removal, so verifying costs the agent no turn. "
        "Implies --prompt-variant autosim unless one is given explicitly.",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=["urged", "optional", "autosim"],
        default=None,
        help="ARM fb-optional / fb-auto: agentic system prompt. 'urged' is the "
        "baseline (simulation commanded but unenforced); 'optional' removes the "
        "mandate; 'autosim' describes the injected feedback. Defaults to "
        "'autosim' with --auto-simulate and 'urged' otherwise.",
    )
    parser.add_argument(
        "--revision-style",
        choices=["off", "retry", "structured"],
        default="off",
        help="ARM rev-retry / rev-structured: what to do when a submitted "
        "solution fails scoring. 'off' ends the episode (baseline); 'retry' "
        "says only that it was rejected; 'structured' returns the scorer's "
        "reason plus the simulator state. Revisions draw on --max-turns. "
        "Requires --provider vllm.",
    )
    parser.add_argument(
        "--revision-rounds",
        type=int,
        default=3,
        help="Maximum rejected answers fed back per task (default: 3). Only "
        "meaningful with --revision-style.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature (default: 0.0, greedy). LLMConfig defaults to 0.7 "
            "and this flag did not exist, so every benchmark run sampled "
            "stochastically: one model re-run with an identical configuration "
            "scored 80/60/0%% and then 100/80/9%%. Greedy decoding removes that "
            "source of variance. Note that vLLM with continuous batching is still "
            "not bit-deterministic — batch composition changes reduction order — so "
            "repeated runs remain advisable for a headline number."
        ),
    )

    # Benchmark options
    parser.add_argument(
        "--challenges-dir", type=Path, default=Path(__file__).parent.parent.parent.parent / "data" / "tasks" / "official" / "challenges" / "json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(str(Path("benchmark_results")))
    )
    parser.add_argument("--pattern", default="tt-official-ch*.json")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=25,
        help="Max agentic turns per task (default: 25). Increase for complex challenges.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Max completion tokens per LLM call (default: 32768). Increase for verbose reasoning models.",
    )
    parser.add_argument(
        "--task-type",
        action="append",
        default=[],
        help="Task type: understanding, agentic_synthesis",
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="HTTP timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for task execution (default: 1 = sequential). "
             "Each worker runs one puzzle at a time.  Set to 4-8 for cloud LLM providers.",
    )
    parser.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=None,
        help="Filter challenges by tier (e.g., --tiers 1 2). If omitted, all tiers are run.",
    )
    parser.add_argument(
        "--save-report", action="store_true", help="Save benchmark report"
    )
    parser.add_argument(
        "--print-board",
        action="store_true",
        help="Print the ASCII board for each task while running",
    )
    parser.add_argument(
        "--compute-complexity",
        action="store_true",
        help="Compute and attach board complexity metrics to each task result",
    )
    parser.add_argument(
        "--capture-logprobs",
        action="store_true",
        help="Request token-level log probabilities from the LLM provider (OpenAI, DeepSeek). "
             "Logprobs are stored alongside each task result for confidence analysis.",
    )

    args = parser.parse_args()

    # Load environment variables from .env (explicit, not import-time side effect)
    llm_client_.load_env()

    # Create LLM client
    llm_config = llm_client_.LLMConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        capture_logprobs=args.capture_logprobs,
    )
    llm_client = llm_client_.create_llm_client(llm_config)

    # Create benchmark
    benchmark = TuringTumbleBenchmark(
        llm_client=llm_client,
        challenges_dir=args.challenges_dir,
        output_dir=args.output_dir,
        print_board=args.print_board,
        declare_zero_parts=args.declare_zero_parts,
        observable_inventory=args.observable_inventory,
        auto_simulate=args.auto_simulate,
        # --auto-simulate without a variant would leave the prompt ordering the
        # model to call run_simulation after every placement, spending turns to
        # re-derive feedback it already has — the arm would then measure that.
        prompt_variant=(
            args.prompt_variant
            if args.prompt_variant is not None
            else ("autosim" if args.auto_simulate else "urged")
        ),
        revision_style=args.revision_style,
        revision_rounds=args.revision_rounds,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        compute_complexity=args.compute_complexity,
    )
    benchmark._workers = args.workers if args.workers > 1 else 1
    # When saving is requested, also persist each challenge's result as it is
    # evaluated (output_dir/per_task/) so a time-out doesn't lose finished work.
    benchmark._save_per_task = args.save_report

    # Run benchmark
    task_types = args.task_type if args.task_type else ["understanding", "agentic_synthesis"]
    report = benchmark.run_benchmark(
        pattern=args.pattern,
        max_tasks=args.max_tasks,
        task_types=task_types,
        tiers=args.tiers,
    )

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Benchmark Results")
    print(f"{'=' * 50}")
    print(f"Provider: {report.provider}")
    print(f"Model: {report.model}")
    print(f"Tasks: {report.total_tasks}")
    print(f"Successful: {report.successful}")
    print(f"Failed: {report.failed}")
    if report.total_tasks > 0:
        print(f"Success Rate: {report.successful / report.total_tasks * 100:.1f}%")
    else:
        print("Success Rate: N/A (no tasks matched)")

    # Per-tier breakdown
    if report.per_tier:
        print(f"\nPer-Tier Breakdown")
        print(f"{'-' * 30}")
        for tier in sorted(report.per_tier):
            stats = report.per_tier[tier]
            total = stats["total"]
            rate = stats["successful"] / total * 100 if total > 0 else 0
            print(f"  Tier {tier}: {stats['successful']}/{total} ({rate:.1f}%)")

    # Save report
    if args.save_report:
        benchmark.save_report(report)

    # Unload the model from memory (for local providers like Ollama)
    llm_client.unload_model()
    logger.info("Model unloaded successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
