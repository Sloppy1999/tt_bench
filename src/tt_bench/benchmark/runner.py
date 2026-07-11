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

from tt_bench.llm import client as llm_client_
from tt_bench.tools import executor as tool_executor_

from tt_bench import simulator as tt_sim
from tt_bench.analytics import metrics as complexity_metrics
from tt_bench.benchmark.prompts import (
    AGENTIC_PROMPT_TEMPLATE,
    AGENTIC_SYSTEM_PROMPT,
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
    ):
        self.llm = llm_client
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
        for comp_dict in board_data.get("fixed_components", []):
            comp = tt_sim.Component.from_dict(comp_dict)
            board.place(comp.x, comp.y, comp)
        if include_solution:
            for comp_dict in task_info.get("solution", {}).get("placed_components", []):
                comp = tt_sim.Component.from_dict(comp_dict)
                board.place(comp.x, comp.y, comp)
        for comp_dict in placements or []:
            comp = tt_sim.Component.from_dict(self._normalize_placement(comp_dict))
            board.place(comp.x, comp.y, comp)
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
    def _format_available_parts(avail: Dict[str, int]) -> str:
        lines = [f"  - {part}: {count}" for part, count in avail.items() if count > 0]
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
        return AGENTIC_PROMPT_TEMPLATE.format(
            board_json=self._format_board_json(board),
            available_parts=self._format_available_parts(task_info["available_parts"]),
            target_behavior=task_info["objective"],
            COMPONENT_RULES=COMPONENT_RULES,
        )

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
        reference_used = Counter(p["type"] for p in reference)
        reference_exceeds_inventory = any(
            count > available.get(component_type, 0)
            for component_type, count in reference_used.items()
        )
        if reference_exceeds_inventory:
            return True, "Inventory check skipped: reference solution exceeds declared inventory"

        used = Counter(p["type"] for p in placements)
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
        """Map catcher hits to the benchmark's blue/red final marble sequence."""
        colours: List[str] = []
        for result in results:
            if result.caught_by == "left_catcher":
                colours.append("blue")
            elif result.caught_by == "right_catcher":
                colours.append("red")
        return colours

    @staticmethod
    def _detect_free_fall(
        board: tt_sim.Board, results: List[tt_sim.MarbleResult]
    ) -> Tuple[bool, str]:
        """Detect illegal in-board movement through empty cells.

        The simulator can physically continue a marble through empty cells, but
        Turing Tumble puzzle solutions are only legal when a marble lands on a
        component at every in-board step after it enters from the hopper.
        """
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
                    and x in (board.left_catcher_x, board.right_catcher_x)
                ):
                    # The final coordinate just above a trigger lever is a
                    # catcher approach slot in several official encodings, not
                    # an illegal mid-board gap.
                    continue

                if 0 <= x < board.cols and 0 <= y < board.rows and curr not in board.components:
                    return True, f"marble {marble_idx} traversed empty cell {curr}"
        return False, ""

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

        actual_final = self._caught_colour_sequence(results)
        expected_final = task_info.get("solution", {}).get("final_marble_state")
        if expected_final is not None:
            if actual_final == expected_final:
                return True, f"Matched final marble sequence: {actual_final}"
            return False, f"Expected final marble sequence {expected_final}, got {actual_final}"

        expected_output = task_info.get("expected_output", {}) or {}
        expected_left = expected_output.get("left_catcher")
        expected_right = expected_output.get("right_catcher")
        if isinstance(expected_left, int) or isinstance(expected_right, int):
            left_count = sum(1 for r in results if r.caught_by == "left_catcher")
            right_count = sum(1 for r in results if r.caught_by == "right_catcher")
            if expected_left is not None and left_count != expected_left:
                return False, f"Expected left_catcher={expected_left}, got {left_count}"
            if expected_right is not None and right_count != expected_right:
                return False, f"Expected right_catcher={expected_right}, got {right_count}"
            return True, f"Matched catcher counts: left={left_count}, right={right_count}"

        # Last-resort heuristic for custom tasks without explicit ground truth.
        objective = task_info.get("objective", "").lower()
        if "blue" in objective and "red" not in objective:
            blue_count = task_info.get("board", {}).get("ball_hoppers", {}).get("blue", {}).get("count", 0)
            if actual_final == ["blue"] * blue_count:
                return True, f"All {blue_count} blue marbles reached the end"
            return False, f"Expected {blue_count} blue marbles, got {actual_final}"
        if "red" in objective and "blue" not in objective:
            red_count = task_info.get("board", {}).get("ball_hoppers", {}).get("red", {}).get("count", 0)
            if actual_final == ["red"] * red_count:
                return True, f"All {red_count} red marbles reached the end"
            return False, f"Expected {red_count} red marbles, got {actual_final}"

        return False, "No explicit expected output/final_marble_state available"

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

    def _validate_against_expected(
        self, board: tt_sim.Board, expected: Dict[str, Any], task_info: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate using explicit expected_output declaration."""
        # ``expected`` is retained for API compatibility. The most precise
        # ground truth for official tasks is ``solution.final_marble_state``;
        # ``expected_output`` may only contain descriptive metadata.
        input_seq = self._normalize_input_sequence(task_info.get("input_sequence", ["blue"]))
        results = board.run(input_seq)
        return self._validate_simulation_results(board, task_info, results)

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

            # Map question types from folder to internal types
            type_mapping = {
                "ball_path": "execution_trace",
                "output_sequence": "execution_trace",
                "component_count": "component_count",
                "trigger_sequence": "trigger_sequence",
            }

            logger.info(f"Running understanding tasks for: {task_id} ({len(questions)} questions)")

            for q in questions:
                q_type_raw = q.get("type", "")
                q_type = type_mapping.get(q_type_raw, q_type_raw) if q_type_raw else "unknown"
                question = q.get("question", "")
                expected_answer = q.get("answer", "")
                qid = q.get("qid", "")

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
                    validation_result = self._validate_understanding(
                        board, q_type, question, predicted, expected_answer=expected_answer
                    )

                    results.append(
                        TaskResult(
                            task_id=f"{task_id}_{qid}",
                            task_type="understanding",
                            success=validation_result["correct"],
                            llm_response=predicted.get("answer", ""),
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

    def run_agentic_task(self, task_path: Path) -> TaskResult:
        """Run an agentic synthesis task using function calling.

        The LLM uses tools to iteratively build and verify a solution.
        """
        start_time = time.time()

        try:
            task_info, _ = self.load_task(task_path)
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
            executor = tool_executor_.create_executor_from_task(
                board_data,
                fixed,
                available_parts=available_parts,
                target_sequence=self._normalize_input_sequence(
                    task_info.get("input_sequence", ["blue"])
                ),
            )

            # Build prompt
            prompt = self.build_agentic_prompt(task_info)

            # Run agent with tools
            final_result, error, tool_calls, tool_results, usage, turn_logprobs = self.llm.generate_with_tools(
                    prompt=prompt,
                    tools=llm_client_.TURING_TUMBLE_TOOLS,
                    tool_executor=executor,
                    system_prompt=AGENTIC_SYSTEM_PROMPT,
                    max_turns=self.max_turns,
                    max_tokens=self.max_tokens,
                )

            is_valid, msg = False, "No solution found"
            placed = executor.get_placed_components()
            solution_used = placed

            # Validate if the LLM submitted a final_answer *or* if it
            # placed components before running out of turns.  This catches
            # the common case where a model finds the right board but
            # exhausts its turn budget before emitting a final_solution.
            if final_result or placed:
                is_valid, msg = self.validate_synthesis(task_info, placed)

            # Fall back to the best board state recorded during successful
            # simulation runs — handles the case where the LLM places a
            # correct component, verifies it, then removes it.
            if not is_valid:
                best = executor.get_best_placement()
                if best and best != placed:
                    is_valid, msg = self.validate_synthesis(task_info, best)
                    if is_valid:
                        solution_used = best

            # Compute component-level accuracy against ground truth
            gt_placements = task_info.get("solution", {}).get("placed_components", [])
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
                },
                metrics={
                    "valid": float(is_valid),
                    "tool_calls_count": len(tool_calls),
                    "turns": len(tool_calls),
                    "component_score": comp_score,
                    "component_correct": comp_correct,
                    "component_placed": comp_placed,
                    "component_gt": comp_gt,
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
            "component_role": "This component [functions as...]",
            "abstraction": "This board performs [computation type]",
        }
        return formats.get(question_type or "", "Provide a clear answer.")

    def _validate_understanding(
        self, board: tt_sim.Board, question_type: str, question: str, predicted: Dict[str, Any],
        expected_answer: str = ""
    ) -> Dict[str, Any]:
        """Validate understanding answer against actual simulation or expected answer."""
        result = {"correct": False, "expected": {}, "error": None}

        try:
            if question_type in ("execution_trace", "ball_path", "output_sequence", "trigger_sequence"):
                if expected_answer:
                    string_values = []
                    if isinstance(predicted, dict):
                        for v in predicted.values():
                            if isinstance(v, str):
                                string_values.append(v)
                    predicted_text = " ".join(string_values).lower()
                    exp_lower = expected_answer.lower()

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

                    matched = any(kw in predicted_text for kw in outcome_checks)
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

            elif question_type == "component_count":
                if expected_answer:
                    import re

                    pred_str = " ".join(v for v in predicted.values() if isinstance(v, str))
                    pred_candidates = re.findall(r'\d+', pred_str)
                    exp_candidates = re.findall(r'\d+', expected_answer)

                    if not pred_candidates or not exp_candidates:
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

            elif question_type == "component_role":
                result["correct"] = None
                result["expected"] = {"type": "explanation"}
                result["note"] = "component_role requires manual review; no automated validation"

            elif question_type == "abstraction":
                result["correct"] = None
                result["expected"] = {"type": "computation description"}
                result["note"] = "abstraction requires manual review; no automated validation"

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
        if max_tasks:
            challenge_files = challenge_files[:max_tasks]

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

        data = {
            "timestamp": report.timestamp,
            "model": report.model,
            "provider": report.provider,
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
