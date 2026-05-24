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
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add simulator to import path before importing modules that depend on it.
sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))

import llm_client as llm_client_
import tool_executor

import tt_sim

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
    error: Optional[str] = None


# ============================================================================
# Prompt Templates
# ============================================================================

COMPONENT_RULES = """Turing Tumble Component Rules:
- CRITICAL: Marbles may NOT fall through empty in-board cells. Every cell a marble
  visits after entering the board must contain a component until reaching a
  catcher or interceptor. Solutions with free-fall gaps are INVALID.
- Coordinates: positions are (x, y) with x the column (0..width-1) and y the row (0..height-1). y grows downward.
- RAMP_RIGHT: Marble entering from above always exits to the lower-right.
- RAMP_LEFT: Marble entering from above always exits to the lower-left.
- CROSSOVER: Marble entering from upper-left exits lower-right; upper-right exits lower-left.
- BIT (state 0, pointing right): Marble exits lower-right AND bit flips to state 1.
- BIT (state 1, pointing left):  Marble exits lower-left  AND bit flips to state 0.
- GEAR_BIT: Behaves like BIT on impact. When one flips, every gear_bit in the same `gear_groups` entry flips with it (instantly, before the marble exits).
- GEAR: Couples neighbouring gear_bits; does not redirect marbles on its own.
- INTERCEPTOR: Marble is caught and the current run ends.
- TRIGGER: Marble passes through AND queues the release of one ball from the OPPOSITE-coloured hopper (blue trigger -> red ball, red trigger -> blue ball). Queued releases fire after the current marble terminates.
- Ball hoppers: a marble from hopper `side` enters the playfield at column `ball_hoppers.<side>.entry_x`, starting at y=0.
- Trigger levers (catchers): a marble that falls off the bottom is caught only if its column equals `trigger_levers.left.x` (left_catcher) or `trigger_levers.right.x` (right_catcher). Any other bottom column is a miss.
- All geometry (hoppers, entry columns, catchers, components, bit states, gear groups) is given exactly in the board JSON; do not assume defaults."""


UNDERSTANDING_SYSTEM_PROMPT = """You are an expert Turing Tumble analyst.
Given a board configuration, analyze its behavior and answer questions about it.
Respond ONLY with valid JSON in the specified format."""


UNDERSTANDING_PROMPT_TEMPLATE = """Analyze this Turing Tumble board configuration.

## Board (JSON)
{board_json}

## Component Rules
{COMPONENT_RULES}

## Question Type: {question_type}

## Question: {question}

## Expected Answer Format
{answer_format}

Respond with JSON containing your answer and reasoning."""


# ============================================================================
# Agentic Synthesis Prompt Templates
# ============================================================================

AGENTIC_SYSTEM_PROMPT = """You are a Turing Tumble solver agent.
You MUST use the provided tools to solve this puzzle. You cannot solve it by just thinking, 
you MUST call the tools.

CRITICAL CONSTRAINT: Marbles may NOT fall through empty cells. Every cell a marble visits
between entering the board and reaching a catcher/interceptor MUST contain a component.
Solutions with any empty-cell traversal will be rejected even if the catcher counts are correct.

INCREMENTAL STRATEGY (you MUST follow this):
- Place ONE component at a time, then run_simulation to verify.
- Target a single problematic cell from the free_fall_errors list.
- After each simulation, observe what changed and place the NEXT component.
- DO NOT try to plan all placements in your head — build the solution step by step.
- Each turn: think briefly, place ONE component, simulate. Repeat.

FIXED vs USER COMPONENTS: The board comes with pre-placed components. In get_board_state,
each component has a "source" field:
- "fixed" = part of the original board layout — you CANNOT remove these.
- "user"  = you placed it via place_component — you CAN remove/replace these.
Never attempt to remove a "fixed" component; it will fail and waste a turn.

REQUIRED WORKFLOW (you MUST follow this exactly):
1. First call get_board_state to see what's already placed (note which are fixed vs user)
2. Call run_simulation to identify free_fall_errors (empty cells marbles pass through)
3. Call place_component to fill ONE empty cell from the error list
4. Call run_simulation to verify the fix
5. Repeat steps 3-4, addressing one cell at a time, until NO free_fall_errors remain
6. ONLY when simulation shows correct results with zero free_fall_errors, output your final solution

You MUST call run_simulation after EVERY component placement to verify!
Be CONCISE — your analysis should be 2-3 sentences, not paragraphs.
Do not just think about the solution - you must USE the tools to build and test it."""


AGENTIC_PROMPT_TEMPLATE = """Solve this Turing Tumble puzzle using the available tools.

## Board (JSON)
{board_json}

## Available Parts 
{available_parts}

## Target Behavior
{target_behavior}

## Component Rules
{COMPONENT_RULES}

## Your Task
Use the tools to build and verify a solution. Placements must target empty cells.
`get_board_state` returns this same canonical JSON shape after each edit;
`run_simulation` returns catcher counts, execution traces, and final bit states.

When you have a correct solution, output:
{{
  "final_solution": [
    {{"component_type": "ramp_left", "x": 3, "y": 5}},
    {{"component_type": "bit",       "x": 5, "y": 6, "state": 0}}
  ],
  "success": true,
  "verification": {{"left_catcher": 8, "right_catcher": 0}}
}}

Use the tools now. Start by checking the current board state."""


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
    ):
        self.llm = llm_client
        self.challenges_dir = challenges_dir
        self.output_dir = output_dir
        self.print_board = print_board
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results: List[TaskResult] = []

        # Load questions from the questions folder
        self.questions_dir = challenges_dir.parent.parent / "questions"
        self._questions_cache: Dict[str, List[Dict[str, Any]]] = {}

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
                    predicted, error, usage = self.llm.generate_json(
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
                            },
                            error=validation_result.get("error", error),
                            latency_ms=int((time.time() - start_time) * 1000),
                            tokens_used=total_tokens,
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

            # Create tool executor with fixed components
            fixed = board_data.get("fixed_components", [])
            available_parts = task_info.get("available_parts", {})
            executor = tool_executor.create_executor_from_task(
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
            final_result, error, tool_calls, tool_results, usage = self.llm.generate_with_tools(
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
                llm_response=json.dumps(final_result) if final_result else "",
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
                },
                error=msg if not is_valid else error,
                latency_ms=int((time.time() - start_time) * 1000),
                tokens_used=usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
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
    ) -> BenchmarkReport:
        """Run the full benchmark.

        Task types:
        - "understanding": Answer questions about board behavior
        - "agentic_synthesis": Use tools to build and verify solution iteratively
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

        logger.info(f"Found {len(challenge_files)} challenge files")

        for task_path in challenge_files:
            if "understanding" in task_types:
                results = self.run_understanding_task(task_path)
                self.results.extend(results)

            if "agentic_synthesis" in task_types:
                result = self.run_agentic_task(task_path)
                self.results.append(result)

        successful = sum(1 for r in self.results if r.success is True)
        failed = len(self.results) - successful

        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            model=self.llm.config.model,
            provider=self.llm.config.provider,
            total_tasks=len(self.results),
            successful=successful,
            failed=failed,
            task_results=self.results,
        )

    def save_report(self, report: BenchmarkReport, filename: Optional[str] = None):
        """Save benchmark report to JSON."""
        if filename is None:
            filename = f"benchmark_{report.timestamp}.json"

        output_path = self.output_dir / filename

        # Convert results to serializable format
        results_data = []
        for r in report.task_results:
            results_data.append(
                {
                    "task_id": r.task_id,
                    "task_type": r.task_type,
                    "success": r.success,
                    "llm_response": r.llm_response,
                    "predicted": r.predicted,
                    "expected": r.expected,
                    "metrics": r.metrics,
                    "error": r.error,
                    "latency_ms": r.latency_ms,
                    "tokens_used": r.tokens_used,
                }
            )

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
        "--provider", default="mock", choices=["openai", "anthropic", "ollama", "deepseek", "mock"]
    )
    parser.add_argument("--model", default="gpt-4")
    parser.add_argument("--api-key", type=str, help="API key (or set env var)")
    parser.add_argument("--base-url", type=str, help="API base URL")

    # Benchmark options
    parser.add_argument(
        "--challenges-dir", type=Path, default=Path("../tasks/official/challenges/json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("scorer/benchmark_results")
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
        "--save-report", action="store_true", help="Save benchmark report"
    )
    parser.add_argument(
        "--print-board",
        action="store_true",
        help="Print the ASCII board for each task while running",
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
    )

    # Run benchmark
    task_types = args.task_type if args.task_type else ["understanding", "agentic_synthesis"]
    report = benchmark.run_benchmark(
        pattern=args.pattern,
        max_tasks=args.max_tasks,
        task_types=task_types,
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

    # Save report
    if args.save_report:
        benchmark.save_report(report)

    # Unload the model from memory (for local providers like Ollama)
    llm_client.unload_model()
    logger.info("Model unloaded successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
