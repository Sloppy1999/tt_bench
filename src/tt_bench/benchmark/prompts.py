"""Prompt templates for the Turing Tumble benchmark."""

from __future__ import annotations

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
Do not just think about the solution - you must USE the tools to build and test it.

In your final answer, the "explanation" field must describe your reasoning: what you observed
from the simulations, why you placed each component where you did, and whether you believe the
puzzle is solvable or unsolvable with the given inventory."""


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
  "explanation": "Step-by-step: I observed free fall at (3,9) from the simulation, placed a ramp_left there, and verified all 8 blue marbles reached the left catcher with no empty-cell traversals.",
  "verification": {{"left_catcher": 8, "right_catcher": 0}}
}}

If you determine the puzzle is unsolvable, set "success": false and explain why in "explanation".

Use the tools now. Start by checking the current board state."""
