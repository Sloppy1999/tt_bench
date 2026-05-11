# Turing Tumble Benchmark -- Prompt Anatomy

*Generated: 2026-05-11 21:31*

---

## 1. Prompt Layer Architecture

Every benchmark prompt -- regardless of task type -- is composed of **five nested layers**:

| Layer | Name | Color | Purpose |
|-------|------|-------|---------|
| 1 | System Prompt | Blue (#2c5aa0) | Role definition + critical behavioral constraints |
| 2 | Board JSON | Mid-blue (#3d6b99) | Board geometry, hoppers, catchers, fixed components |
| 3 | Task Objective | Medium-blue (#4a7ab3) | Puzzle goal and the specific question to answer |
| 4 | Component Rules | Light-blue (#5a8ac4) | Canonical physics reference for all component types |
| 5 | Output Format | Lighter-blue (#7ba3d4) | Exact JSON response structure expected |

### Visual Layer Stack

```
+- Layer 1: System Prompt ------------------------------------------+
|  +- Layer 2: Board JSON ----------------------------------------+ |
|  |  +- Layer 3: Task Objective ------------------------------+ | |
|  |  |  +- Layer 4: Component Rules -----------------------+ | | |
|  |  |  |  +- Layer 5: Output Format --------------------+ | | | |
|  |  |  |  |  [LLM Response Structure]                  | | | | |
|  |  |  |  +---------------------------------------------+ | | |
|  |  |  +---------------------------------------------------+ | |
|  |  +-------------------------------------------------------+ |
|  +-----------------------------------------------------------+ |
+---------------------------------------------------------------+
```

---

## 2. System Prompts

### 2a. Procedural Understanding

```
You are an expert Turing Tumble analyst.
Given a board configuration, analyze its behavior and answer questions about it.
Respond ONLY with valid JSON in the specified format.
```

### 2b. Agentic Synthesis

```
You are a Turing Tumble solver agent.
You MUST use the provided tools to solve this puzzle. You cannot solve it by just thinking,
you MUST call the tools.

CRITICAL CONSTRAINT: Marbles may NOT fall through empty cells. Every cell a marble visits
between entering the board and reaching a catcher/interceptor MUST contain a component.
Solutions with any empty-cell traversal will be rejected even if the catcher counts are correct.

REQUIRED WORKFLOW (you MUST follow this exactly):
1. First call get_board_state to see what's already placed
2. Call place_component to add components from your available parts
3. Call run_simulation to test if it works
4. If wrong, adjust with more place_component or remove_component calls
5. Repeat steps 3-4 until the solution is correct
6. ONLY when simulation shows correct results, output your final solution

You MUST call run_simulation after EVERY component placement to verify!
Do not just think about the solution - you must USE the tools to build and test it.
```

---

## 3. Complete Example Prompts -- tt-official-ch01 "Gravity"

Challenge: *Make all of the blue balls (and only the blue balls) reach the end.*

### 3a. Procedural Understanding -- Execution Trace Question

#### SYSTEM PROMPT
```
You are an expert Turing Tumble analyst.
Given a board configuration, analyze its behavior and answer questions about it.
Respond ONLY with valid JSON in the specified format.
```

#### USER PROMPT
```
Analyze this Turing Tumble board configuration.

## Board (JSON)
{
  "width": 11,
  "height": 11,
  "dimensions": "11x11",
  "components": [
    {
      "type": "ramp_right",
      "x": 2,
      "y": 0
    },
    {
      "type": "ramp_left",
      "x": 3,
      "y": 1
    },
    {
      "type": "ramp_right",
      "x": 2,
      "y": 2
    },
    {
      "type": "ramp_left",
      "x": 3,
      "y": 3
    },
    {
      "type": "ramp_right",
      "x": 2,
      "y": 4
    },
    {
      "type": "ramp_left",
      "x": 3,
      "y": 5
    },
    {
      "type": "ramp_right",
      "x": 2,
      "y": 6,
      "role": "solution"
    },
    {
      "type": "ramp_left",
      "x": 3,
      "y": 7,
      "role": "solution"
    },
    {
      "type": "ramp_right",
      "x": 2,
      "y": 8,
      "role": "solution"
    },
    {
      "type": "ramp_left",
      "x": 3,
      "y": 9,
      "role": "solution"
    }
  ],
  "ball_hoppers": {
    "blue": {
      "x": 2,
      "count": 8
    },
    "red": {
      "x": 8,
      "count": 8
    }
  },
  "trigger_levers": {
    "left": {
      "x": 2
    },
    "right": {
      "x": 8
    }
  },
  "entry_mode": "inward"
}

## Component Rules
Turing Tumble Component Rules:
- CRITICAL: Marbles may NOT fall through empty in-board cells. Every cell a marble
  visits after entering the board must contain a component until reaching a
  catcher or interceptor. Solutions with free-fall gaps are INVALID.
- RAMP_RIGHT: Marble entering from above always exits to the lower-right.
- RAMP_LEFT: Marble entering from above always exits to the lower-left.
- BIT (state 0 pointing right): Marble exits lower-right AND bit flips to state 1.
- BIT (state 1 pointing left):  Marble exits lower-left  AND bit flips to state 0.
- GEAR_BIT: Behaves like BIT on impact. When one flips, every gear_bit in the same
  `gear_groups` entry flips with it (instantly, before the marble exits).
- GEAR: Couples neighbouring gear_bits; does not redirect marbles on its own.
- CROSSOVER: Marble entering from upper-left exits lower-right; upper-right exits lower-left.
- INTERCEPTOR: Marble is caught and the current run ends.
- TRIGGER: Marble passes through AND queues the release of one ball from the OPPOSITE-coloured hopper.
- Ball hoppers: a marble from hopper `side` enters the playfield at column `ball_hoppers.<side>.entry_x`, starting at y=0.
- Trigger levers (catchers): a marble that falls off the bottom is caught only if its column equals
  `trigger_levers.left.x` (left_catcher) or `trigger_levers.right.x` (right_catcher). Any other bottom column is a miss.

## Question Type: execution_trace

## Question: After the 1st blue marble, where does it end up?

## Expected Answer Format
{"final_destination": "left_catcher" or "right_catcher", "reasoning": "step by step..."}

Respond with JSON containing your answer and reasoning.
```

---

### 3b. Agentic Synthesis

#### SYSTEM PROMPT
```
You are a Turing Tumble solver agent.
You MUST use the provided tools to solve this puzzle. You cannot solve it by just thinking,
you MUST call the tools.

CRITICAL CONSTRAINT: Marbles may NOT fall through empty cells. Every cell a marble visits
between entering the board and reaching a catcher/interceptor MUST contain a component.
Solutions with any empty-cell traversal will be rejected even if the catcher counts are correct.

REQUIRED WORKFLOW (you MUST follow this exactly):
1. First call get_board_state to see what's already placed
2. Call place_component to add components from your available parts
3. Call run_simulation to test if it works
4. If wrong, adjust with more place_component or remove_component calls
5. Repeat steps 3-4 until the solution is correct
6. ONLY when simulation shows correct results, output your final solution

You MUST call run_simulation after EVERY component placement to verify!
Do not just think about the solution - you must USE the tools to build and test it.
```

#### USER PROMPT
```
Solve this Turing Tumble puzzle using the available tools.

## Board (JSON)
{
  "width": 11,
  "height": 11,
  "dimensions": "11x11",
  "ball_hoppers": {
    "blue": {
      "x": 2,
      "count": 8
    },
    "red": {
      "x": 8,
      "count": 8
    }
  },
  "trigger_levers": {
    "left": {
      "x": 2
    },
    "right": {
      "x": 8
    }
  },
  "entry_mode": "inward",
  "components": [
    {
      "type": "ramp_right",
      "x": 2,
      "y": 0
    },
    {
      "type": "ramp_left",
      "x": 3,
      "y": 1
    },
    {
      "type": "ramp_right",
      "x": 2,
      "y": 2
    },
    {
      "type": "ramp_left",
      "x": 3,
      "y": 3
    },
    {
      "type": "ramp_right",
      "x": 2,
      "y": 4
    },
    {
      "type": "ramp_left",
      "x": 3,
      "y": 5
    }
  ]
}

## Available Parts
ramp_right: 2, ramp_left: 2, crossover: 0, bit: 0, gear_bit: 0, gear: 0, interceptor: 0, trigger: 0

## Target Behavior
Make all of the blue balls (and only the blue balls) reach the end.

## Component Rules
Turing Tumble Component Rules:
- CRITICAL: Marbles may NOT fall through empty in-board cells. Every cell a marble
  visits after entering the board must contain a component until reaching a
  catcher or interceptor. Solutions with free-fall gaps are INVALID.
- RAMP_RIGHT: Marble entering from above always exits to the lower-right.
- RAMP_LEFT: Marble entering from above always exits to the lower-left.
- BIT (state 0 pointing right): Marble exits lower-right AND bit flips to state 1.
- BIT (state 1 pointing left):  Marble exits lower-left  AND bit flips to state 0.
- GEAR_BIT: Behaves like BIT on impact. When one flips, every gear_bit in the same
  `gear_groups` entry flips with it (instantly, before the marble exits).
- GEAR: Couples neighbouring gear_bits; does not redirect marbles on its own.
- CROSSOVER: Marble entering from upper-left exits lower-right; upper-right exits lower-left.
- INTERCEPTOR: Marble is caught and the current run ends.
- TRIGGER: Marble passes through AND queues the release of one ball from the OPPOSITE-coloured hopper.
- Ball hoppers: a marble from hopper `side` enters the playfield at column `ball_hoppers.<side>.entry_x`, starting at y=0.
- Trigger levers (catchers): a marble that falls off the bottom is caught only if its column equals
  `trigger_levers.left.x` (left_catcher) or `trigger_levers.right.x` (right_catcher). Any other bottom column is a miss.

## Your Task
Use the tools to build and verify a solution. Placements must target empty cells.
`get_board_state` returns this same canonical JSON shape after each edit;
`run_simulation` returns catcher counts, execution traces, and final bit states.

When you have a correct solution, output:
{
  "final_solution": [
    {"component_type": "ramp_left", "x": 3, "y": 5},
    {"component_type": "bit",       "x": 5, "y": 6, "state": 0}
  ],
  "success": true,
  "verification": {"left_catcher": 8, "right_catcher": 0}
}

Use the tools now. Start by checking the current board state.
```

---

## 4. Understanding vs. Synthesis -- Key Differences

| Aspect | Procedural Understanding | Agentic Synthesis |
|--------|-------------------------|-------------------|
| System prompt role | Expert analyst; JSON-only output | Solver agent; must call tools |
| Board JSON | Includes reference solution components | Fixed components only; no solution |
| Available parts | Not shown | Shown (defines agent's inventory) |
| LLM interaction | Single-shot JSON response | Multi-turn tool calls (up to 100 turns) |
| Output validation | Trace accuracy / state precision | Simulator replay + free-fall check + inventory |
| Reference questions | `execution_trace`, `component_role`, `abstraction` | N/A |
| Tool schema | None | `place_component`, `remove_component`, `run_simulation`, `get_board_state` |

---

## 5. Output Files

| File | Format | Purpose |
|------|--------|---------|
| `prompt_anatomy.html` | HTML | Interactive, self-contained (open in browser) |
| `prompt_anatomy_layer_diagram.svg` | Vector SVG | LaTeX Beamer / PowerPoint insertion |
| `prompt_anatomy_task_comparison.svg` | Vector SVG | Side-by-side task-type comparison |
| `prompt_anatomy_understanding_example.png` | Raster PNG @300dpi | Word / LaTeX documents |
| `prompt_anatomy_agentic_example.png` | Raster PNG @300dpi | Word / LaTeX documents |
| `prompt_anatomy.md` | Markdown | This documentation |
