# Turing Tumble Benchmark Results — Metric Definitions

## Overview

The benchmark evaluates Large Language Models (LLMs) on two task types across
Turing Tumble puzzle boards. Each board challenge (e.g., `ch01-pA`) is assessed
through multiple task variants: **trace** (execution_trace), **role** (component_role),
**abstract** (abstraction), and **synth** (agentic synthesis).

## Columns

| Column | Description |
|--------|-------------|
| **Board** | Board identifier and task variant (e.g., `ch01-pA / trace (run 1)`). Each execution_trace task is run twice to test consistency. |
| **Type** | `understand` = procedural understanding; `agentic` = program synthesis |
| **Succ.** | ✓ if the model produced the correct answer; ✗ otherwise |
| **Tokens** | Total LLM tokens consumed (prompt + completion). Captures model effort per task. |

### Understanding Metrics (green-accented columns)

| Metric | Description |
|--------|-------------|
| **trace_accuracy** | Fraction of correct marble path coordinates in the predicted vs. expected trace (0.0 – 1.0). Measures whether the model correctly predicts where a marble travels through the board. |
| **state_precision** | Fraction of correctly matched component state labels at the final timestep (0.0 – 1.0). Measures whether the model correctly predicts final bit positions, gear rotations, and interceptor states. |

### Agentic Metrics (teal-accented columns)

| Metric | Description |
|--------|-------------|
| **valid** | Boolean; whether the generated piece placement sequence is physically valid on the board (no overlapping pieces, all pieces placed within board bounds). |
| **tool_calls** | Number of tool interactions used to solve the puzzle (place, remove, run, get_state). Lower counts indicate more efficient solutions. |
| **turns** | Number of LLM reasoning-turns (model output → tool result → model output) required to reach a solution. |

## Success Criteria

- **Understanding tasks**: Success requires `trace_accuracy ≥ 0.8` **and**
  `state_precision ≥ 0.8` for execution_trace tasks; component_role and
  abstraction are scored qualitatively (marked —).
- **Agentic tasks**: Success requires `valid = True` **and** the solver produces
  a working piece configuration that passes the board's acceptance test.

## Color Encoding

| Element | Encoding |
|---------|----------|
| Green row headers | Understanding task metrics |
| Teal row headers | Agentic task metrics |
| ✓ in dark green | Task succeeded |
| ✗ in red (faded) | Task failed |
| Dashed cell | Metric not applicable for this task type |

## Example Interpretation

A row showing `ch02-pA / trace (run 1) / understand / ✗ / 1,450 tokens / 0.00 / 0.00`
means:
- The model failed to correctly predict the marble path for board ch02's pA variant
- 1,450 total tokens were consumed
- 0% of path coordinates matched; 0% of final states matched

---

*Generated from `scorer/benchmark_results/benchmark_2026-05-11T23:23:31.739466.json`
(gpt-4o-mini, OpenAI provider, 3 challenges, 12 tasks, 8.3% success rate)*