# AGENTS.md

Turing Tumble benchmark for evaluating LLMs on puzzle-solving tasks (procedural understanding + program synthesis). Pure Python, no external dependencies beyond `requests` and `python-dotenv`.

## Quick commands

```bash
# Run all tests (PYTHONPATH required so tests find the simulator module)
PYTHONPATH=simulator uv run python -m pytest tests/test_tt_sim.py tests/test_canonical_board.py -v

# Run a single test
PYTHONPATH=simulator uv run python -m pytest tests/test_tt_sim.py::TestBit::test_bit_flip -v

# Run benchmark (mock provider, no API key needed)
# IMPORTANT: default --challenges-dir is relative to CWD. Run from scorer/ OR pass explicit path.
uv run python scorer/run_benchmark.py --provider mock --max-tasks 3 --save-report --challenges-dir tasks/official/challenges/json

# Run benchmark with a real provider
OPENAI_API_KEY=... uv run python scorer/run_benchmark.py --provider openai --model gpt-4o --save-report
ANTHROPIC_API_KEY=... uv run python scorer/run_benchmark.py --provider anthropic --model claude-opus-4-7 --save-report
```

## Environment

- **Python 3.12** (set in `.python-version`)
- **Package manager:** `uv` (no pip/poetry/pdm)
- **No lint, typecheck, or formatter config** — there is no ruff/mypy/black config
- No CI/CD workflows exist

## Import gotcha — PYTHONPATH

`tests/test_tt_sim.py` imports from `tt_sim` directly. The simulator lives in `simulator/tt_sim.py`, so you **must** add `simulator/` to `PYTHONPATH` before running tests. The `scorer/run_benchmark.py` script handles this internally via `sys.path.insert`, but pytest does not.

The old CLAUDE.md references `simulator/tests/test_tt_sim.py` — that path does **not exist**. Tests are at `tests/test_tt_sim.py`.

## Architecture

```
simulator/tt_sim.py          — Physics engine: Board, Ramp, Bit, GearBit, Gear, Crossover, Interceptor, Trigger
simulator/board_renderer.py  — Matplotlib-based board renderer + MP4 animation export
scorer/run_benchmark.py      — Benchmark orchestrator (loads challenges, dispatches to LLM, produces reports)
scorer/llm_client.py         — LLM provider abstraction (OpenAI, Anthropic, Ollama, Mock) using raw `requests`
scorer/tool_executor.py      — Bridges LLM tool calls (place/remove/run/get_state) to the simulator
tasks/official/challenges/   — Challenge JSONs (57+ puzzles)
tasks/official/questions/    — Procedural understanding questions per challenge
tasks/official/INDEX.json    — Master index: tier, tags, metadata for all challenges
tests/                       — Test suite (test_tt_sim.py, test_canonical_board.py, test_llms/)
```

## Key details

- **Coordinate system:** `(x, y)`, x = column 0..10, y = row 0..10, origin top-left. Hoppers at y = -1, triggers at y = 11. See `tasks/official/COORDINATES.md`.
- **Board is 11×11 by default**, configurable via `Board(rows=N, cols=M)`.
- **Marble step limit:** 500 steps per marble to prevent infinite loops.
- **`scorer/scorer/benchmark_results/`** is a nested directory that really exists (not a mistake).
- **No official APRC** — CLAUDE.md mentions APRC but the `APRC/` directory is not present in this repo. Ignore APRC references.
- **`--task-type`** flag on `run_benchmark.py` accepts `understanding` and/or `agentic_synthesis`. Default runs both.
- **Provider env vars:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. Ollama and mock need none.
- **`--pattern`** accepts a glob to filter tasks (e.g., `"ch0[1-5]*"`).
