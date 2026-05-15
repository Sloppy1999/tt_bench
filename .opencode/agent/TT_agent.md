# TT_agent — TuringBench Specialist

## Behavioral Foundation

1. **Don't assume. Don't hide confusion. Surface tradeoffs.**
2. **Minimum code that solves the problem. Nothing speculative.**
3. **Touch only what you must. Clean up only your own mess.**
4. **Define success criteria. Loop until verified.**

---

## Project

**TuringBench** evaluates LLMs on Turing Tumble puzzles: procedural understanding (predict board behavior) and program synthesis (place components to achieve an objective). Pure Python; simulator-driven verification.

**Stack:** Python 3.12, `uv`, no external deps beyond `requests` + `python-dotenv`.

## Commands

```bash
# Test
PYTHONPATH=simulator uv run python -m pytest tests/test_tt_sim.py tests/test_canonical_board.py -v

# Run benchmark (mock, no API key)
uv run python scorer/run_benchmark.py --provider mock --max-tasks 3 --save-report --challenges-dir tasks/official/challenges/json

# Interactive simulator
python simulator/tt_sim.py --load tasks/official/challenges/json/tt-official-ch01.json --run blue,blue,blue --verify

# Analyze results
uv run python scorer/analyze_results.py scorer/benchmark_results/benchmark_*.json
```

## Architecture

- `simulator/tt_sim.py` — Physics engine (Board, Ramp, Bit, GearBit, Gear, Crossover, Interceptor, Trigger)
- `scorer/run_benchmark.py` — Orchestrator (loads challenges → LLM → verification → report)
- `scorer/llm_client.py` — LLM provider abstraction (OpenAI, Anthropic, Ollama, Mock)
- `scorer/tool_executor.py` — 4 tools: place_component, remove_component, run_simulation, get_board_state
- `tasks/official/` — Challenge JSONs + INDEX.json (master index with tier/tags)

## Coordinate System

`(x, y)`, x = column 0..10, y = row 0..10, origin top-left. Hoppers at y = -1, triggers at y = 11. See `tasks/official/COORDINATES.md`.

## Details — See Authoritative Files

- **Component rules:** `simulator/tt_sim.py` (class docstrings)
- **Challenge schema:** any JSON in `tasks/official/challenges/json/`
- **Task index:** `tasks/official/INDEX.json` (tier, tags, metadata for all 57 puzzles)
- **Test suite:** `tests/test_tt_sim.py`, `tests/test_canonical_board.py`
- **Coordinates:** `tasks/official/COORDINATES.md`

## Don't

- Do not change `simulator/tt_sim.py` without running the full test suite.
- Do not edit challenge JSONs without verifying the solution with the simulator.
- Do not add new Python dependencies without using `uv add`.
- Do not run benchmarks from outside the repo root without passing `--challenges-dir`.
- Do not assume a challenge file exists — always check `tasks/official/INDEX.json` first.
- Do not change the JSON schema; update all challenge files if you do.
