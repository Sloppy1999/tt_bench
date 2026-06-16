# Turing Tumble Benchmark

<!-- TODO: Add CI/CD badge -->
<!-- TODO: Add PyPI version badge -->
<!-- TODO: Add Python version badge -->
<!-- TODO: Add license badge -->

**A benchmark for evaluating large language models on mechanical computation and program synthesis through Turing Tumble puzzles.**

---

## Why This Exists

LLMs are increasingly used as coding agents — but writing code is more than syntax generation. It demands procedural reasoning: the ability to trace execution, model state changes, and synthesize physical configurations from constraints. Standard coding benchmarks rarely isolate these skills.

This benchmark uses **[Turing Tumble](https://www.turingtumble.com/)** — a physical board game where marbles roll through arrangements of ramps, bits, gears, and crossovers to perform computation. Each puzzle requires understanding how a spatial, clocked mechanical system processes information. The constraints are unforgiving: pieces must fit on a grid, gears must mesh, and marbles must reach specific catchers in a specified order.

**What we measure:**
- **Procedural Understanding** — Can the model trace marble paths, predict final bit states, explain component roles, and reason about counterfactuals?
- **Agentic Synthesis** — Given a board spec, available parts, and target behavior, can the model iteratively place, test, and refine components to build a working solution?

The benchmark includes 58 puzzles across 4 difficulty tiers, 160+ rubric-scored questions, and a full physics simulator for validation.

---

## Architecture

<!-- TODO: Add Architecture diagram -->

---

## Installation

**Requirements:** Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/hacktheduck/tt-bench.git
cd tt-bench
uv sync
```

The project installs as the package `tt_bench` with two CLI entry points:

| Command | Description |
|---------|-------------|
| `tt-benchmark` | Run LLM evaluations |
| `tt-simulate` | Run the simulator interactively |

---

## Quick Start

### Run the benchmark (mock provider — no API key needed)

```bash
uv run tt-benchmark --provider mock --max-tasks 3 --save-report
```

Or directly:

```bash
PYTHONPATH=simulator uv run python scorer/run_benchmark.py \
  --provider mock --max-tasks 3 --save-report \
  --challenges-dir tasks/official/challenges/json
```

### Run the simulator

```bash
# Interactive mode
python simulator/tt_sim.py

# Load and render a challenge
python simulator/tt_sim.py --load tasks/official/challenges/json/tt-official-ch01.json

# Run a marble sequence and verify
python simulator/tt_sim.py \
  --load tasks/official/challenges/json/tt-official-ch01.json \
  --run blue,blue,blue --verify

# Export an MP4 animation
python simulator/board_renderer.py --task tt-official-ch02 --animate --fps 12
```

### Run the test suite

```bash
PYTHONPATH=simulator uv run python -m pytest tests/ -v
```

---

## Usage

### Benchmarking an LLM

```bash
# OpenAI
OPENAI_API_KEY=sk-... uv run tt-benchmark \
  --provider openai --model gpt-4o --save-report

# Anthropic
ANTHROPIC_API_KEY=sk-ant-... uv run tt-benchmark \
  --provider anthropic --model claude-sonnet-4-20250514 --save-report

# Ollama (local)
uv run tt-benchmark --provider ollama --model llama3.1 --save-report

# DeepSeek
DEEPSEEK_API_KEY=sk-... uv run tt-benchmark \
  --provider deepseek --model deepseek-chat --save-report

# Custom OpenAI-compatible endpoint
uv run tt-benchmark \
  --provider openai --base-url https://your-api.com/v1 --api-key your-key
```

**Key options:**

| Flag | Description |
|------|-------------|
| `--provider` | `openai`, `anthropic`, `ollama`, `deepseek`, `mock` |
| `--model` | Model name (provider-specific) |
| `--pattern` | Glob filter for tasks (e.g., `"ch0[1-5]*"`) |
| `--task-type` | `understanding`, `agentic_synthesis`, or both (default) |
| `--max-tasks` | Cap the number of tasks run |
| `--save-report` | Write results JSON to `scorer/benchmark_results/` |

### Analyzing Results

```bash
# Single report
uv run python scorer/analyze_results.py scorer/benchmark_results/benchmark_*.json

# Compare models
uv run python scorer/analyze_results.py \
  --compare results_openai.json results_anthropic.json
```

---

## Task Types

### 1. Procedural Understanding

Given a complete board configuration, the model answers questions that probe its ability to mentally simulate mechanical computation:

| Question Type | Example |
|---------------|---------|
| **Execution trace** | "After the 3rd marble, list each (x,y) coordinate the marble passes through." |
| **Component role** | "What function does the bit at (2,3) serve in this board?" |
| **Abstraction** | "Which logical operation does this board implement?" |
| **Counterfactual** | "If bit (2,3) were flipped to state 1, how would the output sequence change?" |

**Metrics:** Trace accuracy (fraction of correct coordinates), state precision (component state match).

### 2. Agentic Synthesis

Given an empty or partially-filled board, a parts inventory, and target behavior, the model iteratively builds a solution using the following tools:

| Tool | Function |
|------|----------|
| `place_component(x, y, type, ...)` | Place a ramp, bit, gear, etc. |
| `remove_component(x, y)` | Remove a placed component |
| `run_simulation(sequence)` | Release marbles and observe output |
| `get_board_state()` | Inspect current board configuration |


---

## Challenge Tiers

Puzzles are organized into 4 difficulty tiers drawn from the Turing Tumble Practice Guide:

| Tier | Skills |
|------|--------|
| 1 | Basic routing, ramps |
| 2 | Bits, state, counting |
| 3 | Gears, gear bits, synchronization |
| 4 | Multi-component integration, complex logic |

Each challenge may have practice variants (e.g., `ch01-pA`) and bonus variants (e.g., `ch29-bA`).

---

## Providers

| Provider | Model Examples | Environment Variable |
|----------|---------------|---------------------|
| OpenAI | `gpt-4o`, `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| Ollama | `llama3.1`, `qwen3` | (none — local) |
| DeepSeek | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| Mock | (testing only) | (none) |

All providers implement a unified interface in `src/tt_bench/llm/client.py`.

---

## Project Structure

```
.
├── simulator/              # Original physics engine (standalone)
│   ├── tt_sim.py           #   Board, components, marble simulation
│   └── board_renderer.py   #   Matplotlib renderer + MP4 export
├── src/tt_bench/           # Package source (in migration)
│   ├── simulator/          #   Physics engine (package version)
│   ├── llm/                #   LLM client (multi-provider)
│   ├── benchmark/          #   Runner, prompts, task dispatch
│   ├── tools/              #   Tool executor (place, remove, run, inspect)
│   ├── analytics/          #   Metrics, analysis
│   └── cli/                #   CLI entry points
├── scorer/                 # Benchmark scripts (legacy + primary)
│   ├── run_benchmark.py    #   Main benchmark orchestrator
│   ├── llm_client.py       #   LLM provider abstraction
│   ├── analyze_results.py  #   Results analysis
│   └── benchmark_results/  #   Output directory
├── tasks/official/         # 58 Turing Tumble challenges
│   ├── challenges/json/    #   Board definitions (JSON)
│   ├── questions/          #   160+ rubric-scored questions
│   └── INDEX.json          #   Task metadata (tier, tags, etc.)
├── tests/                  # Test suite
├── experiments/            # Raw experimental results
│   ├── gpt-4o/             #   OpenAI results
│   ├── gpt-5.4/            #   OpenAI results
│   └── complexity_boards/  #   Board complexity visualizations
├── analytics/              # Analysis notebooks and scripts
├── assets/                 # Diagrams, visuals
└── pyproject.toml          # Project configuration
```

---

## Contributing

Contributions are welcome. 


---

## Citation

<!-- TODO: Add Citation -->

---

## License

<!-- TODO: Add license -->
