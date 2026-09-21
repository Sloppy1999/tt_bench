# Turing Tumble Benchmark

<!-- TODO: Add CI/CD badge -->
<!-- TODO: Add PyPI version badge -->
<!-- TODO: Add Python version badge -->
<!-- TODO: Add license badge -->

**A benchmark for evaluating large language models on mechanical computation and program synthesis through Turing Tumble puzzles.**

---

## Why This Exists

LLMs are increasingly used as coding agents, but writing code is more than syntax generation. It demands procedural reasoning: tracing execution, modeling state changes, and synthesizing physical configurations from constraints. Standard coding benchmarks rarely isolate these skills.

This benchmark uses **[Turing Tumble](https://www.turingtumble.com/)**, a physical board game where marbles roll through arrangements of ramps, bits, gears, and crossovers to perform computation. Each puzzle requires understanding how a spatial, clocked mechanical system processes information. The constraints are unforgiving: pieces must fit on a grid, gears must mesh, and marbles must reach specific catchers in a specified order.

**What we measure:**
- **Procedural understanding**: can the model trace marble paths, predict final bit states, explain component roles, and reason about counterfactuals?
- **Agentic synthesis**: given a board spec, available parts, and target behavior, can the model iteratively place, test, and refine components to build a working solution?

The dataset is built from 58 official Turing Tumble puzzles across 4 difficulty tiers, with 176 rubric-scored procedural-understanding questions and a full physics simulator for validation. For agentic synthesis, those 58 puzzles are also scaled into larger sets: `challenges_1comp` and `challenges_2comp` (variants that need exactly one or two additional components), and `scaled` (the same puzzles rebuilt on larger 13×13 and 15×15 boards), totaling over 4,000 task files.

---

## Current status

The project has moved past the original single-machine prototype. Benchmarking runs on the JURECA HPC cluster (`jureca/`) against a local vLLM server, and **seven** instruction-tuned models have now been evaluated end to end across **all four difficulty tiers**: `qwen3.8-flash-next-awq4`, `qwen3.6-35B-A3B`, `gpt-oss-120b`, `gemma-4-31B-it`, `gemma-4-26B-A4B-it`, `Qwen2.5-Coder-7B-Instruct`, and `DeepSeek-Coder-V2-Lite`.

The full write-up, with every table regenerated directly from the stored results, lives in [`docs/results_analysis.md`](docs/results_analysis.md). The short version:

- **Tiers 3 and 4 were unscoreable until recently, and the repair mattered more than the run.** Only 39 of the 58 official boards could be scored at all — with the damage concentrated exactly where the difficulty was meant to be measured (tier 3 at 4/10, tier 4 at 3/15). Benchmarking before the repair would have scored most of tiers 3–4 as automatic zeros for every model and produced a confident "tiers 3–4 are much harder" that measured corpus decay, not capability. The corpus is now at 54/58 (§14.1).
- **Tier 4 is the hardest tier**, by a factor of 2.7 over tier 1 pooled across all seven models (32.5% → 12.2%). Tier 3 is not separable from tier 2 at n = 8 boards. Reference solutions grow from 6.1 parts (tier 1) to 12.9 (tier 4), which is the likely mechanism.
- **`qwen3.8-flash-next-awq4` leads every tier**, at 72.7% overall against ≤27.3% for the rest of the roster.
- **The dominant limitation is still spatial, not conceptual.** For six of seven models, 55–93% of failures are an *incomplete path* — the marble crosses a cell the model never filled. Only `qwen3.8-flash-next-awq4` has an inverted profile (67% of its failures are boards that run correctly and compute the wrong answer), which means it is the only model currently being tested on procedural reasoning rather than on whether it can finish building a board.
- **Generalising across inputs is out of reach above two bits.** Eight boards are scored by a *trial table* — the same board re-run under several ball counts or every starting bit configuration, all of which must pass. No model solved the 3-bit (8-configuration) or 9-bit (512-configuration) reversal boards.
- Models fail in two distinct ways, replicated on a third corpus: the Gemma models run out of their 25-turn budget still working (~90% of failures), while everything else stops early at a median of 2–11 turns, convinced (wrongly) that they're done.
- Decoding temperature is not a free reproducibility win: greedy decoding on identical inputs still varies (vLLM's continuous batching, not sampling). Two runs of the same seven models over an identical 22-board set differed by −3 to +2 tasks, which widened the previous ±2-task "unresolved by a single run" rule to ±3 (§14.2).

`docs/results_analysis.md` also documents what's still open: the four-tier results are a **single run per cell** with no confidence intervals, so no within-tier ranking below the leader is resolved; four boards remain unrepaired (three of them unscoreable by construction); and §7's five-repetition statistical design has not been re-applied to the corrected corpus or the expanded roster.

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

### Run the benchmark (mock provider, no API key needed)

```bash
uv run tt-benchmark --provider mock --max-tasks 3 --save-report
```

This defaults to the 58 official challenges at `data/tasks/official/challenges/json`. Point `--challenges-dir` at `data/tasks/scaled`, `data/tasks/challenges_1comp`, or `data/tasks/challenges_2comp` to run the larger sets.

### Run the simulator

```bash
# Interactive mode
uv run tt-simulate

# Load and render a challenge
uv run tt-simulate --load data/tasks/official/challenges/json/tt-official-ch01.json

# Run a marble sequence and verify
uv run tt-simulate \
  --load data/tasks/official/challenges/json/tt-official-ch01.json \
  --run blue,blue,blue --verify
```

### Run the test suite

```bash
uv run python -m pytest tests/ -v
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

# vLLM (local or cluster-hosted OpenAI-compatible server)
uv run tt-benchmark --provider vllm --model Qwen/Qwen3.6-35B-A3B \
  --base-url http://localhost:8000/v1 --save-report

# DeepSeek
DEEPSEEK_API_KEY=sk-... uv run tt-benchmark \
  --provider deepseek --model deepseek-chat --save-report
```

**Key options:**

| Flag | Description |
|------|-------------|
| `--provider` | `openai`, `anthropic`, `ollama`, `lmstudio`, `vllm`, `deepseek`, `cloud`, `mock` |
| `--model` | Model name (provider-specific) |
| `--temperature` | Sampling temperature (default `0.0`, greedy) |
| `--pattern` | Glob filter for tasks (e.g., `"ch0[1-5]*"`) |
| `--tiers` | Filter challenges by tier, e.g. `--tiers 1 2` |
| `--task-type` | `understanding`, `agentic_synthesis`, or both (default) |
| `--max-tasks` | Cap the number of tasks run |
| `--max-turns` | Turn budget per task (default `25`) |
| `--workers` | Parallel task workers (default `1`) |
| `--save-report` | Write results JSON to `benchmark_results/` |

### Analyzing results

```bash
uv run python jureca/inspect_results.py            # summary table
uv run python jureca/inspect_results.py --errors    # failure families
bash jureca/make_figures.sh                         # figures
```

### Running on JURECA

`jureca/` holds the HPC evaluation pipeline: `setup.sh` provisions a venv and vLLM on the login node, `submit_all.sh` submits one Slurm job per model against the full task roster (`bash jureca/submit_all.sh --list` shows it), and `aggregate_results.sh` / `inspect_results.py` collect and summarize the resulting reports. `submit_all.sh --help` (or reading the script header) documents the per-model GPU allocation and the known quirks of each model on that cluster.

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

**Metrics:** per-item answer correctness (binary). The earlier `trace_accuracy`
and `state_precision` fields were never computed and have been removed; item-level
trace scoring is future work.

### 2. Agentic Synthesis

Given an empty or partially filled board, a parts inventory, and target behavior, the model iteratively builds a solution using the following tools:

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
| Ollama | `llama3.1`, `qwen3` | (none, local) |
| vLLM | any OpenAI-compatible server | (none, or `--api-key`) |
| DeepSeek | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| Mock | (testing only) | (none) |

All providers implement a unified interface in `src/tt_bench/llm/client.py`.

---

## Project Structure

```
.
├── src/tt_bench/          # Package source
│   ├── simulator/         #   Physics engine: Board, components, renderer
│   ├── llm/               #   LLM client (multi-provider)
│   ├── benchmark/         #   Runner, prompts, task dispatch
│   ├── tools/             #   Tool executor (place, remove, run, inspect)
│   ├── analytics/         #   Metrics, logprob analysis
│   └── cli/               #   CLI entry points (tt-benchmark, tt-simulate)
├── data/tasks/            # Challenge sets
│   ├── official/          #   58 official challenges + 176 questions
│   ├── challenges_1comp/  #   Variants needing one additional component
│   ├── challenges_2comp/  #   Variants needing two additional components
│   └── scaled/            #   Official challenges rebuilt on 13x13/15x15 boards
├── jureca/                # JURECA HPC evaluation pipeline
│   ├── setup.sh             #   Provision venv + vLLM on the login node
│   ├── submit_all.sh        #   Submit one Slurm job per model
│   ├── aggregate_results.sh #   Collect per-model, per-set reports
│   ├── inspect_results.py   #   Summary tables and failure-family breakdown
│   └── make_figures.sh      #   Render the presentation figures
├── scripts/               # Analysis and visualization scripts
├── experiments/           # Raw experimental results by model
├── tests/                 # Test suite
├── docs/
│   └── results_analysis.md # Full results write-up, regenerated from stored data
├── assets/                # Diagrams, visuals
└── pyproject.toml         # Project configuration
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
