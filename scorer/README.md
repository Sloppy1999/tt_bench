# Turing Tumble Benchmark

A benchmark framework for evaluating Large Language Models on Turing Tumble puzzles.

## Overview

This benchmark evaluates LLMs on two complementary cognitive tasks:

1. **Program Synthesis**: Given a board configuration and desired output, generate the correct component placement
2. **Procedural Understanding**: Given a solution, predict behavior and explain the mechanical computation

## Architecture

```
scorer/
├── run_benchmark.py       # Main benchmark runner
├── llm_client.py          # LLM provider interface
├── analyze_results.py    # Results analysis toolkit
├── auto_complexity_scorer.py  # Question complexity scoring
├── test_auto_complexity_scorer.py
└── benchmark_results/     # Output directory
```

## Quick Start

### Run Benchmark with Mock LLM

```bash
cd /home/hacktheduck/Projects/Thesis/LLM_exp
uv run python scorer/run_benchmark.py --provider mock --max-tasks 3 --save-report
```

### Run with OpenAI

```bash
export OPENAI_API_KEY=your_key_here
uv run python scorer/run_benchmark.py \
  --provider openai \
  --model gpt-5.4 \
  --save-report
```

## LLM Providers

Supported providers in `llm_client.py`:

| Provider | Model Examples | Environment Variable |
|----------|----------------|---------------------|
| `openai` | gpt-5.4,gpt-5.4-mini | `OPENAI_API_KEY` |
| `anthropic` | claude-4.7-Opus, claude-4.6-sonnet | `ANTHROPIC_API_KEY` |
| `ollama` | gemma4, Qwen3.5 (local) | (none) |
| `mock` | (testing only) | (none) |

## Benchmark Runner

### CLI Options

```bash
uv run python scorer/run_benchmark.py --help
```

| Option | Description |
|--------|-------------|
| `--provider` | LLM provider (openai/anthropic/ollama/mock) |
| `--model` | Model name |
| `--api-key` | API key (or set env var) |
| `--base-url` | API base URL |
| `--challenges-dir` | Path to challenge JSON files |
| `--pattern` | Glob pattern for tasks |
| `--max-tasks` | Limit number of tasks |
| `--save-report` | Save results to JSON |

### Program Synthesis Task

Given a board specification and target behavior, the LLM must generate a valid component placement.

**Prompt includes:**
- Board dimensions and fixed components
- Available parts inventory
- Target behavior specification

**Validation:**
- Solution is executed in the simulator
- Output sequence is checked against expected behavior

### Procedural Understanding Task

Given a board configuration, the LLM must predict behavior or explain mechanisms.

**Question types:**
- Execution trace: "After the 3rd marble, what are the bit states?"
- Component role: "What function does this bit serve?"
- Counterfactual: "If we flip bit (2,3), how does output change?"

## Analysis Toolkit

### Basic Analysis

```bash
uv run python scorer/analyze_results.py scorer/benchmark_results/benchmark_2024*.json
```

### Compare Models

```bash
uv run python scorer/analyze_results.py \
  --compare scorer/benchmark_results/benchmark_openai.json \
            scorer/benchmark_results/benchmark_anthropic.json
```

### Output

The analysis produces:
- Success/failure counts
- Error categorization (syntax, API, logic, validation, timeout)
- Average latency
- Per-task breakdown table

## Task Files

Tasks are loaded from:
```
tasks/official/challenges/json/tt-official-ch*.json
```

Each task includes:
- Board specification
- Fixed components
- Available parts
- Solution (for validation)
- Objective (target behavior)

## Metrics

### Synthesis Metrics
- **Functional Correctness**: Binary pass/fail
- **Part Efficiency**: Parts used vs. optimal
- **Latency**: Time to generate solution

### Understanding Metrics
- **Trace Accuracy**: % of execution steps correct
- **State Precision**: Hamming distance on bit states

## Question Complexity Scorer

Also includes a complexity scorer for categorizing questions:

```bash
uv run python scorer/auto_complexity_scorer.py --write
```

See `scorer/README.md` for detailed documentation.

## Testing

```bash
# Test LLM client
uv run python scorer/llm_client.py --provider mock

# Test complexity scorer
uv run python -m pytest scorer/test_auto_complexity_scorer.py -v

# Test benchmark runner (dry run)
uv run python scorer/run_benchmark.py --provider mock --max-tasks 2
```

## Configuration

### Custom LLM Endpoint

For OpenAI-compatible APIs:

```bash
uv run python scorer/run_benchmark.py \
  --provider openai \
  --base-url https://your-api.com/v1 \
  --api-key your_key
```

### Ollama (Local)

```bash
uv run python scorer/run_benchmark.py \
  --provider ollama \
  --model llama2
```

## Output Format

Benchmark results are saved as:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "model": "gpt-4",
  "provider": "openai",
  "total_tasks": 10,
  "successful": 8,
  "failed": 2,
  "success_rate": 0.8,
  "results": [
    {
      "task_id": "tt-official-ch01",
      "task_type": "synthesis",
      "success": true,
      "latency_ms": 1500,
      "predicted": {...},
      "metrics": {"valid": 1.0, "parts_used": 4}
    }
  ]
}
```
