# AGENTS.md

Turing Tumble benchmark for procedural understanding and program synthesis.
Use Python 3.12 and `uv`. Package configuration and pytest import paths live in
`pyproject.toml`; tests do not require the obsolete `PYTHONPATH=simulator` setting.

## Commands

```bash
uv run python -m pytest tests/test_regressions.py tests/test_tt_sim.py tests/test_canonical_board.py -v
uv run python -m pytest tests/test_target_contract.py -q
uv run python scripts/audit_official.py --all
uv run python scripts/build_dataset_manifest.py --output review/remediation/manifest.json
uv run tt-benchmark --help
uv run tt-simulate --load data/tasks/official/challenges/json/tt-official-ch01.json --verify
```

## Layout

- `src/tt_bench/simulator/`: board physics, components, rendering and validation.
- `src/tt_bench/simulator/targets.py`: shared explicit-output contract.
- `src/tt_bench/benchmark/runner.py`: loading, scoring and orchestration.
- `src/tt_bench/llm/client.py`: provider clients.
- `src/tt_bench/tools/executor.py`: agent tools and early-stop checks.
- `data/tasks/official/challenges/json/`: official challenges.
- `data/tasks/official/questions/`: understanding questions and schema README.
- `data/tasks/official/INDEX.json`: official metadata index.
- `data/tasks/challenges_1comp/`, `challenges_2comp/`, `scaled/`: derived tasks.
- `scripts/`: generation, transcription and audit utilities.
- `tests/`: test suite; pytest imports from `src/` via project configuration.

## Validation and dataset work

Coordinates are `(x, y)`, origin top-left. Dimensions come from each task;
hoppers have y=-1, trigger levers y=height. Use the task's input_sequence when
validating: Board.run() without arguments has different release behavior.

All supplied output declarations must agree: solution.final_marble_state,
required_output, numeric expected_output counts and declared final bit states.
Sequences support blue, red and intercepted. Empty/placeholder targets and
objective-text heuristics cannot establish a successful solution.

Preserve user edits. Keep audit artifacts outside data/. Do not regenerate boards
or overwrite targets merely to make checks pass. The eligibility manifest is a
mechanical screen, not proof of semantic correctness; negative labels require
separate certificates. See `DATASET_REMEDIATION_PLAN.md` for follow-up work.

Provider credentials use OPENAI_API_KEY and ANTHROPIC_API_KEY; mock needs no key.
Inspect the current CLI help before composing benchmark commands. Ruff and mypy
settings are declared in pyproject.toml; pytest is the primary regression check.
