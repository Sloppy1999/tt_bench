# Results

Draft for the Results chapter. Every number here comes from
`jureca/inspect_results.py` over `benchmark_results/jureca_tier1/`; the confidence
intervals and significance tests are recomputed from the task counts, not taken
from any report field. Regenerate with:

```
python3 jureca/inspect_results.py            # table
python3 jureca/inspect_results.py --errors   # failure families
bash jureca/make_figures.sh                  # figures
```

---

## 1. Experimental setup

Five instruction-tuned models were evaluated on TT-Bench Tier 1 through a local
vLLM 0.24.0 OpenAI-compatible server on JURECA `dc-hwai` nodes (4× NVIDIA H100).
All models ran under an identical harness configuration:

| Parameter | Value |
|---|---|
| Task type | `agentic_synthesis` (tool-calling agent loop) |
| Turn budget | 25 |
| Max tokens per turn | 32 768 |
| Target context | 131 072 |
| Parallel workers | 8 |
| Per-task timeout | 600 s |

Three challenge sets form a difficulty axis:

| Set | n (Tier 1) | Description |
|---|---|---|
| `scaled_1comp` | 432 | Solutions requiring **one** component |
| `scaled_2comp` | 402 | Solutions requiring **two** components |
| `scaled` | 1 013 | Official challenges on **larger boards** (13×13, 15×15) |

The small `official` / `1comp` / `2comp` sets (11, 5 and 5 tasks) were also run
but are **not used for any quantitative claim**: at n = 5 a single task moves the
rate by 20 percentage points. Section 7 documents why that matters.

---

## 2. Overall performance

Success rate with 95 % Wilson confidence intervals:

| Model | `scaled_1comp` (n=432) | `scaled_2comp` (n=402) | `scaled` (n=1013) |
|---|---|---|---|
| qwen3.6-35B-A3B | **45.6 %** [41.0, 50.3] | **40.3 %** [35.6, 45.2] | 11.3 % [9.5, 13.3] |
| gpt-oss-120b | 44.9 % [40.3, 49.6] | 38.6 % [33.9, 43.4] | **11.8 %** [10.0, 14.0] |
| gemma-4-31B-it | 39.8 % [35.3, 44.5] | 33.3 % [28.9, 38.1] | 5.8 % [4.5, 7.4] |
| gemma-4-26B-A4B-it | 36.8 % [32.4, 41.5] | 21.6 % [17.9, 25.9] | 3.9 % [2.9, 5.3] |
| DeepSeek-Coder-V2-Lite | 5.8 % [4.0, 8.4] | 0.0 % [0.0, 0.9] | 1.4 % [0.8, 2.3] |

**The ranking is stable across all three difficulty levels.** Not every gap in it
is statistically meaningful, however. Two-proportion z-tests on adjacent pairs:

| Comparison | `scaled_1comp` | `scaled_2comp` | `scaled` |
|---|---|---|---|
| qwen3.6 vs gpt-oss | p = 0.84 | p = 0.61 | p = 0.68 |
| gpt-oss vs gemma-31B | p = 0.13 | p = 0.12 | **p < 0.001** |
| gemma-31B vs gemma-26B | p = 0.36 | **p < 0.001** | p = 0.050 |
| gemma-26B vs DeepSeek | **p < 0.001** | **p < 0.001** | **p < 0.001** |

Three performance tiers are supported by the data:

1. **qwen3.6-35B-A3B and gpt-oss-120b** — statistically indistinguishable at
   every difficulty level (largest gap 1.7 pp, p ≥ 0.61). They should be reported
   as tied, not ranked.
2. **gemma-4-31B-it and gemma-4-26B-A4B-it** — consistently below tier 1, and
   separable from each other only on `scaled_2comp`.
3. **DeepSeek-Coder-V2-Lite** — separated from every other model at p < 0.001 on
   every set, scoring zero on 402 two-component tasks.

---

## 3. Degradation with compositional depth

Every model degrades monotonically as the number of required components grows,
and the collapse is steepest on the larger boards:

| Model | 1 comp | 2 comp | larger boards | 1→2 comp | 2 comp→scaled |
|---|---|---|---|---|---|
| qwen3.6-35B-A3B | 45.6 % | 40.3 % | 11.3 % | −5.3 pp | −29.0 pp |
| gpt-oss-120b | 44.9 % | 38.6 % | 11.8 % | −6.4 pp | −26.7 pp |
| gemma-4-31B-it | 39.8 % | 33.3 % | 5.8 % | −6.5 pp | −27.5 pp |
| gemma-4-26B-A4B-it | 36.8 % | 21.6 % | 3.9 % | −15.2 pp | −17.7 pp |
| DeepSeek-Coder-V2-Lite | 5.8 % | 0.0 % | 1.4 % | −5.8 pp | +1.4 pp |

Adding a second required component costs 5–6 pp for the three strongest models
but 15 pp for gemma-4-26B-A4B-it. Moving to 13×13 and 15×15 boards costs a
further 27–29 pp for every model in tiers 1 and 2 — the dominant difficulty
factor is board size, not component count.

---

## 4. Two distinct failure regimes

The most robust qualitative finding is not the rate but *how* the models fail.
`ceil` counts failures whose turn count equals the 25-turn budget — runs that
were still working when the budget expired, as opposed to runs that committed to
a wrong answer and stopped.

| Model | `scaled_1comp` | `scaled_2comp` | `scaled` | Median turns to failure |
|---|---|---|---|---|
| gemma-4-26B-A4B-it | 210/273 (**77 %**) | 297/315 (**94 %**) | 774/973 (**80 %**) | 25 |
| gemma-4-31B-it | 174/260 (**67 %**) | 247/268 (**92 %**) | 734/954 (**77 %**) | 25 |
| gpt-oss-120b | 0/238 (**0 %**) | 31/247 (13 %) | 26/893 (3 %) | 10–13 |
| qwen3.6-35B-A3B | 4/235 (2 %) | 27/240 (11 %) | 45/899 (5 %) | 9–13 |
| DeepSeek-Coder-V2-Lite | 45/407 (11 %) | 61/402 (15 %) | 80/999 (8 %) | 8–10 |

Two regimes separate cleanly:

- **Budget exhaustion (both Gemma models).** Two thirds to 94 % of failures sit
  exactly on the turn ceiling. These models keep acting until they are stopped.
  Their scores are therefore a lower bound conditioned on the 25-turn budget, and
  the budget is an *active experimental variable* for them, not headroom.
- **Premature convergence (gpt-oss, qwen3.6).** Failures terminate at a median of
  9–13 turns, well inside the budget. On `scaled_1comp` gpt-oss-120b hit the
  ceiling in **zero** of 238 failures. These models decide they are finished and
  are wrong; more turns would not help them.

The distinction matters for interpretation: the two groups achieve comparable
rates by different routes, and only one of them would benefit from a larger
action budget.

---

## 5. Failure mode taxonomy

Error strings normalised into families (coordinates and counts replaced by
placeholders), `scaled_2comp`:

| Family | gemma-4-31B-it | gpt-oss-120b | qwen3.6-35B-A3B |
|---|---|---|---|
| Incomplete path (*illegal free fall*) | 221 (82 %) | 206 (83 %) | 212 (88 %) |
| **Complete but wrong output** (*expected sequence*) | 2 (0.7 %) | **32 (13 %)** | **27 (11 %)** |
| No solution submitted | 42 (16 %) | 4 (1.6 %) | 1 (0.4 %) |
| Turn budget exhausted | 25 (9 %) | 1 (0.4 %) | 6 (2.5 %) |

The dominant failure everywhere is an **incomplete path**: the marble traverses a
cell the model never filled. But the second row separates the tiers. gpt-oss and
qwen3.6 produce a *complete, simulable* board that yields the wrong marble
sequence in 11–13 % of their failures — a qualitatively more advanced failure
than not finishing at all — against 0.7 % for gemma-4-31B-it, a factor of 17.

---

## 6. Computational cost

Accuracy is not the only axis on which these models differ:

| Model | tokens / task | median latency / task | Relative cost at equal accuracy |
|---|---|---|---|
| gpt-oss-120b | 48 k – 82 k | 21–27 s | 1× |
| qwen3.6-35B-A3B | 183 k – 284 k | **335–436 s** | ~15× |
| gemma-4-31B-it | 97 k – 142 k | 12–14 s | — |
| gemma-4-26B-A4B-it | 109 k – 121 k | 11–12 s | — |
| DeepSeek-Coder-V2-Lite | 9 k – 10 k | 34–43 s | — |

qwen3.6-35B-A3B and gpt-oss-120b are statistically tied on accuracy (§2), but
qwen3.6 consumes roughly 3.5× the tokens and **15× the wall-clock time per
task**. It is an explicit reasoning model and emits its deliberation into the
response, which the harness stores verbatim — a single `scaled` report is 120 MB.
On a fixed compute budget gpt-oss-120b dominates it outright.

DeepSeek's profile is the inverse and diagnostic of its failure: it emits an
order of magnitude fewer tokens than any other model while taking longer per task
than gpt-oss, consistent with short, quickly-abandoned attempts.

---

## 7. Excluded runs

Two runs are reported as excluded rather than as scores of zero, because neither
measured model capability:

| Run | Observed | Cause |
|---|---|---|
| `google/gemma-4-31B` | 0 % across 21 tasks, **0 turns**, 0 tokens | Base (non-instruct) checkpoint ships no chat template; vLLM answered every request with HTTP 400. |
| `openai/gpt-oss-120b` (first attempt) | 0 % across 21 tasks, **0 turns** | The tiktoken/harmony vocabulary is fetched over the network at request time; compute nodes have no route out, so every request returned HTTP 500. Fixed by pre-populating the cache on the login node; the model then completed all sets and appears in §2. |

Both share a signature worth naming: **an exit status of 0, a complete-looking
report, and zero generated tokens.** The analysis tooling now drops any report
whose tasks all record zero turns rather than plotting it as a 0 % score.

`Qwen/Qwen2.5-Coder-7B-Instruct` reached `TIMEOUT` after 12 h without completing
any scaled set and is absent from §2. It is the only model whose native context
(32 768) falls below the 131 072 target, so it ran with YaRN at factor 4; it is
therefore not directly comparable to the other five even once re-run.

---

## 8. Threats to validity

Each item below states what was found, what has been changed in the harness, and
what still has to be re-run before the numbers in §2 can be treated as final.

### 8.1 Stochastic sampling — cause found, fixed, re-run required

`LLMConfig.temperature` defaults to **0.7** and the benchmark CLI never set it,
so every run reported here sampled stochastically. The magnitude was measured
incidentally: gemma-4-31B-it scored 80 % / 60 % / 0 % on the small sets and, on
an identical re-run, 100 % / 80 % / 9.1 %. At n = 5 that is three single-task
flips, which is why the scaled sets are used for every claim — but the variance
is unquantified at n ≈ 400–1000.

A `--temperature` flag now exists and defaults to **0.0**. The numbers in §2 were
produced before it and are therefore single samples from a stochastic process.
**They must be regenerated under greedy decoding.** Note that vLLM with
continuous batching remains non-deterministic — batch composition changes
floating-point reduction order — so a small residual variance will persist and
repeated runs are still the honest way to report a headline figure.

### 8.2 The `turns` metric — cause found, fixed, affects §4

`turns` was recorded as `len(tool_calls)`. That is not a turn count: a model
emitting several tool calls in one assistant message inflates it past the budget,
which is why DeepSeek shows 213, 225, 276 and 444 against a limit of 25. Models
that emit one call per turn coincidentally matched, which is why the field looked
correct everywhere else.

It is now `len(turn_logprobs)` — one entry per API call, i.e. per agent-loop
iteration — with a `turns_source` field recording which definition was used. The
two-regime split in §4 (0–15 % against 67–94 %) is far too large to be an artefact
of this, but the exact `ceil` fractions for DeepSeek should not be quoted from the
current data.

### 8.3 Inventory violations are intermediate events — resolved

Models repeatedly attempt to place component types they have already exhausted;
the executor rejects the action with an informative message and the agent
continues. These consume turns and feed the budget-exhaustion regime, but they
are **not** the recorded failure mode of any task — the terminal outcome is
almost always an incomplete path.

An earlier draft of this analysis conflated the two by counting raw error strings
across a whole report, which overstated inventory violations as a terminal cause.
`inspect_results.py --turn-errors` now reports them separately, per occurrence
and per affected task. §5 counts terminal outcomes only.

### 8.4 Resource state is declared once and never observable — ablations available

The available-parts inventory is rendered into the initial prompt
(`prompts.py:94`) with zero-count types omitted, the prompt is built once before
the agent loop starts (`runner.py:718`), and `get_board_state` does not report
parts. The agent must therefore infer unavailability from absence and track its
own consumption across up to 25 turns. The benchmark consequently measures
working memory over a resource constraint alongside planning.

Two opt-in ablations now isolate the two contributions, both off by default so
existing results remain the baseline:

| Flag | Removes |
|---|---|
| `--declare-zero-parts` | The inference step: zero-count types are listed explicitly rather than omitted. |
| `--observable-inventory` | The memory step: `get_board_state` reports the remaining inventory. |

Running `scaled_2comp` in the four cells of that 2×2 decomposes the failure rate
into inference, memory and residual reasoning. This is the single most
informative follow-up experiment available and costs roughly four hours per model.

### 8.5 Composition of the `scaled` set — resolved

The job script described this set as "variants + insight + unsolvable", which
would have put a ceiling below 100 % on the rates in §3. Inspection of all 2 121
task files shows otherwise: every file carries a `solution` object with
`placed_components`, and the naming (`..._scl6_var_1`) identifies them as scaled
variants of the official challenges. **There are no unsolvable tasks**, and the
comment in the job script is wrong.

Two caveats remain. 18 of 2 121 files (0.85 %) carry a reference solution with an
empty component list. And the `verified` flag is `False` on every file — but it
is also `False` on all 58 official challenges, so it is dead metadata rather than
a signal that this set is less trustworthy than the others.
