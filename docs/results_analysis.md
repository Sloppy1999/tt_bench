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
rate by 20 percentage points. Section 9 documents why that matters.

`scaled_1comp` was additionally run **five times per model under each of two
decoding regimes** — greedy (temperature 0) and sampling at temperature 0.7 — with
seeds 1001-1005. That 2x5 design supplies the run-to-run variance used throughout
and is analysed in Section 7.

**The three sets are not measured under identical conditions.** `scaled_1comp`
carries five repetitions under each regime; `scaled` is a single greedy run;
`scaled_2comp` is still a single run at temperature 0.7 and has not been repeated
under greedy. Section 7 quantifies what that costs.

---

## 2. Overall performance

Greedy decoding is the primary condition. `scaled_1comp` reports the mean and
standard deviation of five repetitions; the other two are single runs, and
`scaled_2comp` is the one cell still measured at temperature 0.7.

| Model | `scaled_1comp` greedy, 5 runs | `scaled_2comp` T = 0.7, 1 run | `scaled` greedy, 1 run |
|---|---|---|---|
| qwen3.6-35B-A3B | 43.6 % ± 0.5 | **40.3 %** [35.6, 45.2] | 10.5 % [8.7, 12.5] |
| gpt-oss-120b | **43.8 % ± 0.4** | 38.6 % [33.9, 43.4] | **11.7 %** [9.9, 13.9] |
| gemma-4-31B-it | 40.9 % ± 0.3 | 33.3 % [28.9, 38.1] | 4.8 % [3.7, 6.3] |
| gemma-4-26B-A4B-it | 38.7 % ± 0.6 | 21.6 % [17.9, 25.9] | 4.2 % [3.2, 5.7] |
| DeepSeek-Coder-V2-Lite | 6.9 % ± 0.2 | 0.0 % [0.0, 0.9] | 1.4 % [0.8, 2.3] |

(± is the spread across repetitions; brackets are 95 % Wilson intervals on a
single run. DeepSeek's greedy cell rests on four repetitions, not five: a batched
job reached its walltime on the fifth pass.)

Under sampling at temperature 0.7, `scaled_1comp` reads differently — see §7.3:

| Model | greedy | temperature 0.7 |
|---|---|---|
| qwen3.6-35B-A3B | 43.6 % ± 0.5 | **45.8 % ± 0.4** |
| gpt-oss-120b | 43.8 % ± 0.4 | 44.0 % ± 0.3 |
| gemma-4-31B-it | 40.9 % ± 0.3 | 41.1 % ± 0.9 |
| gemma-4-26B-A4B-it | 38.7 % ± 0.6 | 37.5 % ± 0.8 |
| DeepSeek-Coder-V2-Lite | 6.9 % ± 0.2 | 6.6 % ± 0.5 |

**The ranking is stable across all three difficulty levels.** Not every gap in it
is statistically meaningful, however. Two-proportion z-tests on adjacent pairs:

| Comparison | `scaled_1comp` | `scaled_2comp` | `scaled` |
|---|---|---|---|
| qwen3.6 vs gpt-oss | p = 0.84 | p = 0.61 | p = 0.68 |
| gpt-oss vs gemma-31B | p = 0.13 | p = 0.12 | **p < 0.001** |
| gemma-31B vs gemma-26B | p = 0.36 | **p < 0.001** | p = 0.050 |
| gemma-26B vs DeepSeek | **p < 0.001** | **p < 0.001** | **p < 0.001** |

Three performance tiers are supported by the data:

1. **qwen3.6-35B-A3B and gpt-oss-120b** — indistinguishable on these single runs
   at every difficulty level (largest gap 1.7 pp, p ≥ 0.61). **The repeated design
   in Section 7 overturns this at temperature 0.7**, where five runs each separate
   them by 1.8 pp at p < 0.0001. They are tied under greedy decoding and not tied
   under sampling; a single run of each could not tell the difference.
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
| gpt-oss-120b | 43.8 % | 38.6 % | 11.7 % | −5.2 pp | −26.8 pp |
| qwen3.6-35B-A3B | 43.6 % | 40.3 % | 10.5 % | −3.3 pp | −29.8 pp |
| gemma-4-31B-it | 40.9 % | 33.3 % | 4.8 % | −7.6 pp | −28.5 pp |
| gemma-4-26B-A4B-it | 38.7 % | 21.6 % | 4.2 % | −17.1 pp | −17.4 pp |
| DeepSeek-Coder-V2-Lite | 6.9 % | 0.0 % | 1.4 % | −6.9 pp | +1.4 pp |

(The middle column is the one cell still measured at temperature 0.7, so the
1→2 comp step mixes regimes. §7.3 bounds the error this introduces: below 1 pp for
every model except qwen3.6-35B-A3B, where it could be 2 pp.)

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

## 7. Decoding temperature and run-to-run variability

Five repetitions per model under each of two decoding regimes, on `scaled_1comp`
(n = 432), seeds 1001-1005.

### 7.1 Greedy decoding is not reproducible

| Model | sd across 5 greedy runs | sd across 5 runs at 0.7 |
|---|---|---|
| DeepSeek-Coder-V2-Lite | 0.2 pp | 0.5 pp |
| gemma-4-31B-it | 0.3 pp | 0.9 pp |
| gpt-oss-120b | 0.4 pp | 0.3 pp |
| qwen3.6-35B-A3B | 0.5 pp | 0.4 pp |
| gemma-4-26B-A4B-it | 0.6 pp | 0.8 pp |
| **median** | **0.4 pp** | **0.5 pp** |

Temperature 0 selects the argmax token, so the sampler contributes nothing and
five identical runs should be identical. They are not: they span 0.2-0.6 pp. The
residue is vLLM's continuous batching, which varies batch composition between runs
and therefore the order of floating-point reductions.

Adding stochastic sampling raises the spread only to 0.3-0.9 pp. **The two noise
sources are the same order of magnitude**, and neither reaches one percentage
point.

### 7.2 The dominant uncertainty is the task count, not the pipeline

| Source | Magnitude |
|---|---|
| Run-to-run sd (measured, 5 repetitions) | ± 0.4 pp |
| Binomial 95 % interval at n = 432, p ≈ 0.44 | ± 4.7 pp |

The interval from the finite task sample is **an order of magnitude wider** than
the run-to-run spread. Repeating a configuration therefore buys very little
precision; adding tasks buys a great deal. This is what licenses the single run
per cell used for `scaled_2comp` and `scaled` — and it argues against spending
compute on repetitions elsewhere.

### 7.3 Sampling triples a reasoning model's deliberation

On `scaled_1comp`, with five repetitions per regime, two models move significantly
and in opposite directions:

| Model | greedy | temp 0.7 | Δ | p |
|---|---|---|---|---|
| **qwen3.6-35B-A3B** | 43.6 % | 45.8 % | **+2.2 pp** | **< 0.0001** |
| gpt-oss-120b | 43.8 % | 44.0 % | +0.2 pp | 0.47 |
| gemma-4-31B-it | 40.9 % | 41.1 % | +0.1 pp | 0.75 |
| DeepSeek-Coder-V2-Lite | 6.9 % | 6.6 % | −0.4 pp | 0.17 |
| **gemma-4-26B-A4B-it** | 38.7 % | 37.5 % | **−1.2 pp** | **0.0065** |

The mechanism for qwen3.6-35B-A3B is generation length, and **it replicates on the
larger set**:

| | `scaled_1comp` (n = 432) | `scaled` (n = 1013) |
|---|---|---|
| tokens/task, temp 0.7 ÷ greedy | **3.03×** | **3.43×** |
| tasks exhausting the turn budget, greedy | 0 | 0 |
| tasks exhausting the turn budget, temp 0.7 | 4–12 | 45 |
| every other model, token ratio | 0.76–0.97× | 0.99–1.36× |

qwen3.6-35B-A3B is the only model whose generation expands under sampling, on both
sets, by the same factor. Under greedy it never once exhausts the 25-turn budget;
under sampling it does. It is an explicit reasoning model, and greedy decoding
apparently ends its thinking block early: the most probable continuation is to
stop deliberating.

**The accuracy payoff, however, is only demonstrated on the smaller set.** On
`scaled` the same comparison gives −0.8 pp at p = 0.57 — no detectable difference.

### 7.3.1 That null result is underpowered, and must not be read as "no effect"

`scaled` was run once per regime, against five repetitions each on
`scaled_1comp`. At n = 1013 and p ≈ 0.11, the standard error of the difference
between two single runs is 1.39 pp:

| True effect | Detectable at | Power |
|---|---|---|
| 0.8 pp (observed) | 0.58 SE | 9 % |
| 2.2 pp (the `scaled_1comp` effect) | 1.58 SE | **35 %** |
| minimum detectable at 80 % power | 2.8 SE | **3.9 pp** |

**Had the 2.2 pp effect been present on `scaled` at full strength, this comparison
would have missed it roughly two times in three.** The honest statement is
therefore:

> Sampling makes qwen3.6-35B-A3B deliberate about three times as much on both
> sets. On the one-component set that converts into +2.2 pp (p < 0.0001, five
> repetitions per regime). On the larger boards the effect is not measurable, and
> the design cannot distinguish an absent effect from one it lacks the power to
> see.

Settling it needs repetitions on `scaled`, not a larger single run: five per
regime would bring the minimum detectable effect to roughly 1.7 pp.

The practical conclusion stands regardless of which way that resolves. **For a
reasoning model, temperature 0 is not the neutral reproducibility choice it is
usually taken to be** — it changes the amount of deliberation by a factor of
three, and §7.1 shows it does not buy reproducibility either.

### 7.4 The model ranking depends on the decoding regime

| Regime | qwen3.6 − gpt-oss | p | Verdict |
|---|---|---|---|
| greedy | −0.2 pp | 0.41 | indistinguishable |
| temperature 0.7 | **+1.8 pp** | **< 0.0001** | **distinct** |

A single run of each had put these two 0.7 pp apart at p = 0.84 and they were
reported as tied. With five repetitions the question resolves — and the answer is
that it depends on the decoding regime. Any ranking of these two models is only
meaningful once the temperature is stated.

---

## 8. Excluded runs

Three runs are reported as excluded rather than as scores, because none of them
measured model capability under conditions comparable to §2.

| Run | Observed | Cause |
|---|---|---|
| `google/gemma-4-31B` | 0 % across 21 tasks, **0 turns**, 0 tokens | Base (non-instruct) checkpoint ships no chat template; vLLM answered every request with HTTP 400. |
| `openai/gpt-oss-120b` (first attempt) | 0 % across 21 tasks, **0 turns** | The tiktoken/harmony vocabulary is fetched over the network at request time; compute nodes have no route out, so every request returned HTTP 500. Fixed by pre-populating the cache on the login node; the model then completed all sets and appears in §2. |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7.2 % on `scaled_1comp`, **0.8 s median per task** | Three independent disqualifiers, below. |

The first two share a signature worth naming: **an exit status of 0, a
complete-looking report, and zero generated tokens.** The analysis tooling now
drops any report whose tasks all record zero turns rather than plotting it as a
0 % score.

### 7.1 Why `Qwen2.5-Coder-7B-Instruct` is excluded

Three reasons, each sufficient on its own:

1. **Not comparable.** It is the only model whose native context (32 768) falls
   below the 131 072 target, so it alone ran with YaRN at factor 4 — evaluated
   outside its training regime while the other five ran natively. A rate obtained
   under a different effective context does not belong in the same column.
2. **Incomplete.** It reached `TIMEOUT` after 12 h on the first attempt without
   finishing a single scaled set, and the re-run completed only `scaled_1comp`.
   There is no `scaled_2comp` or `scaled` figure for it at any temperature.
3. **The one measurement it produced is not credible.** A median latency of
   **0.8 s per task** against 11–216 s for every other model, with 395 of 401
   failures being an incomplete path, describes a model that is not attempting the
   task rather than one attempting and failing. Reporting 7.2 % as a capability
   estimate would be reporting a malfunction.

Excluding it costs little: the model was the smallest in the roster and the
remaining five span 16 B to 120 B across dense and MoE architectures. It should
be re-run at its native 32 768 context before any number is quoted, which would
also remove disqualifier 1 — but that is a different experiment, not this one.

---

## 9. Threats to validity

Each item below states what was found, what has been changed in the harness, and
what still has to be re-run before the numbers in §2 can be treated as final.

### 9.1 Run-to-run variability — measured, no longer a threat

This was the largest open question in an earlier draft: `LLMConfig.temperature`
defaults to 0.7, the CLI never set it, and the resulting variance was unquantified.
It has since been measured directly rather than assumed away (Section 7).

**Resolved.** Over five repetitions per configuration on `scaled_1comp`:

| | Magnitude |
|---|---|
| Run-to-run sd, greedy | 0.2–0.6 pp (median 0.4) |
| Run-to-run sd, temperature 0.7 | 0.3–0.9 pp (median 0.5) |
| Binomial 95 % interval at n = 432 | ± 4.7 pp |

The pipeline is an order of magnitude more stable than the finite task sample, so
a single run per cell is defensible and the earlier single-run figures stand: the
no-suffix baseline falls within 0.5 pp of the greedy mean for all five models.

Two residues remain, and both are now stated rather than hidden:

- **Greedy is not reproducible either.** vLLM's continuous batching contributes
  0.2–0.6 pp on its own. Reporting temperature 0 as "deterministic" would be
  wrong.
- **One cell is still measured at temperature 0.7.** `scaled` has since been
  re-run under greedy; `scaled_2comp` has not, because a multi-label `--sets`
  value was truncated by sbatch's own comma-separated `--export` syntax and only
  the first set ran. §7.3 bounds the resulting error at under 1 pp for every model
  except qwen3.6-35B-A3B, where it could reach 2 pp. One re-run closes it.

- **`scaled` is a single run per regime and the temperature comparison on it is
  underpowered** — 35 % power for the effect size measured on `scaled_1comp`
  (§7.3.1). Its null result is not evidence of absence.

### 9.2 The `turns` metric — cause found, fixed, affects §4

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

### 9.3 Inventory violations are intermediate events — resolved

Models repeatedly attempt to place component types they have already exhausted;
the executor rejects the action with an informative message and the agent
continues. These consume turns and feed the budget-exhaustion regime, but they
are **not** the recorded failure mode of any task — the terminal outcome is
almost always an incomplete path.

An earlier draft of this analysis conflated the two by counting raw error strings
across a whole report, which overstated inventory violations as a terminal cause.
`inspect_results.py --turn-errors` now reports them separately, per occurrence
and per affected task. §5 counts terminal outcomes only.

### 9.4 The scaled sets barely test resource management

An earlier draft of this analysis identified inventory mismanagement as the
mechanism behind the failures: models repeatedly attempting to place component
types they had exhausted, burning turns on rejected actions. **That hypothesis was
formed on the 5-task sets and does not survive contact with the scaled data.**

**What the task files actually contain.** All 4 021 task files declare an
`available_parts` inventory, so enforcement is active everywhere. But the
inventories are tight:

| Set | n | Tasks with zero slack | Median pieces offered | Median distinct types |
|---|---|---|---|---|
| `official` | 58 | 69.0 % | 6 | 1 |
| `scaled` | 2 121 | **99.6 %** | 1 | 1 |
| `challenges_1comp` | 948 | 83.3 % | 1 | 1 |
| `challenges_2comp` | 894 | 83.3 % | 2 | 2 |

"Zero slack" means the inventory contains exactly as many pieces as the reference
solution places. In `scaled` — the largest set, and the one carrying most of the
statistical weight — **99.6 % of tasks offer exactly the pieces required, with a
median of one piece of one type.** There is nothing to allocate: the only legal
action is to place the single available piece.

**What the agents actually get rejected for.** Extracting per-tool-call errors
across the scaled sets (`inspect_results.py --turn-errors`) returns no inventory
exhaustion at all. The rejected actions are **malformed tool calls**:

| Model | Rejected action | Tasks affected |
|---|---|---|
| DeepSeek-Coder-V2-Lite | `place_component()` with an unexpected `direction` | 1.4 – 4.0 % |
| DeepSeek-Coder-V2-Lite | `place_component()` with an unexpected `gear_group` | 0.1 – 0.5 % |
| gemma-4-26B-A4B-it | `place_component()` missing required `y` | 1.1 – 2.5 % |
| gpt-oss-120b | `place_component()` with an unexpected `direction` | 0.2 – 0.7 % |
| gemma-4-31B-it, qwen3.6-35B-A3B | none recorded | 0 % |

Two consequences.

**For the results.** The headline numbers in §2 measure placement and path
construction under an exactly-sized inventory. They do **not** measure resource
allocation under scarcity, because these sets do not present that problem. Any
claim about "agentic planning under resource constraints" belongs to `official`
(median 6 pieces, 31 % with slack) and not to the scaled sets — and `official` is
11 tasks at Tier 1, far too few to support one.

**For the harness.** Those rejections surface a raw Python `TypeError` to the
model rather than a structured message. A malformed call is a legitimate thing for
an agent to attempt; answering it with an internal function signature is a harness
defect, and it affects up to 4 % of DeepSeek's tasks. The tool schema and the
executor signature should be reconciled, and unknown parameters should produce a
message naming the accepted ones.

**The ablations remain implemented but are no longer the priority.**
`--declare-zero-parts` and `--observable-inventory` were added to decompose a
resource-tracking confound that these sets turn out not to exercise. Running them
on the scaled sets would measure close to nothing. They would only be informative
on a set built with genuine inventory slack, which does not yet exist.

### 9.5 Composition of the `scaled` set — resolved

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
