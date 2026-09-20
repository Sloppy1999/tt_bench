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

> **Read this section together with §11.3.** The `1 comp` and `2 comp` columns
> above are aggregate success rates on `scaled_1comp` and `scaled_2comp`. In the
> current corpus those two sets are about two-thirds *unsolvable variants*,
> where success means declaring the task unsolvable rather than building a
> board — so an aggregate rate over them blends two different abilities, and
> §11.3 shows that blending reverses the model ranking. Whether the corpus
> snapshot these particular numbers were computed on had the same composition
> has **not** been re-verified (it predates the corpus growth described in §13,
> and two of the rates here sit above the ceiling that today's composition would
> impose, which suggests it did not). The figures above are therefore left as
> recorded, and the compositional-depth claim should be taken as provisional
> until it is recomputed on the corrected corpus with the two halves reported
> separately.

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

---

## 10. Dataset scoring-contract defects (found and fixed after §1–9)

The analysis above was written against a scorer with a defect that was found and
fixed afterward. It is documented here rather than silently folded into revised
numbers, because it changed what counted as a passing task, not merely how a task
was measured.

### 10.1 `required_output` was compared against the wrong reading of the run

The benchmark declares two sequence-shaped targets. `solution.final_marble_state`
names the **catcher** each marble reached (left → blue, right → red). But an
official-challenge board can send a blue ball out the *right* side, so the guide's
printed strip — `required_output`, the colour of each ball as it leaves — is not
always the same sequence. Both the dataset-side check and the benchmark scorer
compared `required_output` against the catcher-derived sequence, which rejects a
correct board whenever a ball's colour and its exit side disagree.

Independently of this session, a concurrent dataset-certification pass over the
full corpus quantified the damage precisely: **1 819 of 1 842 (98.7 %) "reference
solution rejected" quarantine reasons cited `required_output`.** The guide's own
solutions — ground truth, not model output — were being marked wrong by the
scorer that was supposed to check them. After each target was compared against
its own correct reading of the run (`final_marble_state` against catchers,
`required_output` against ball colour), that count fell to 20 on the same corpus,
and the remaining 20 are genuine data issues, not a contract mismatch.

Official-challenge verification: **16/58 → 38/58** boards. Tier 2 alone, the set
used throughout §11 below, is now fully verified: **22/22.**

### 10.2 Trigger levers are bars, not single cells

The simulator's catcher check required a marble to land on the *exact* column of
a lever. On the physical board a lever is a bar covering its half of the bottom
row; on `ch09-pA` this meant a red ball landing at column 3 and a blue ball
landing at column 4 *or* 6 could never both be caught, because no single column
serves all three. The catcher was rewritten as a span, with the return chute at
the midpoint **between the two lever columns** rather than at the middle of the
board — the two coincide only on an unpadded board, and a first attempt using the
board's midpoint silently broke 477 width-padded tasks whose levers stay where
they were while the board grows. Caught by re-verifying the whole corpus before
and after, not by inspection.

### 10.3 Bit-state objectives were recorded but never scored

Two challenges (`ch11`, `ch11-pA`) ask to "flip bits 2 and 5 to the right." Their
target bit configuration was written into `solution.final_bit_state`, a field
nothing reads; scoring saw only that both balls reached a catcher and accepted
any board that did, regardless of the bits. Moved to
`expected_output.final_bit_states`, the field the shared target contract actually
checks; a board that reaches the same catchers with the wrong bits is now
rejected. The repo's own recorded objective for `ch11` — "flip bits 1 and 4" — was
also wrong; the guide (page 45) asks for bits 2 and 5, confirmed by simulating the
transcribed board and reading which two bits it actually leaves flipped.

### 10.4 Net effect

Three previously-broken tier-2 boards (`ch09-pA`, `ch11`, `ch11-pA`) were
re-transcribed once §10.1–10.3 were fixed, using ball counts and objective text
read from the guide PDF rather than trusted from the task file (the guide prints
`×10`/`×10` hoppers for `ch09-pA` and `×2`/`×0` for `ch11`; the task file had
hard-coded 8/8, which makes a zero-red-ball puzzle unsolvable outright). **Tier 2
official challenges are now 22/22 verified**, and every result quoted in §11
below is measured against that fully-verified set.

---

## 11. Tier 2, the corrected corpus, and a matched tier-1 re-run

### 11.1 What changed since §1–9

- **Tier 2 exists in this analysis for the first time.** §1–9 covered Tier 1
  only. All 22 Tier-2 official challenges now verify (§10), so a same-day,
  same-stack Tier-1/Tier-2 comparison is possible for the first time.
- **A sixth model was added: `qwen3.8-flash-next-awq4`**
  (`cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4`). The originally-requested GGUF build
  (`unsloth/Qwen3.8-Flash-Next-GGUF`) could not be used on any installable vLLM
  release — 0.24.0 and 0.29.0 were both checked, and neither registers a `gguf`
  quantization method. The model's architecture (`Qwen4ExpForConditionalGeneration`)
  is also unknown to vLLM 0.24.0. This is the AWQ INT4 (`compressed-tensors`,
  `pack-quantized`) repack of the same weights, ~176 GB on disk against
  4 × 93.6 GiB of H100 memory, running under vLLM 0.29.0.
- **vLLM was upgraded 0.24.0 → 0.29.0** to serve the new model. This changes the
  serving stack under all seven models, not only the new one, so its effect was
  checked directly rather than assumed away: `gemma-4-31B-it` was run four times
  on the same 22 Tier-2 official boards at temperature 0 — three times on 0.24.0
  (6, 4, 5 solved) and once on 0.29.0 (4 solved). The 0.29.0 result sits inside
  the 0.24.0 spread; **the version bump is not a detectable confound at this
  sample size,** though nothing here rules out a smaller effect than four single
  runs can resolve.
- **A harness startup bug was found and fixed in the same investigation.** Every
  4-GPU (tensor-parallel) model on vLLM 0.29.0 initially failed to start: the
  script's 600 s readiness wait expired while vLLM was still compiling, because
  0.29.0 attempts a FlashInfer all-reduce workspace this node topology does not
  support (no NVSwitch multicast) and falls back to a slower path mid-init. Two
  models that would have failed under the old limit — `qwen2.5-coder-7b` (ready
  at 840 s) and `gemma-4-31B-it` (ready at 920 s) — came up cleanly once the wait
  was raised to 1 800 s.

### 11.2 Overall performance, both tiers

Single run per cell, temperature 0, 25-turn budget — **not** the five-repetition
design of §7. §11.5 states exactly what that limits.

> Every figure in §11 is generated from `results_summary.json` and
> `unsolvable_breakdown.json` by `jureca/plot_summary.py`, so the figures and the
> tables beside them are computed from the same two files and cannot drift
> apart. Regenerate with `uv run python jureca/plot_summary.py`. The image files
> themselves are not version-controlled — this repository's `.gitignore`
> excludes `*.png`/`*.svg` — so run that command once after a fresh clone to
> render them.

Only the four sets in which **every task is solvable** appear here.
`scaled_1comp` and `scaled_2comp` are roughly two-thirds unsolvable variants and
their aggregate rate is not a like-for-like measure of the same thing; they are
reported separately in §11.3.

| Model | T1 official (11) | T1 1comp (5) | T1 2comp (5) | T1 scaled (1397) | T2 official (22) | T2 1comp (13) | T2 2comp (13) | T2 scaled (2170) |
|---|---|---|---|---|---|---|---|---|
| **qwen3.8-flash-next-awq4** | **90.9 %** | 100.0 % | 100.0 % | **99.6 %** | **63.6 %** | 100.0 % | 100.0 % | **98.3 %** |
| gemma-4-31B-it | 36.4 % | 80.0 % | 80.0 % | 92.0 % | 18.2 % | 100.0 % | 100.0 % | 79.0 % |
| qwen3.6-35B-A3B | 27.3 % | 100.0 % | 60.0 % | 90.2 % | 22.7 % | 92.3 % | 92.3 % | 82.5 % |
| gpt-oss-120b | 45.5 % | 80.0 % | 80.0 % | 83.8 % | 22.7 % | 76.9 % | 69.2 % | 75.5 % |
| gemma-4-26B-A4B-it | 27.3 % | 80.0 % | 60.0 % | 77.8 % | 18.2 % | 100.0 % | 84.6 % | 65.8 % |
| Qwen2.5-Coder-7B | 0.0 % | 80.0 % | 40.0 % | 17.0 % | 0.0 % | 30.8 % | 15.4 % | 20.9 % |
| DeepSeek-Coder-V2-Lite | 0.0 % | 20.0 % | 0.0 % | 3.2 % | 0.0 % | 7.7 % | 0.0 % | 1.7 % |

![Success rate by model and challenge set, tier 1 and tier 2](assets/figures/success_matrix.png)

**Figure 1.** The table above, read as a matrix. Rows are ordered by mean rate
across the eight cells, and that order is reused in every figure below. The two
sets excluded from this figure are the subject of §11.3.

**The difficulty ladder now runs the right way, with one exception worth
naming.** On `scaled` — the set that carries the weight, at 1 397 and 2 170
tasks — six of the seven models score lower on Tier 2 than Tier 1. The seventh,
`Qwen2.5-Coder-7B`, moves the other way (17.0 % → 20.9 %). On `official`, five of
seven drop and the remaining two score **0.0 % on both tiers**, so they order
nothing. The honest summary is therefore *not* that every model finds Tier 2
harder: it is that every model scoring meaningfully above the floor does.

![Tier 1 to Tier 2 success rate by model](assets/figures/tier_gap.png)

**Figure 2.** Tier 1 and Tier 2 for each model on the two sets with no
unsolvable variants. Where only one dot is visible the two tiers are equal
(`Qwen2.5-Coder-7B` and `DeepSeek-Coder-V2-Lite` on `official`, both 0.0 %).

This direction is itself the result of §10. The same models scored *higher* on
the broken Tier-2 corpus in July, because the scorer was rejecting Tier 2's
genuinely-correct solutions more often than Tier 1's — not because Tier 2 was
easier.

**`qwen3.8-flash-next-awq4` leads or ties for the lead in every column.** The
margin is wide on `official` — 45.5 pp over `gpt-oss-120b` on Tier 1, 40.9 pp
over `qwen3.6-35B-A3B` on Tier 2 — and much narrower on `scaled`, at 7.7 pp and
15.8 pp over the runner-up, where the leaders are all compressed against the top
of the scale. On `1comp`/`2comp` it only ties: four of the seven models reach
100 % on at least one tier, which is what a 5- and 13-task set can resolve.

`DeepSeek-Coder-V2-Lite` and `Qwen2.5-Coder-7B` sit at exactly 0.0 % on
`official` on both tiers, consistent with the disengagement pattern in §6 and
§8 — low token counts, few tool calls, an attempt that does not resemble serious
engagement with the board.

### 11.3 `scaled_1comp` / `scaled_2comp` measure two abilities, not one

These two sets are **not** simply harder versions of `scaled`. Roughly two
thirds of their tasks are *unsolvable variants*, built by removing from the
inventory a component the board requires. On those, success does not mean
building anything: it means recognising that no solution exists and saying so.
The prompt asks for this in as many words — it instructs the model to determine
"whether the puzzle is solvable or unsolvable with the given inventory," and to
set `"success": false` with an explanation if it concludes the latter.

| Set | Tier | Solvable | Unsolvable | Total | Aggregate ceiling if no unsolvable task is ever detected |
|---|---|---|---|---|---|
| `scaled_1comp` | 1 | 144 | 288 | 432 | 33.3 % |
| `scaled_1comp` | 2 | 262 | 344 | 606 | 43.2 % |
| `scaled_2comp` | 1 | 134 | 268 | 402 | 33.3 % |
| `scaled_2comp` | 2 | 260 | 328 | 588 | 44.2 % |

This composition makes the aggregate success rate on these sets very easy to
misread, and the misreading is severe. Taken at face value, the aggregate says
every model except `gpt-oss-120b` collapses by 30–70 pp moving from `scaled` to
`scaled_1comp` — which invites the conclusion that completing a partial board is
dramatically harder than building one from scratch. **That conclusion would be
wrong.** Decomposing each set into its two halves shows why:

| Model | 1comp·T1 solvable | 1comp·T1 detected | 1comp·T2 solvable | 1comp·T2 detected | 2comp·T1 solvable | 2comp·T1 detected | 2comp·T2 solvable | 2comp·T2 detected |
|---|---|---|---|---|---|---|---|---|
| qwen3.8-flash-next-awq4 | 100.0 % | 0.0 % | 98.9 % | 0.0 % | 97.8 % | 0.0 % | 98.5 % | 0.0 % |
| gemma-4-31B-it | 86.8 % | 0.0 % | 93.9 % | 0.0 % | 74.6 % | 0.0 % | 86.2 % | 0.0 % |
| qwen3.6-35B-A3B | 94.4 % | 0.0 % | 87.8 % | 0.0 % | 77.6 % | 0.0 % | 77.7 % | 0.0 % |
| gpt-oss-120b | 88.2 % | **89.2 %** | 82.1 % | **79.1 %** | 64.9 % | **86.6 %** | 70.8 % | **79.3 %** |
| gemma-4-26B-A4B-it | 68.8 % | 0.0 % | 86.3 % | 0.0 % | 45.5 % | 0.0 % | 71.2 % | 0.0 % |
| Qwen2.5-Coder-7B | 29.9 % | 0.0 % | 30.5 % | 0.0 % | 6.7 % | 0.0 % | 21.9 % | 0.0 % |
| DeepSeek-Coder-V2-Lite | 11.1 % | 4.5 % | 6.5 % | 0.0 % | 0.0 % | 4.9 % | 3.5 % | 0.3 % |

![Solvable accuracy versus unsolvable detection on scaled_1comp and scaled_2comp](assets/figures/solvable_split.png)

**Figure 3.** The same numbers, with the dashed line marking where a model's
aggregate rate is pinned if it never declares a task unsolvable.

**Five of the seven models never once declared a task unsolvable** — not in any
of the four set/tier combinations, across 1 228 unsolvable tasks each. Their
aggregate score on these sets is therefore capped at the solvable fraction by
construction, and the apparent "collapse with compositional depth" is that cap,
not a loss of building skill. `qwen3.8-flash-next-awq4` makes the point
exactly: its Tier-1 `scaled_1comp` aggregate of 33.3 % is not an approximation
of the 33.3 % ceiling, it *is* the ceiling — it solved **144 of 144** solvable
tasks and **0 of 288** unsolvable ones.

**`gpt-oss-120b` is the only model that detects unsolvability at all** (79–89 %
across the four cells; `DeepSeek-Coder-V2-Lite` manages 0–5 %, which is closer to
noise than to a capability). This, and not a talent for localised repair, is the
whole of its apparent advantage on these two sets. Judged only on the tasks that
*can* be solved, it is mid-pack: on `scaled_2comp` Tier 1 it solves 64.9 % of
solvable tasks while `gemma-4-31B-it` solves 74.6 %, `qwen3.6-35B-A3B` 77.6 %,
and `qwen3.8-flash-next-awq4` 97.8 % — yet `gpt-oss-120b` has the highest
aggregate of any model on that set. An aggregate rate that inverts the ranking
of the ability it appears to measure is a good reason to report the two halves
separately, which is what §11.2 and this section now do.

Two things follow for the benchmark itself. First, declaring a task unsolvable
is a **near-binary trait** in this field rather than a graded skill: models score
either ~0 % or ~80 %, with nothing in between. Second, any headline number
computed over a corpus that mixes solvable and unsolvable tasks will rank models
mostly by whether they possess that trait — so the unsolvable variants belong in
their own reported metric, not blended into a single success rate.

### 11.4 Budget exhaustion replicates on the new corpus

`fail_at_ceiling` — failures whose turn count equals the 25-turn budget — on
Tier-2 `official` and `scaled`:

| Model | T2 official, ceiling / failed | T2 scaled, ceiling / failed |
|---|---|---|
| gemma-4-31B-it | 17 / 18 (**94.4 %**) | 367 / 456 (**80.5 %**) |
| gemma-4-26B-A4B-it | 17 / 18 (**94.4 %**) | 89 / 743 (12.0 %) |
| Qwen2.5-Coder-7B | 4 / 22 (18.2 %) | 43 / 1716 (2.5 %) |
| qwen3.8-flash-next-awq4 | 3 / 8 (37.5 %) | 1 / 37 (2.7 %) |
| qwen3.6-35B-A3B | 3 / 17 (17.6 %) | 10 / 379 (2.6 %) |
| gpt-oss-120b | 4 / 17 (23.5 %) | 6 / 531 (1.1 %) |
| DeepSeek-Coder-V2-Lite | 0 / 22 (0.0 %) | 0 / 2134 (0.0 %) |

![Share of failures reaching the 25-turn ceiling, tier 2](assets/figures/budget_exhaustion.png)

**Figure 4.** Running out of turns and getting it wrong early are different
failure modes; the success rate alone cannot tell them apart.

§4's two-regime split replicates: both Gemma models still exhaust the budget on
the overwhelming majority of `official` failures, exactly as before. On `scaled`,
however, `gemma-4-26B-A4B-it`'s ceiling share (12.0 %) is now far below
`gemma-4-31B-it`'s (80.5 %) — a divergence between the two Gemma models that §4
did not show on the smaller, earlier corpus. Whether this is a genuine capability
difference or an artefact of the corpus having grown since §4 was written (§13)
is not resolved by a single run and should not be read as a finding on its own.

### 11.5 What a single run does and does not support

This section reuses none of §7's machinery: no repeated runs, no confidence
intervals, no significance tests. What licenses treating it as informative
anyway is narrower than what §7 established for Tier 1's `scaled_1comp`:

- The only measured repeat at this session's settings is `gemma-4-31B-it` on
  Tier-2 `official` (n = 22): four runs gave 6, 4, 5, 4 solved — a spread of
  ±2 tasks (±9 pp) at this sample size, from decoding noise, batching-order
  effects, and the vLLM version change combined, not disentangled.
- No equivalent repeat exists for `scaled`/`scaled_1comp`/`scaled_2comp`
  (hundreds to thousands of tasks) under this session's settings. §7.2 showed
  run-to-run noise is an order of magnitude smaller than the sampling interval
  at n = 432; nothing here re-confirms that at the sizes used above, though it is
  the same pipeline.
- **Practical rule carried forward:** treat any gap under ~2 tasks on the small
  sets (`official`, `1comp`, `2comp`) as unresolved by a single run. The gaps
  reported in §11.2–11.4 that exceed this — `qwen3.8-flash-next-awq4`'s lead
  everywhere, `gpt-oss-120b`'s unsolvable-detection gap over every other model,
  the Gemma budget-exhaustion split — are well outside that band. The
  unsolvable-detection gap in particular is not a marginal call: it separates
  ~0 % from ~80 % on sets of 268–344 tasks.

---

## 12. Context-length and truncation audit

Two distinct signals were checked, because the run logs surfaced both a hard API
error and a softer mid-turn warning, and they carry very different weight.

**Hard `400` context-length errors are negligible.** Only `qwen2.5-coder-7b`
produces them, and only 1–2 tasks out of thousands per set. No other model in
the roster hits this at all.

**Mid-turn truncation (`finish_reason=length`, "`max_tokens=32768` may be too
low") is common for two models** and near-absent for the rest:

| Model | Truncation warnings across the sweep |
|---|---|
| qwen3.6-35B-A3B | **2 388** |
| Qwen2.5-Coder-7B | 379 |
| qwen3.8-flash-next-awq4 | 81 |
| gemma-4-26B-A4B-it | 78 |
| DeepSeek-Coder-V2-Lite | 4 |
| gpt-oss-120b, gemma-4-31B-it | 0 |

The real question is whether this measurably suppresses a score. For
`qwen3.6-35B-A3B`, checked directly: 2 295 distinct tasks showed at least one
truncation warning during their episode; of those, 914 (**39.8 %**) still
succeeded — *higher* than the ~33 % baseline rate on the comparable
`scaled_1comp`/`scaled_2comp` sets. Truncation-affected tasks are not failing
more than average. (Both figures are aggregates over sets that are ~2/3
unsolvable variants, per §11.3. The comparison is still like-for-like — the same
tasks under both readings — but it does not separate truncation's effect on
building a board from its effect on declaring a task unsolvable, and
`qwen3.6-35B-A3B` never does the latter.) The more plausible reading is that
`qwen3.6-35B-A3B`, an
explicit reasoning model (§6), emits longer responses on harder tasks it often
still solves, and the warning is a symptom of that verbosity rather than a
capability bottleneck the harness is imposing on it.

**Conclusion: the `official`/`1comp`/`2comp`/`scaled` numbers in §11 are not
meaningfully distorted by context or generation-length limits for any model in
the roster.**

---

## 13. Reconciling §1–9 with §10–12

§1–9 and §10–12 describe two different eras of this project, and their numbers
should not be read as one continuous series without the following caveats.

**The corpus has grown.** §1–9's `scaled` set held 1 013 Tier-1 tasks;
today's holds 1 397 for the same tier — a ~38 % increase from dataset generation
that happened independently of §10's contract fixes. The specific task files
behind §1–9's numbers no longer constitute the current corpus, so those numbers
cannot be reproduced against it and describe a corpus snapshot rather than the
current one.

**§1–9's statistical rigour has not been re-applied to the corrected corpus or
the expanded roster.** The five-repetition design, the Wilson intervals, the
two-proportion z-tests, and the temperature-sensitivity analysis in §7 all
belong to the Tier-1-only, five-model, vLLM-0.24.0 configuration. Re-running that
design — five repetitions per model per set, at both decoding temperatures, on
the corrected corpus, across all seven models and both tiers — is the natural
next step before the two eras of results can be merged into a single table with
one consistent standard of evidence. §11.5 states plainly what the single-run
numbers in §11 can and cannot support in the meantime.

**The unsolvable-variant composition cuts across both eras.** §11.3's finding —
that `scaled_1comp` and `scaled_2comp` are ~2/3 unsolvable variants, and that an
aggregate rate over them ranks models mostly by whether they ever declare a task
unsolvable — is a property of the *sets*, not of this session's runs. Every
earlier result computed as a bare aggregate over those two sets inherits it,
which includes §3's compositional-depth table and §7's five-repetition variance
and significance work, both of which use `scaled_1comp` as their primary set.
Nothing here says those numbers are miscomputed; it says the quantity they
compute is a blend of two abilities that §11.3 shows can rank in opposite
directions. Re-reporting them as a solvable/unsolvable split belongs in the same
re-run described above. §3 carries a note to this effect; §7 has not been
revised, because re-deciding which of its significance results survive the split
requires the per-task data from those five repetitions, not a re-reading of the
summary.

**What is not in question.** The scoring-contract defects in §10 are
independent of both the corpus-growth and the repetition-design gaps: they were
found by direct construction of a failing case (a board whose reference solution
provably matches the guide's printed output and was still rejected) and confirmed
by an independent audit process, not inferred statistically. Their fix is not
something a future re-run could contradict.
