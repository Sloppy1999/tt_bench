# Harness Ablation Plan

**Research question.** Which harness interventions improve performance, and for which models?

**Status.** Harness built and verified locally (Section 6 complete except the
per-arm figure work); no jobs submitted. Section 3 must still be committed to disk
before Section 7 starts. Section 12 records what the build changed about the plan.

---

## 1. What the existing data already answers, and what it does not

The Tier-1 results in `docs/results_analysis.md` are the starting point, not
background. Three facts from them drive the whole design.

**(a) Two failure regimes already separate the roster (§4).** `ceil` counts
failures whose turn count equals the 25-turn budget:

| Model | failures on the ceiling, `scaled_1comp` | median turns to failure | regime |
|---|---|---|---|
| gemma-4-26B-A4B-it | 210/273 (77 %) | 25 | budget exhaustion |
| gemma-4-31B-it | 174/260 (67 %) | 25 | budget exhaustion |
| gpt-oss-120b | 0/238 (0 %) | 10–13 | premature convergence |
| qwen3.6-35B-A3B | 4/235 (2 %) | 9–13 | premature convergence |
| DeepSeek-Coder-V2-Lite | 45/407 (11 %) | 8–10 | not engaging (§8 of that doc) |

This is a *mechanistic* split, and it makes the research question answerable
rather than exploratory: a model stopped by the budget and a model that stops
itself need opposite interventions. That yields signed, falsifiable, per-model
predictions (Section 3) instead of a grid to be inspected afterwards.

**(b) The precision budget is known (§7.2).** Run-to-run sd is 0.2–0.6 pp
(greedy) — an order of magnitude below the ±4.7 pp binomial interval at n = 432.
So: **do not buy repetitions, buy pairing.** Every comparison here is
within-task against a baseline that is already on disk.

**(c) The `scaled` sets do not test resource allocation (§9.4).** 99.6 % of
`scaled` tasks offer exactly the pieces the reference solution places. The
existing `--declare-zero-parts` / `--observable-inventory` ablations therefore
measure nothing on these sets and are **excluded** from this plan. Interventions
here target verification, budget and stopping behaviour — not inventory
reasoning. No claim in the write-up may drift into the latter.

**What is not answered:** whether any harness change moves the rates, and whether
the direction depends on the model. That is this experiment.

---

## 2. Scope and choice of cells

| Dimension | Value | Why |
|---|---|---|
| Models | all five from §2 of `results_analysis.md` | the failure-regime split is what is being tested; dropping a model drops an arm of the contrast |
| Task type | `agentic_synthesis` | the only type with a harness loop to ablate |
| Tier | 1 | every stored baseline is Tier 1; changing it forfeits the paired design |
| Decoding | greedy (`--temperature 0.0`) | matches the 5-rep baseline; §7.4 shows the ranking is temperature-dependent, so mixing regimes across arms would confound the intervention with decoding |
| Primary set | `scaled_1comp`, n = 432 | ~40 % baseline (headroom both ways), and **five greedy repetitions already on disk** — the only cell where an arm's effect can be checked against measured pipeline noise |
| Co-primary (budget arm only) | `scaled_sub500` (Section 6.9) | budget exhaustion is worst there (77–80 % of Gemma failures); the sharpest test of the budget prediction |
| Confirmatory | `scaled_sub500`, arms that clear Phase 1 | generalisation to larger boards without paying for 1 013 tasks × 5 models × 7 arms |

**Phase 2 carries its own baseline.** The stored `scaled` run cannot be paired
against anything drawn today — commit `a365ceb` rebuilt the corpus after that run
was made (Section 12.1). The primary set is unaffected and pairs exactly.

`scaled_2comp` is deliberately out of scope: its only run is at temperature 0.7
(§9.1) so it has no greedy baseline to pair against. Re-running it greedily is a
prerequisite of a *different* open item in that document, not of this one.

---

## 3. Pre-registered predictions

**This table is the deliverable's scoring key.** It goes into
`review/harness_ablation/predictions.md`, committed and the commit hash recorded,
**before the first job is submitted** (Section 7, step P0). An intervention plan
that reports which cells moved after seeing them answers a weaker question than
one that says in advance which cells should move and is then checked.

Arms are defined in Section 5. `++` large gain, `+` small gain, `0` no change,
`−` loss. The prediction is the sign, and `0` means "inside the ±5 pp the design
can resolve" (Section 4.3) — not "exactly zero".

| Model (regime) | `t10` | `t15` | `t40` | `fb-optional` | `fb-auto` | `rev-retry` | `rev-structured` |
|---|---|---|---|---|---|---|---|
| gemma-4-31B-it (exhaustion) | `−−` | `−` | `++` | `−` | `++` | `0` | `0` |
| gemma-4-26B-A4B-it (exhaustion) | `−−` | `−` | `++` | `−` | `++` | `0` | `0` |
| gpt-oss-120b (convergence) | `0` | `0` | `0` | `−` | `+` | `+` | `++` |
| qwen3.6-35B-A3B (convergence) | `0` | `0` | `0` | `−` | `+` | `+` | `++` |
| DeepSeek-Coder-V2-Lite (not engaging) | `0` | `0` | `0` | `0` | `0` | `0` | `0` |

**The reasoning behind each column, so a miss is informative:**

- `t10`/`t15`/`t40` — a model that never reaches the ceiling cannot be helped by
  raising it and cannot be hurt much by lowering it to just above its median.
  gpt-oss hit the ceiling in **zero** of 238 failures, so `t40` must be flat for
  it; if it is not, the `turns` field is wrong (Section 7, step P1) or the
  ceiling statistic was misread.
- `fb-optional` — strips the prompt's simulation mandate. If this does *not*
  hurt, the current mandate was never binding and "mandatory vs optional
  simulation feedback" is a statement about enforcement, not wording. Either
  result is worth having; this is the manipulation check for the feedback axis.
- `fb-auto` — simulator feedback after every placement at zero turn cost. For an
  exhauster this converts verification turns into placement turns, which is
  precisely its binding constraint. For a converger it puts refuting evidence in
  front of a model that was not asking for it, which should help less.
- `rev-retry` vs `rev-structured` — both continue the loop after a rejected
  answer; only the second says *why* it was rejected, in simulator terms. The
  difference isolates grounded feedback from mere non-stopping, which is the
  central claim of the self-correction literature (Section 10) tested inside this
  harness. Predicted to be the largest single effect for the two convergers and
  near-nil for the exhausters, which rarely emit an answer to reject at all.

**The cross-cutting prediction, which is the actual thesis result:**

> The sign of the budget×model interaction and the sign of the
> revision×model interaction are **opposite**, and both are predictable from one
> cheap baseline statistic — the fraction of failures sitting on the turn
> ceiling. A model above ~50 % should be bought turns; a model below ~10 %
> should be bought a rejection-and-critique loop. Buying the wrong one is
> predicted to be worthless, not merely less good.

If that holds, the output is a selection rule, not a table. If the interaction is
flat, the honest finding is that harness gains are model-independent here — also
publishable, and stated as such rather than mined for a sub-cell that moved.

---

## 4. Analysis plan

Fixed before the data exists. Deviations get logged in the write-up.

### 4.1 Unit and pairing

The unit is the **task**, paired across arms. Every arm runs the identical task
list; `scripts/pair_arms.py` (Section 6.8) intersects the two `per_task/`
directories by `task_id` and **aborts** if the sets differ rather than comparing
an intersection silently. A mismatched glob or tier filter is the single easiest
way to invalidate this design, so it is checked mechanically, not by inspection.

### 4.2 Tests

| Question | Test |
|---|---|
| Did arm A move model M? | McNemar exact on the 432 paired outcomes, arm vs baseline repetition `s1` |
| Effect size | Δ success rate in pp, 95 % CI for paired proportions (Agresti–Min) |
| Is it inside pipeline noise? | Repeat against all five baseline repetitions; report the range of Δ and p. **A sign that flips across repetitions is reported as a null**, whatever the `s1` p-value |
| "…and for which models?" | Logistic mixed model `success ~ arm * model + (1 \| task)`, LRT on the interaction term. The task random intercept absorbs task difficulty, which is the dominant variance component |
| Multiplicity | Holm within each model's family of 7 arms. The Section-3 signs are primary at one-sided α = 0.05; anything else is exploratory and labelled so |
| Cost | Δ tokens/task and Δ turns/task, plus successes per 100 k tokens. **No arm is called an improvement on success rate alone** |
| Mechanism | failure-regime cross-tab per arm (Section 4.4), χ² on the §5 error families |

### 4.3 Power — stated now, so a null cannot be over-read

Paired MDE at n = 432, 80 % power, as a function of how many tasks the
intervention actually flips (`d` = discordance):

| `d` | b+c | SE(Δ) | MDE, α = 0.05 two-sided | MDE after Holm (7 tests) |
|---|---|---|---|---|
| 0.08 | 35 | 1.36 pp | 3.8 pp | 4.8 pp |
| 0.15 | 65 | 1.86 pp | 5.2 pp | 6.6 pp |
| 0.25 | 108 | 2.41 pp | 6.7 pp | 8.5 pp |

For contrast, two *independent* runs at p ≈ 0.42 give SE = 3.4 pp and an MDE of
9.4 pp. **Pairing is worth a factor of ~2.5, and is the entire reason this
experiment is affordable.**

Consequence to write down before running: **effects below ~5 pp are not
detectable on this set.** A flat cell means "no effect larger than ~5 pp", never
"no effect". `results_analysis.md` §7.3.1 is the cautionary precedent — a null
read at 35 % power — and this plan must not repeat it.

On the low-base-rate `scaled_sub500` (p ≈ 0.11) discordance is smaller, so the
absolute MDE is *better* (~3 pp at d = 0.06). That is counter-intuitive and it is
why the confirmatory phase is cheap.

### 4.4 Failure-mode accounting

`success rate` alone cannot distinguish the two regimes, so every arm reports:

**Failures** split into
- `ceiling` — `turns == budget` (budget exhaustion)
- `converged-early` — submitted an answer, `turns < budget` (premature convergence)
- `no-answer-early` — stopped without an answer below budget (the §5 "no solution submitted" family)
- `harness` — timeout, HTTP error, empty response

**Successes** split by route, which matters more than it looks:
- `submitted` — model emitted a valid `final_solution`
- `early-stop` — the loop's `is_solution_found` check fired (`executor.py:502`)
- `ceiling-valid` — budget expired but the board on it validated (`runner.py:793`)
- `fallback` — credited from `get_best_placement()` after the model removed a correct component (`runner.py:800`)

A gain concentrated in `fallback` or `early-stop` is a **scoring-credit shift, not
a capability gain**, and must be reported that way. `fb-auto` is especially
exposed here: auto-injection calls `run_simulation` on the target sequence, which
sets `_solution_found` and `_best_placement` for the model. Both routes are legal
under the current scoring contract, but the credit assignment changes, so the
split is mandatory for that arm rather than optional.

---

## 5. The interventions

Baseline = the exact configuration behind `results_analysis.md` §2: budget 25,
greedy, `AGENTIC_SYSTEM_PROMPT` unchanged, simulation urged in the prompt but not
enforced, loop returns as soon as the model emits content. The baseline prompt
string must stay **byte-identical**; a regression test pins it (Section 6.10) so
the five stored repetitions remain a valid comparator.

### A. Turn-budget sweep — 10 / 15 / 25 / 40

Already supported end to end: `--max-turns`, `submit_all.sh --turns N`, results
land in `<set>_tN/`. **No code needed.** 25 is the baseline, so three new runs per
model per set.

10 and 15 sit *below* the baseline deliberately. They are not padding: they
locate where each model's curve saturates and answer the cheaper-and-equal
question, which a purely upward sweep cannot. 10 is just below the convergers'
median failure turn (9–13), which is what makes their predicted flatness a real
test.

### B. Mandatory vs optional simulation feedback — three arms

The current harness is neither: it *commands* simulation in the prompt
(`prompts.py`, "You MUST call run_simulation after EVERY component placement")
and enforces nothing. Collapsing that into a two-level factor would hide the
distinction between wording and mechanism, so it is three levels.

| Arm | Prompt | Harness |
|---|---|---|
| `fb-optional` | `REQUIRED WORKFLOW` block and every "you MUST call run_simulation" line removed; tools described neutrally | unchanged |
| `fb-urged` (**baseline**) | current text | unchanged |
| `fb-auto` | baseline text, with only the sentence naming *where* feedback comes from rewritten | every `place_component` / `remove_component` response carries the simulation result for the target sequence, at **zero turn cost** |

Two new runs per model per set. The `fb-auto` prompt diff is confined to one
sentence and is recorded verbatim in the write-up — otherwise the arm confounds
mechanism with prompt rewriting, which is exactly the confound this three-level
split exists to avoid.

`fb-auto` needs a payload cap (Section 6.2): the full `run_simulation` return
carries grouped execution traces, and 40 turns of them will exhaust the 131 072
context. Injection carries catcher counts, `free_fall_errors` and final bit
states, and drops `execution_traces`.

### C. Failure-driven structured revision — two arms

Today the loop returns the moment the model emits content
(`client.py:2336-2351`); validation happens afterwards in the runner, and the
model never learns it failed. Both arms keep the loop alive after a *rejected*
answer, inside the same turn budget, for at most **R = 3** revision rounds.

| Arm | Message injected after a rejected answer |
|---|---|
| `rev-retry` | "Your submitted solution was rejected. Try again." — nothing task-specific |
| `rev-structured` | which scoring check failed; offending cells with coordinates; observed vs required output sequence; final bit states vs required; remaining inventory; the user-placed components |

`rev-retry` is the **compute-matched control** and is not optional. Both arms use
turns the baseline left unused, so neither is compute-matched *against baseline*;
`rev-structured − rev-retry` isolates the grounded content of the critique, and
`rev-retry − baseline` isolates merely not stopping. Without the control, any
gain is unattributable between the two.

The critique is generated from the **same** code path as scoring
(`validate_synthesis`, Section 6.3). A critique that describes a different check
from the one that rejected the answer would be feeding the model misinformation
and would invalidate the arm.

### D. Structured tool-parameter errors — recommended addition

`results_analysis.md` §9.4 found that malformed `place_component` calls surface a
raw Python `TypeError` to the model, affecting up to 4 % of DeepSeek's tasks, and
calls it a harness defect. It is also, precisely, a harness intervention — and
the only one predicted to help the one model the other seven cannot reach.
`executor.execute` (`executor.py:80`) already dispatches through a dict; wrapping
the call to return a message naming the accepted parameters is a contained change
and one extra arm per model.

**This needs your decision** (Section 11). It is the cheapest arm, it is the only
non-trivial prediction for DeepSeek, and it converts a known defect into a
measured result. It is listed separately because it was not in the brief.

---

## 6. Build contract

Ordered, each with its own verification. Every change is additive and off by
default; the baseline code path must stay observably identical.

**6.1 — `executor.py`: `auto_simulate` flag.** Add the keyword to
`ToolExecutor.__init__`. When set, `place_component` and `remove_component`
attach a `simulation` key to their return dict, from the existing
`run_simulation(self._target_sequence)` — which is already non-destructive
(saves and restores hoppers, bit states, trigger queue, history).
*Verify:* new unit test asserts the key is present when on, absent when off, and
that board state after a placement is identical either way.

**6.2 — `executor.py`: injection payload cap.** The injected dict carries
`left_catcher`, `right_catcher`, `interceptor`, `free_fall_errors`,
`final_bit_states`, `total_marbles`. No `execution_traces`.
*Verify:* test asserts the serialised payload stays under a fixed byte budget on
a 15×15 board with 8 marbles, and that `execution_traces` is absent.

**6.3 — `runner.py`: critique builder.** `build_revision_critique(task_info,
placed) -> str | None`, placed next to `validate_synthesis` and sharing its
checks. Returns `None` when the solution validates. Both `rev-*` styles route
through it; `rev-retry` discards the body and emits the fixed string, so the two
arms cannot diverge in *when* they fire.
*Verify:* test asserts `None` on the reference solution of a known task, and that
a free-fall failure's critique names the same cell the scorer rejected.
*Note:* `VLLMClient` returns the final answer as `{"content": "<raw text>"}` and
never parses a `final_solution` field (`client.py:2351`); the runner already
scores `executor.get_placed_components()` instead. The critique must validate the
**executor's board**, not a parsed answer field, or it will reject every vLLM
submission regardless of content.

**6.4 — `client.py`: `on_final_answer` callback.** `generate_with_tools` gains an
optional callback. When the model emits content, the loop calls it; a returned
string is appended as a `user` message and the loop continues, up to R rounds;
`None` returns as today. Implement in **`VLLMClient` and `MockClient` only** —
vLLM is the JURECA path and Mock is the test path. The other six clients are left
untouched.
*Verify:* mock-provider test — loop continues on a rejected answer, stops after
exactly R rounds, returns immediately on an accepted one, and behaves identically
to today when the callback is absent.

**6.5 — `prompts.py`: variants.** Add `AGENTIC_SYSTEM_PROMPT_OPTIONAL` and
`AGENTIC_SYSTEM_PROMPT_AUTOSIM`. Leave `AGENTIC_SYSTEM_PROMPT` untouched.
*Verify:* test pins the baseline string's hash (Section 6.10).

**6.6 — `runner.py`: wiring and provenance.** New `TuringTumbleBenchmark`
kwargs `auto_simulate`, `revision_rounds`, `revision_style`, `prompt_variant`,
and matching CLI flags. All four are written into the saved report's config block
next to the existing `max_turns` (`runner.py:1333`).
*Verify:* run the mock provider for each arm; assert every setting round-trips
into the report JSON. Without this, an arm's provenance lives only in a directory
name — and §9.1 of `results_analysis.md` records what that costs.

**6.7 — `jureca/`: plumbing.** `submit_all.sh` gains `--arm LABEL`, passed via
`--export` as `ARM=`; `run_benchmark.sbatch` maps the label to CLI flags and
appends `_a<label>` to `OUT_LABEL`. Arm labels must contain **no comma** — sbatch
splits `--export` on commas, and that exact failure already cost a silently
truncated `--sets` value (§9.1). Add the label to the existing comma guard.
*Verify:* `submit_all.sh -n --arm fb-auto <model>` prints the intended sbatch
line; a comma-bearing label is rejected before submission.

**6.8 — `inspect_results.py` + `scripts/pair_arms.py`.** Extend
`parse_set_label` with the arm suffix, returning a 5-tuple; the ripple reaches
`set_sort_key`, `resolve_temperature` (which already strips the sample suffix and
must now strip the arm too — the anchoring bug fixed there once), `discover` and
`render`. `pair_arms.py` reads two `per_task/` directories, asserts identical
task-id sets, and emits the 2×2 table, McNemar exact p, paired CI, the
Section-4.4 cross-tab and the token/turn deltas.
*Verify:* re-run `inspect_results.py` over existing `benchmark_results/jureca_tier1/`
and diff against the current output — **the stored numbers must not move**;
`pair_arms.py` on a baseline repetition against itself must return Δ = 0, p = 1.

**6.9 — `scaled_sub500`.** 500 `scaled` Tier-1 tasks drawn once with a recorded
seed, stratified by board size (13×13 / 15×15) and ground-truth component count,
materialised as a symlink directory so no loader changes are needed, and
registered as a set label in the sbatch roster. The task-id list is committed.
*Rationale:* qwen3.6 at 335–436 s/task would need ~13.5 h for the full 1 013
tasks at 8 workers, past the 12 h walltime; 500 fits, and Section 4.3 shows the
paired MDE is *better* there than on the primary set.
*Verify:* the list is exactly 500 ids, all resolve, strata proportions match the
parent set within 2 pp.

**6.10 — Regression guard.** One test asserting that, with every new flag at its
default, the agentic path produces byte-identical prompts and the same tool
sequence as the current code on a fixed mock transcript. This is the test that
protects the five stored baseline repetitions' comparability, and it is the reason
to write it first rather than last.
*Verify:* `uv run python -m pytest tests/ -v` green before and after each of 6.1–6.9.

---

## 7. Execution

### Phase 0 — measurement integrity (no GPU time)

Cheap, and the sweep is uninterpretable without it.

- **P0.** Commit `review/harness_ablation/predictions.md` (Section 3). Record the
  hash. Nothing is submitted before this.
- **P1. Certify the `turns` field, per model.** §9.2 records that `turns` was
  `len(tool_calls)`, not turns — DeepSeek logged 444 against a budget of 25 — and
  says its ceiling fractions should not be quoted from the stored data. The
  budget sweep's entire mechanism story rests on this field. Recompute
  `max(transcript.turn) + 1` from the stored `per_task/*.json` and compare with
  `metrics.turns` / `turns_source` for all five models. Re-derive the §4 ceiling
  fractions from the certified counts. **If DeepSeek's regime changes, Section 3
  changes — and it changes before any job runs, not after.**
  *Note:* `VLLMClient` appends one `turn_logprobs` entry per API call whether or
  not `--capture-logprobs` is set (`client.py:2240-2248`), so new runs report
  `turns_source: api_calls` without extra flags. This check is about the stored
  data.
- **P2.** Confirm all five greedy `scaled_1comp` repetitions cover the identical
  432 task ids. DeepSeek's fifth repetition hit walltime (§2), so its baseline
  family is four — recorded, not worked around.
- **P3.** Baseline truncation rate per model via `jureca/count_truncations.py`,
  so `fb-auto` and `t40` context growth can be judged against a known figure.

### Phase 1 — primary set, `scaled_1comp` (432)

7 arms × 5 models = 35 runs. Cost from the §6 latencies at 8 workers:

| Model | ≈ per arm | 7 arms | note |
|---|---|---|---|
| gemma-4-31B-it | 12 min | ~1.5 h | |
| gemma-4-26B-A4B-it | 12 min | ~1.5 h | |
| gpt-oss-120b | 22 min | ~2.5 h | |
| DeepSeek-Coder-V2-Lite | 34 min | ~4 h | |
| **qwen3.6-35B-A3B** | **5.8 h** | **~40 h** | 335–436 s/task; dominates the budget |

**≈ 50 h of node time, ±40 %** — `t10`/`t15` are cheaper, `fb-auto`/`rev-*`/`t40`
dearer. qwen3.6 is ~80 % of it and must be submitted as one job per one-to-two
arms to stay inside 12 h; the fast four can batch.

Gate before Phase 2, checked per arm: truncation rate must not exceed the P3
baseline by more than 2 pp. An arm that fails this is confounded with truncation
and is re-run with a trimmed injection payload, not reported.

### Phase 2 — confirmatory, `scaled_sub500` (conditional)

- **A baseline cell per model**, which Phase 1 did not need. See Section 12.1:
  there is no stored `scaled` run left to pair against.
- The three budget arms, for all five models, **regardless of Phase 1** — the
  exhaustion mechanism is strongest on the larger boards and the budget
  prediction is sharpest there.
- Any other arm whose Phase-1 effect clears the Holm threshold for any model.

Cost scales as 500/432 ≈ 1.16× per arm; baseline plus the budget arms is ~20 h if
no other arm qualifies, roughly doubling if several do.

---

## 8. Threats to validity

Each names the handling, not just the risk.

| Threat | Handling |
|---|---|
| 35 primary tests invite a garden of forking paths | Section 3 committed with a hash before submission; Holm within each model's family of 7; exploratory cells labelled |
| `fb-auto` is not compute-matched — more tokens per turn | tokens/task and successes per 100 k tokens reported beside every rate; no arm is called an improvement on rate alone |
| `rev-*` is not compute-matched — uses turns the baseline left idle | `rev-retry` control isolates non-stopping from grounded critique |
| Auto-injection can trip `is_solution_found` on a board the model did not intend | Section 4.4 success-route split; those successes counted and reported separately |
| Best-placement fallback rewards arms that simulate more | same split; a gain concentrated in `fallback` is reported as a credit shift |
| Context overflow at `t40` + `fb-auto` | payload cap (6.2); P3 baseline; the +2 pp truncation gate |
| vLLM continuous batching, 0.2–0.6 pp | five-baseline sensitivity check; a sign that flips across repetitions is a null |
| A stale shell variable silently changes an arm | every arm setting written into the report config (6.6); labels pinned on every submission, as `submit_all.sh` already does for `SETS`/`MAX_TURNS` |
| `turns` may be the old metric in stored data | P1 certification gates Section 3 |
| Tier 1 only, and one derived set | stated as an external-validity limit; Phase 2 is the only generalisation claimed |
| 99.6 % zero-slack inventory (§9.4) | no inventory claim is made; the critique's "remaining inventory" line is near-vacuous on `scaled` and informative only on `official`, and is reported as such |
| Task sets drifting between arms | `pair_arms.py` aborts on a task-id mismatch rather than intersecting silently |

---

## 9. Deliverables

1. `review/harness_ablation/predictions.md` — Section 3, committed pre-hoc.
2. `docs/harness_ablation.md` — written in the style of `results_analysis.md`:
   every number regenerable from `inspect_results.py` and `pair_arms.py`, with the
   regeneration commands at the top.
3. A prediction-vs-outcome scorecard: the Section-3 grid with the measured sign
   beside each cell, and a paragraph on every miss.
4. Figures via `make_figures.sh`: success vs turn budget per model (the
   interaction, one line per model); Δ success vs Δ tokens per arm (the cost
   frontier); the Section-4.4 regime stack per arm.
5. The answer, in the form the question asks for: a table of which interventions
   help which models, and the selection rule from Section 3 marked as confirmed
   or refuted.

---

## 10. Related work

Grounding for each axis, and the reason the design looks the way it does.

- **[Harness-Bench: Measuring Harness Effects across Models in Realistic Agent
  Workflows](https://arxiv.org/pdf/2605.27922)** (Yao et al.) is the closest
  precedent and asks nearly this question, reporting that identical harness
  configurations do not improve models uniformly — which is the premise Section 3
  makes falsifiable on a mechanically verifiable task. I confirmed the title,
  authors and that abstract-level claim; the PDF did not yield its experimental
  detail on fetch, so its design should be read properly before the write-up
  positions against it.
- **[Thinking vs. Doing: Agents that Reason by Scaling Test-Time
  Interaction](https://arxiv.org/pdf/2506.07976)** motivates the turn-budget axis:
  more interaction beats more within-step deliberation as budgets grow. The
  Section-3 prediction is sharper — it says the gain is confined to models that
  are demonstrably budget-bound.
- **[Large Language Models Cannot Self-Correct Reasoning
  Yet](https://arxiv.org/abs/2310.01798)** (Huang et al., ICLR 2024) is why
  `rev-retry` exists as a control rather than as an afterthought: intrinsic
  self-correction without external signal does not help and can hurt. This
  harness owns a ground-truth simulator, so `rev-structured − rev-retry` is a
  clean instance of the external-feedback contrast.
- **[When Can LLMs Actually Correct Their Own
  Mistakes?](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713/125177/When-Can-LLMs-Actually-Correct-Their-Own-Mistakes)**
  (Kamoi et al., TACL) supplies the conditions under which self-correction works
  — reliable verification signal, actionable critique — both of which the
  simulator supplies here, making this a favourable rather than neutral setting.
- **[Beyond Prompts: Measuring and Optimizing LLM Tool-Agent
  Harnesses](https://arxiv.org/html/2609.05736)** and
  **[HarnessOpt-Bench](https://arxiv.org/html/2608.06301v1)** frame the harness as
  a searchable object. Worth citing as the wider programme, and worth explicitly
  *not* doing: this plan ablates a fixed factorial to attribute effects, rather
  than searching for a best harness, which would not answer "for which models".

---

## 11. Decisions I made, and two that are yours

**Made, with the alternative stated:**

- **Paired single runs, not repetitions.** §7.2 shows run-to-run sd is an order
  of magnitude below the sampling interval. Repetitions would cost 5× for less
  precision than pairing buys. The five stored baseline repetitions are reused as
  a sensitivity check instead of being matched arm-for-arm.
- **Three feedback levels, not two.** The brief says "mandatory vs. optional", but
  the current harness is *urged-and-unenforced*, which is neither. A two-level
  contrast would silently compare enforcement against the status quo and label it
  a prompt effect.
- **`rev-retry` added as a control.** Without it, a revision gain is
  unattributable between "did not stop" and "was told why". This is the one
  addition to the brief I would not drop.
- **`scaled_1comp` primary, `scaled_sub500` co-primary for the budget arm.** The
  budget mechanism is strongest on the larger boards, so testing it only where the
  baseline is most convenient would weaken the sharpest prediction in the plan.
- **`--declare-zero-parts` / `--observable-inventory` excluded.** §9.4 establishes
  they measure nothing on these sets.
- **`scaled_2comp` excluded.** Its only run is at temperature 0.7, so it has no
  greedy baseline to pair against.

**Yours:**

1. **Compute.** ~50 node-hours for Phase 1, ~80 % of it qwen3.6-35B-A3B. If that
   is too much, the honest cut is to drop qwen3.6 from Phase 1 and run it only on
   the arms where gpt-oss-120b shows an effect — the two are the same regime and
   statistically tied under greedy (§2), so qwen mainly buys a replication of the
   convergence row. The cut I would *not* make is trimming the task count, which
   is what the power in Section 4.3 rests on.
2. **Arm D (structured tool-parameter errors).** Not in the brief. It is the
   cheapest arm, the only non-trivial prediction for DeepSeek, and it turns a
   defect already recorded in §9.4 into a measured result. Include it, or leave
   the DeepSeek row flat across all seven arms?

---

## 12. What the build found

Three things surfaced while implementing Section 6 that change the plan rather
than merely executing it.

### 12.1 The `scaled` corpus was rebuilt after the stored run

Commit `a365ceb` ("data: transcribed official boards and a fully verified task
corpus") grew `data/tasks/scaled` from 2 121 files to 3 579. Measured against the
frame that run actually covered (`a365ceb^`), with the same glob and tier filter:

| | count |
|---|---|
| stored `scaled` tier-1 tasks | 1 013 |
| deleted or renamed by `a365ceb` | 292 |
| surviving filename, tier-1 in both | 721 |
| of those, unchanged on board / solution / available_parts / targets | **108** |

**Only 108 of 1 013 are still the same task.** The stored `scaled` figures in
`docs/results_analysis.md` §2–§5 describe a corpus that no longer exists, so no
`scaled` arm can be paired against them. Phase 2 therefore runs its own baseline,
and `scripts/build_sub500.py` samples the working tree by default;
`--frame-commit a365ceb` reproduces the audit above.

**The primary set is untouched.** `scaled_1comp` is 432 tier-1 tasks at
`a365ceb^` and 432 today, all 432 byte-identical on every scoring-relevant field.
`scaled_2comp` (402) likewise. Phase 1's pairing against the five stored greedy
repetitions is sound, which is the assumption the whole design rests on.

This is worth reporting independently of this experiment: it also means the
`scaled` rows of the results chapter are not reproducible from the current
checkout without `git checkout a365ceb^ -- data/tasks/scaled`.

### 12.2 The `turns` field is already demonstrably wrong on some stored data

`scripts/certify_turns.py` (step P1) run over `benchmark_results/tier1_comparison`
flags `deepseek-v4-pro` immediately: tasks recording 28, 30 and 34 turns against a
budget of 25, with `turns_source: unrecorded`, where the transcript caps at 25.
That is the `len(tool_calls)` bug of §9.2, detected mechanically rather than
noticed. The script exits non-zero on any such cell and reports a ceiling fraction
derived from the transcript beside the recorded one.

It must be run on `benchmark_results/jureca_tier1` — which is **not in this
checkout** — before the prediction table is committed, because the table's
regime assignment is built from exactly that statistic.

### 12.3 `fb-auto` shifts scoring credit, and the shift is now measured

The first end-to-end paired run showed it directly: under `fb-auto` every success
was credited through the loop's early-stop path, against none in the baseline.
The injected simulation calls `run_simulation` on the target sequence, which sets
`_solution_found`, so the loop ends the episode before the model ever submits.
Section 4.4 predicted this; the harness now records `metrics.credit_route` at the
point the decision is made, because the `fallback` route cannot be reconstructed
afterwards — the per-task record keeps the credited board, not the executor's
final one. `scripts/pair_arms.py` prints the route split and warns when an arm's
net gain is fully accounted for by routes the agent did not drive.

### 12.4 Two deviations from Section 6

- **6.4** says to implement the revision hook in `VLLMClient` and `MockClient`.
  `MockClient.generate_with_tools` is a stub that returns an error immediately, so
  extending it would have meant building a scripted mock loop that could drift
  from the production one. The tests instead drive the real `VLLMClient` loop
  against a stubbed HTTP transport. `MockClient` is untouched; the runner refuses
  a revision arm on any provider whose loop would ignore the hook.
- **4.4** treated the success-route split as an analysis step. It is a
  measurement step: `fallback` is not recoverable downstream, so the runner
  records it (12.3).
