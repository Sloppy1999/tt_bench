# Question Complexity Criteria

This document defines a consistent method to classify question complexity for all files in this folder.

## Scope

Applies to all question files under this directory, including the three observed schema variants:

- Variant A (legacy labeled): top-level keys `task_id`, `questions`; question keys include `qid`, `type`, `question`, `answer`, `difficulty`.
- Variant B (minimal): top-level key `questions` (sometimes no `task_id`); question keys include `question_id`, `type`, `question`, `answer`, optional `options`.
- Variant C (guided): top-level keys `task_id`, `challenge_number`, `questions`; question keys include `id`, `type`, `question`, `expected_answer`, `hints`.

## Normalization Rules

Before scoring, normalize each question into this internal shape:

- `question_id`: use first available field in this order: `qid`, `question_id`, `id`.
- `answer`: use `answer` if present, else `expected_answer`.
- `question_type`: use `type`.
- `task_id`: use top-level `task_id` and remove `_questions` suffix if present; if missing, derive from filename by removing `_questions`.
- `challenge_tier` and `tags`: load from `tasks/official/INDEX.json` using normalized `task_id`.

## Complexity Model

Complexity is a weighted sum of six factors.

### 1. Reasoning Depth (RD): 0-4

- 0: Direct lookup/count (single fact)
- 1: Single-step inference (one causal step)
- 2: Multi-step local inference (short chain)
- 3: State transition reasoning across events
- 4: Multi-mechanism causal reasoning or counterfactual logic

### 2. Temporal Horizon (T): 0-4

How many sequential events/balls must be tracked.

- 0: 1 event
- 1: 2-3 events
- 2: 4-7 events
- 3: 8-15 events
- 4: >15 events or steady-state/loop behavior

### 3. State Breadth (S): 0-4

How many mutable elements must be tracked (bits, gear bits, interceptors, latch states).

- 0: No mutable state
- 1: One mutable element
- 2: Two mutable elements
- 3: Three to four mutable elements
- 4: Five or more mutable elements or coupled register+latch behavior

### 4. Response (R): 0-4

Expected output complexity.

- 0: Binary/multiple-choice response (`select` with small option set)
- 1: Single numeric/token response
- 2: Short sentence/sequence (1-2 clauses)
- 3: Structured trace/table style response
- 4: Explanatory answer requiring causal justification

### 5. Mechanic Complexity (M): 0-5

From challenge tags in `INDEX.json`, assign base mechanic weight by the most complex tag present:

- 0: `routing`
- 1: `triggering`, `interceptor`, `crossover`
- 2: `state_bit`, `write_op`, `non_destructive_read`
- 3: `binary_counting`, `looping`
- 4: `gear`, `gear_bit`, `latch`, `latching`, `overflow`

Add +1 (cap at 5) if two or more tags have weight >= 2.

### 6. Tier Adjustment (A): 0-3

From challenge tier in `INDEX.json`:

- Tier 1 -> 0
- Tier 2 -> 1
- Tier 3 -> 2
- Tier 4 -> 3

## Final Score and Labels

Compute:

`complexity_score = RD + T + S + R + M + A`

Map score to label:

- 0-7: `easy`
- 8-14: `medium`
- 15-24: `hard`

Optional future extension:

- 20+: `very_hard` (only if the project adopts a 4-level scale)

## Type-Based Starting Defaults

Use this as an initial estimate before factor scoring:

- Usually easy base: `component_count`, `parts_count`, `select` (binary), simple `numeric`
- Usually medium base: `ball_path`, `output_sequence`, `trigger_sequence`, `alternating_mechanism`, `configuration`
- Usually hard base: `state_trace`, `state_tracking`, `logic_analysis`, high-step `calculation`

Then adjust with full scoring model.

## Calibration Guidance

Use this process when adding/updating questions:

1. Normalize schema fields.
2. Score each factor (RD, T, S, R, M, A).
3. Compute `complexity_score` and assign label.
4. Compare with existing `difficulty` if present.
5. If mismatch is >= 2 bands (easy vs hard), review wording and expected answer detail.

## Dataset Notes (Current Observations)

- Existing explicit labels appear mainly in Variant A files (`easy`, `medium`, `hard`).
- Newer files (Variant C) often omit `difficulty`; this rubric is intended to fill that gap consistently.
- Question counts per file are typically 2-4; this does not directly determine complexity and should not be used as a scoring factor by itself.

## Recommended Metadata Addition

For future consistency, add this block per question:

```json
"complexity": {
  "score": 11,
  "label": "medium",
  "factors": {
    "R": 2,
    "T": 2,
    "S": 1,
    "B": 2,
    "M": 3,
    "A": 1
  }
}
```

This keeps complexity explainable and auditable.
