# Dataset remediation plan

Baseline: September 18 audit under `review/dataset_audit/`. Existing local interception,
catcher-geometry, transcription and regression edits belong to the user and must be preserved.
Audit counts are historical, not immutable expectations after simulator changes.

## 1. Validation safeguards (implement first)

- Share explicit target validation across simulator, benchmark and tool early-stop.
- Check final_marble_state, required_output, numeric counts and final bit states together.
- Reject empty/placeholder sequences, malformed numeric targets and absent explicit targets.
- Carry required_output through task loading and tool construction.
- Add regression coverage for conflicting targets, interception, empty targets and parity.
- Generate a content-hashed eligibility manifest without rewriting or moving dataset files.
  Separate negative fixtures from positive puzzles; do not accept an unproved negative label.
- Correct stale AGENTS instructions.

Acceptance: tests pass; all files are classified reproducibly; inconsistent coexisting
outputs cannot score or terminate an agent as successful. A mechanical pass is not a
certification of objective semantics or provenance.

## 2. Negative-label integrity

Re-run the existing single-part exhaustive checker using the strengthened validator.
Separate refuted, proven and unproven labels. A changed target contract can turn a
previous witness into a rejection, which is not proof of physical impossibility.
Require consistent targets and an auditable witness/certificate before trusting labels.
Extend enumeration to initial states and larger inventories with explicit budgets;
never convert timeouts or unenumerated tasks to proven negatives. Integrate certificates
with scorer preflight before spending API calls or rewarding refusal.

## 3. Canonical identity and derivation

Choose official filename stems as canonical IDs; produce a reviewed migration map for
legacy IDs. Repair inventory and component-count inconsistencies. Record parent content
hashes, generation parameters and generator version; rebuild descendants only from
validated parents. Do not overwrite objectives or targets merely to make verification pass.

## 4. Questions and metadata

Repair the nine contradicted mechanical answers after their boards are validated.
Review the mis-typed bit-state question and other semantic answers separately. Derive
INDEX totals from files; normalize titles, flags and count descriptions only after fresh
validation. Preserve original data and migration diffs for review.

## 5. Dataset release gate and evaluation splits

Promote reproducible structural, simulator, scorer-parity, identity and provenance checks
to a release gate. Assign duplicate families before train/test splitting; keep a family
in one split. Publish a manifest of eligible positives, certified negatives, quarantine
reasons and unresolved human-review items. Wire the manifest into benchmark discovery
with file-hash checks so later edits cannot silently bypass validation.

## Implementation record

See `review/remediation/` for fresh gate output and test results. Data repair and negative
certification follow the safeguards; the historical audit remains preserved.

### Stage 1 completed

- Shared output contract implemented in `src/tt_bench/simulator/targets.py` and
  called by simulator verification, runner scoring and tool early-stop.
- Required-output declarations survive loading and reach the tool executor.
- Tool catcher-approach handling follows the current Board.catcher_at geometry.
- `scripts/build_dataset_manifest.py` records per-file and validator hashes,
  rejects concurrent changes, and separates positive candidates, quarantine and
  negative review. It does not yet filter benchmark discovery automatically.
- AGENTS paths and commands updated. Existing state-restoration regression fixture
  uses one marble so automatic trigger releases do not confound its purpose.
- New regression tests exercise conflicting targets, malformed/absent sequences,
  interception, final bit states, loader propagation, manifest determinism,
  duplicate IDs, negative review and concurrent edits.

Stages 2–5 remain open. Candidate files are not a trusted release: semantic targets,
parent provenance, negative certificates and question repairs remain to be reviewed.
