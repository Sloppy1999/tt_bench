#!/usr/bin/env bash
# =============================================================================
# Render every figure for the presentation, into one directory
# =============================================================================
# The visualisation scripts predate each other and write to three different
# places — jureca/plot_results.py takes --out, scripts/visualize_benchmark_v2.py
# hardcodes data/visualizations/, and scripts/three_panel_viz.py drops a PNG in
# the repo root. This collects all of it under one bundle so downloading the lot
# is a single rsync.
#
# Usage:
#   bash jureca/make_figures.sh                 # bundle in ./presentation
#   bash jureca/make_figures.sh /path/to/out
#
# A failing step does NOT abort the run: the rest still render and the failures
# are listed at the end with a non-zero exit. A half-finished bundle you know
# about beats a complete-looking one that silently skipped something.
# =============================================================================

set -uo pipefail   # NOT -e: individual steps are allowed to fail, see run_step

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
banner() { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "  ${RED}✘${NC} $1"; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

BUNDLE="$(cd "$(dirname "${1:-$REPO/presentation}")" 2>/dev/null && pwd)/$(basename "${1:-presentation}")"
RESULTS="${RESULTS_DIR:-benchmark_results/jureca_tier1}"

# The scaled_* variants carry ~1000 tasks against 5 for the small sets. At n=5 a
# single task moves the rate by 20 points — the noise floor that made two
# identical runs of one model differ by 20. Presentation figures use the big ones.
PLOT_SETS="${PLOT_SETS:-scaled_1comp,scaled_2comp,scaled}"

# Models to leave out of the comparison figures. Not "broken" — these produced
# results that are not comparable with the rest, so averaging them into the same
# chart would be the mistake. qwen2.5-coder-7b is the only model whose native
# context (32768) is below TARGET_CTX, so it alone ran YaRN-extended; it also
# completed only one scaled set, at a median 0.8s per task against 11-216s for
# everything else. See docs/results_analysis.md section 7.1.
# Per-model heatmaps are still generated for excluded models — those never
# compare across models, so there is nothing to contaminate.
EXCLUDE_MODELS="${EXCLUDE_MODELS:-qwen2.5-coder-7b}"

FAILED=()
run_step() {
    local label="$1"; shift
    if "$@" >/tmp/mkfig.$$.log 2>&1; then
        ok "$label"
    else
        fail "$label — last lines:"
        tail -n 6 /tmp/mkfig.$$.log | sed 's/^/      /'
        FAILED+=("$label")
    fi
    rm -f /tmp/mkfig.$$.log
}

banner "Setup"
if [ ! -d "$RESULTS" ]; then
    fail "No results at $RESULTS — nothing to plot"
    exit 1
fi

# Resolve the interpreter ONCE. `uv run` re-resolves the project on every call,
# which is slow across a dozen invocations and turns any packaging hiccup into a
# per-step failure. Prefer the venv directly and verify it can actually import
# matplotlib rather than assuming.
PY_CMD=()
for cand in "$REPO/.venv/bin/python" "$REPO/.venv-ttbench/bin/python"; do
    if [ -x "$cand" ] && "$cand" -c 'import matplotlib' >/dev/null 2>&1; then
        PY_CMD=("$cand"); break
    fi
done
if [ ${#PY_CMD[@]} -eq 0 ] && command -v uv >/dev/null 2>&1 \
   && uv run python -c 'import matplotlib' >/dev/null 2>&1; then
    PY_CMD=(uv run python)
fi
if [ ${#PY_CMD[@]} -eq 0 ]; then
    fail "No interpreter with matplotlib found"
    echo "      Tried $REPO/.venv/bin/python, $REPO/.venv-ttbench/bin/python, and 'uv run python'."
    echo "      Run jureca/setup.sh, or point this at a venv that has matplotlib."
    exit 1
fi
ok "Interpreter: ${PY_CMD[*]}"
mkdir -p "$BUNDLE"/{comparison,comparison-dark,heatmaps,benchmark}
ok "Bundle: $BUNDLE"
ok "Results: $RESULTS"
ok "Sets: $PLOT_SETS"
[ -n "$EXCLUDE_MODELS" ] && warn "Excluded from comparisons: $EXCLUDE_MODELS"

# ── Which models are worth plotting ──────────────────────────────────────────
# A report whose tasks all record zero turns never generated a token: a base
# checkpoint with no chat template, or a tokenizer vLLM could not load. Plotting
# it plots a 0% that looks like a score. Detect it from the data instead of
# maintaining a hand-written skip list that goes stale.
banner "Model readiness"
MODELS=()
for d in "$RESULTS"/*/; do
    [ -d "$d" ] || continue
    m="$(basename "$d")"

    # Three outcomes, kept distinct on purpose. An earlier version treated any
    # non-zero exit as "every task recorded 0 turns", so a broken interpreter
    # reported a confident, specific and completely wrong diagnosis for every
    # model. "Could not determine" is not "determined to be bad".
    verdict="$("${PY_CMD[@]}" - "$d" 2>/tmp/mkfig.probe.$$ <<'PY'
import json, sys, glob
turns = 0
for f in glob.glob(sys.argv[1] + "/*/benchmark_*.json"):
    with open(f) as fh:
        for r in json.load(fh).get("results", []):
            turns = max(turns, (r.get("metrics") or {}).get("turns") or 0)
print("MEASURED" if turns > 0 else "ZERO_TURNS")
PY
)"
    rc=$?
    if [ $rc -ne 0 ] || [ -z "$verdict" ]; then
        fail "$m: could not inspect the reports (exit $rc) — NOT treating this as a bad model"
        sed 's/^/      /' /tmp/mkfig.probe.$$ | tail -n 4
        FAILED+=("readiness probe: $m")
    elif [ "$verdict" = "MEASURED" ]; then
        MODELS+=("$m"); ok "$m"
    elif [ "$verdict" = "ZERO_TURNS" ]; then
        warn "$m: every task recorded 0 turns — not a measurement, skipping"
    else
        fail "$m: unexpected probe output '$verdict'"
        FAILED+=("readiness probe: $m")
    fi
    rm -f /tmp/mkfig.probe.$$
done

if [ ${#MODELS[@]} -eq 0 ]; then
    fail "No model produced a usable measurement"
    exit 1
fi

# ── 1. Cross-model comparison ────────────────────────────────────────────────
banner "Comparison figures"
run_step "success / turns / failures (light)" \
    "${PY_CMD[@]}" jureca/plot_results.py \
        --results-dir "$RESULTS" --sets "$PLOT_SETS" --exclude "$EXCLUDE_MODELS" \
        --out "$BUNDLE/comparison"
run_step "success / turns / failures (dark)" \
    "${PY_CMD[@]}" jureca/plot_results.py \
        --results-dir "$RESULTS" --sets "$PLOT_SETS" --exclude "$EXCLUDE_MODELS" --dark \
        --out "$BUNDLE/comparison-dark"

# ── 2. Per-model board heatmaps ──────────────────────────────────────────────
# One model per invocation, deliberately. analyze_tier1_experiment.py collects
# reports with rglob, so pointing it at the results root would pool all models
# into a single heatmap of nobody — averaged, plausible-looking and wrong.
banner "Board heatmaps (one model per run)"
for m in "${MODELS[@]}"; do
    run_step "heatmap: $m" \
        "${PY_CMD[@]}" scripts/analyze_tier1_experiment.py \
            --results-dir "$RESULTS/$m" \
            --output-dir "$BUNDLE/heatmaps/$m" \
            --model-label "$m"
done

# ── 3. Benchmark description (no results needed) ─────────────────────────────
# These describe the benchmark itself — the task inventory and an example board.
# Useful for the "what is TT-Bench" slides that come before any numbers.
banner "Benchmark description"
run_step "task inventory" "${PY_CMD[@]}" scripts/visualize_benchmark_v2.py
run_step "example board (three panel)" "${PY_CMD[@]}" scripts/three_panel_viz.py

# Both write outside any --output-dir we can pass, so gather them.
if [ -d "$REPO/data/visualizations" ]; then
    cp -f "$REPO/data/visualizations"/*.png "$BUNDLE/benchmark/" 2>/dev/null && \
        ok "collected data/visualizations → benchmark/"
fi
[ -f "$REPO/board_three_panel.png" ] && \
    cp -f "$REPO/board_three_panel.png" "$BUNDLE/benchmark/" && ok "collected board_three_panel.png"

# ── Summary ──────────────────────────────────────────────────────────────────
banner "Done"
COUNT=$(find "$BUNDLE" -type f \( -name '*.png' -o -name '*.svg' -o -name '*.csv' \) | wc -l | tr -d ' ')
echo ""
echo "  $COUNT file(s) in $BUNDLE"
find "$BUNDLE" -mindepth 1 -type d | sort | while read -r d; do
    n=$(find "$d" -maxdepth 1 -type f | wc -l | tr -d ' ')
    printf "    %-40s %s file(s)\n" "${d#$BUNDLE/}" "$n"
done

echo ""
echo "  Download to your machine (run LOCALLY, not on the login node):"
echo "    rsync -avz -e 'ssh -i ~/.ssh/id_ed25519' \\"
echo "      peral1@jureca.fz-juelich.de:$BUNDLE/ ~/Projects/Thesis/tt_bench/presentation/"
echo ""

if [ ${#FAILED[@]} -gt 0 ]; then
    fail "${#FAILED[@]} step(s) failed:"
    printf '      %s\n' "${FAILED[@]}"
    echo ""
    warn "The bundle is incomplete. Fix these before using it in the talk."
    exit 1
fi
ok "All steps completed"
