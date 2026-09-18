#!/usr/bin/env bash
# =============================================================================
# Sync TT-Bench to JURECA
# =============================================================================
# Copies the task corpus and the code that reads it to the project directory on
# JURECA, so a Slurm run benchmarks the same data and the same simulator as the
# local tree.
#
# Usage:
#   JURECA_HOST=user@jureca.fz-juelich.de bash jureca/sync_to_jureca.sh        # preview
#   JURECA_HOST=user@jureca.fz-juelich.de bash jureca/sync_to_jureca.sh --go   # transfer
#
# Optional:
#   JURECA_DIR=/p/scratch/westai0070/$USER/tt-bench   # override the destination
#
# This mirrors: files deleted locally are deleted on JURECA. That is deliberate
# — invalid tasks that were removed here must not stay behind and get scored
# there — but it means the destination is made to match this tree exactly.
# Benchmark results on JURECA are never touched.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
banner() { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC} $1"; }
warn()  { echo -e "  ${YELLOW}!${NC} $1"; }
fail()  { echo -e "  ${RED}✘${NC} $1"; }

GO=false
[ "${1:-}" = "--go" ] && GO=true

if [ -z "${JURECA_HOST:-}" ]; then
    fail "JURECA_HOST is not set."
    echo "    Set the login node you use, for example:"
    echo "      export JURECA_HOST=your-user@jureca.fz-juelich.de"
    exit 1
fi

REMOTE_USER="${JURECA_HOST%@*}"
[ "$REMOTE_USER" = "$JURECA_HOST" ] && REMOTE_USER="$USER"
REMOTE_DIR="${JURECA_DIR:-/p/scratch/westai0070/$REMOTE_USER/tt-bench}"

# What the benchmark actually needs: the tasks, the code that reads them, and
# the job scripts. Everything else is local clutter or remote output.
PATHS=(
    data/tasks
    src
    scripts
    tests
    jureca
    pyproject.toml
    uv.lock
)

EXCLUDES=(
    --exclude '__pycache__/'
    --exclude '*.pyc'
    --exclude '.venv/'
    --exclude '.venv-ttbench/'
    --exclude '.cache/'
    --exclude '.guide-cache/'
    --exclude 'benchmark_results/'
    --exclude 'slurm_logs/'
    --exclude '.env'
)

banner "Plan"
echo "  from : $PROJECT_ROOT"
echo "  to   : $JURECA_HOST:$REMOTE_DIR"
echo "  paths: ${PATHS[*]}"
$GO && warn "LIVE transfer (mirrors deletions)" || ok "dry run — nothing will be written"

banner "Local corpus"
for d in data/tasks/*/; do
    printf "  %6s tasks  %s\n" "$(find "$d" -name '*.json' | wc -l)" "$d"
done

RSYNC_OPTS=(-az --human-readable --itemize-changes --delete "${EXCLUDES[@]}")
$GO || RSYNC_OPTS+=(--dry-run)

banner "Transfer"
if $GO; then
    ssh "$JURECA_HOST" "mkdir -p '$REMOTE_DIR'" || {
        fail "Could not reach $JURECA_HOST or create $REMOTE_DIR"; exit 1; }
fi

# --relative keeps each path under the same layout on the far side.
rsync "${RSYNC_OPTS[@]}" --relative "${PATHS[@]}" "$JURECA_HOST:$REMOTE_DIR/" \
    | tee /tmp/tt-bench-sync.$$ || { fail "rsync failed"; exit 1; }

CHANGED=$(grep -cvE '^(sending|sent|total|$|created |\.d\.\.\.)' /tmp/tt-bench-sync.$$ || true)
DELETED=$(grep -c '^deleting ' /tmp/tt-bench-sync.$$ || true)
rm -f /tmp/tt-bench-sync.$$

banner "Summary"
echo "  entries changed: $CHANGED"
echo "  entries deleted: $DELETED"
if $GO; then
    ok "Synced to $JURECA_HOST:$REMOTE_DIR"
    echo ""
    echo "  Next, on the JURECA login node:"
    echo "    cd $REMOTE_DIR"
    echo "    bash jureca/setup.sh          # only if the venv is not built yet"
    echo "    TIERS=2 bash jureca/submit_all.sh"
else
    warn "Dry run only. Re-run with --go to transfer."
fi
