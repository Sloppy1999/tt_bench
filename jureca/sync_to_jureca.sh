#!/usr/bin/env bash
# =============================================================================
# Deploy TT-Bench to JURECA
# =============================================================================
# The project directory on JURECA is a git checkout of this repository, so the
# safe way to update it is to push a branch and pull it there. A file-level
# mirror would delete whatever has been committed or produced on the cluster
# since the local tree last diverged — results, logs and plotting scripts that
# only exist there.
#
# This script therefore inspects the remote checkout, reports whether a pull
# would be clean, and prints the commands to run. It changes nothing by itself.
#
# Usage:
#   bash jureca/sync_to_jureca.sh              # inspect, using ssh host "jureca"
#   JURECA_HOST=user@host bash jureca/sync_to_jureca.sh
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

HOST="${JURECA_HOST:-jureca}"
BRANCH="$(git branch --show-current)"
LOCAL_HEAD="$(git rev-parse --short HEAD)"

banner "Local"
echo "  branch : $BRANCH"
echo "  head   : $LOCAL_HEAD  $(git log -1 --format=%s)"
if [ -n "$(git status --porcelain)" ]; then
    warn "working tree is dirty — commit before deploying"
else
    ok "working tree clean"
fi
if git rev-parse --verify -q "origin/$BRANCH" >/dev/null; then
    ok "branch exists on origin ($(git rev-list --count "origin/$BRANCH..HEAD") commits unpushed)"
else
    warn "branch is not on origin yet — it must be pushed before JURECA can pull it"
fi

banner "Remote checkout"
REMOTE_INFO=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" 'bash -s' <<'EOF' 2>/dev/null || true
# $USER is not always set in a non-login ssh shell.
me=$(id -un)
for d in /p/scratch/westai0070/$me/tt_bench /p/scratch/westai0070/$me/tt-bench; do
    [ -d "$d/.git" ] || continue
    echo "DIR=$d"
    echo "BRANCH=$(git -C "$d" branch --show-current)"
    echo "HEAD=$(git -C "$d" rev-parse --short HEAD)"
    echo "DIRTY=$(git -C "$d" status --porcelain | wc -l)"
    echo "VENV=$([ -d "$d/.venv-ttbench" ] && echo yes || echo no)"
    exit 0
done
echo "DIR="
EOF
)

REMOTE_DIR=$(sed -n 's/^DIR=//p' <<<"$REMOTE_INFO")
if [ -z "$REMOTE_DIR" ]; then
    fail "Could not inspect a git checkout on $HOST"
    echo "    Either no checkout exists under /p/scratch/westai0070/<user>/tt_bench,"
    echo "    or the connection needs re-authenticating. JURECA uses MFA, so open a"
    echo "    session yourself first (ssh $HOST) and leave it running — the"
    echo "    ControlPersist master in ~/.ssh/config lets this script reuse it."
    exit 1
fi
ok "checkout: $REMOTE_DIR"
echo "  branch : $(sed -n 's/^BRANCH=//p' <<<"$REMOTE_INFO")"
echo "  head   : $(sed -n 's/^HEAD=//p' <<<"$REMOTE_INFO")"
echo "  venv   : $(sed -n 's/^VENV=//p' <<<"$REMOTE_INFO")"

REMOTE_DIRTY=$(sed -n 's/^DIRTY=//p' <<<"$REMOTE_INFO")
if [ "${REMOTE_DIRTY:-0}" -gt 0 ]; then
    warn "$REMOTE_DIRTY uncommitted entries on JURECA — results and logs live there"
    echo "    Commit or stash them there before pulling; do not overwrite them."
else
    ok "remote working tree clean"
fi

banner "To deploy"
cat <<EOF
  1. locally:
       git push -u origin $BRANCH

  2. on $HOST, in $REMOTE_DIR:
       git stash -u            # only if the remote tree is dirty
       git fetch origin
       git checkout $BRANCH
       git pull --ff-only origin $BRANCH

  3. run tier 2 (submit_all.sh takes a --tiers flag):
       bash jureca/submit_all.sh -n -t 2      # dry run first
       bash jureca/submit_all.sh -t 2
EOF
warn "Nothing was changed on either side by this script."
