#!/usr/bin/env bash
# =============================================================================
# JURECA Environment Setup for TT-Bench + vLLM
# =============================================================================
# Run once on the login node to install uv, vLLM, and project dependencies.
#
# Usage:
#   chmod +x jureca/setup.sh
#   ./jureca/setup.sh
#
# This installs everything in a Python venv at <project>/.venv-ttbench (on
# scratch, NOT in $HOME) so Slurm jobs can activate it without module conflicts
# and without depending on a quota-limited home filesystem.
# =============================================================================

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

banner() { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "  ${RED}✘${NC} $1"; }

# ---------------------------------------------------------------------------
# Repoint a uv-created venv at uv's PATCH-LESS interpreter alias.
#
# `uv venv` records the exact patch directory it resolved (…/cpython-3.12.12-…).
# When uv later upgrades its managed CPython to 3.12.13 it PRUNES 3.12.12, and
# the venv's bin/python becomes a dangling symlink. bin/activate keeps working,
# so nothing complains until a batch job runs `python` and gets
# "command not found". uv also maintains …/cpython-3.12-… (no patch component)
# tracking the newest 3.12; patch releases are ABI-compatible, so pointing at
# the alias survives upgrades without rebuilding the venv.
# ---------------------------------------------------------------------------
pin_venv_to_alias() {
    local venv="$1" alias_py alias_bin
    [ -d "$venv" ] || return 0

    alias_py=$(ls -d "$UV_PYTHON_INSTALL_DIR"/cpython-3.12-*/bin/python3.12 2>/dev/null | head -n1)
    if [ -z "$alias_py" ] || [ ! -x "$alias_py" ]; then
        warn "No cpython-3.12 alias under $UV_PYTHON_INSTALL_DIR"
        warn "$venv stays pinned to an exact patch release and will break on the next uv upgrade."
        return 0
    fi

    alias_bin="$(dirname "$alias_py")"
    ln -sfn "$alias_py" "$venv/bin/python"
    if [ -f "$venv/pyvenv.cfg" ]; then
        sed -i "s|^home = .*|home = $alias_bin|" "$venv/pyvenv.cfg"
        sed -i "s|^version_info = .*|version_info = 3.12|" "$venv/pyvenv.cfg"
    fi
    ok "$(basename "$venv") pinned to patch-less alias: $alias_py"
}

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv-ttbench"

# Move pip cache to scratch to avoid HOME quota
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export UV_CACHE_DIR="$PROJECT_DIR/.cache/uv"

# uv installs its managed interpreters under $HOME by default. Keep them on
# scratch: HOME has a tiny quota and its contents are the first casualty of a
# filesystem incident — and a venv is worthless the moment its base interpreter
# disappears. Must be set BEFORE any `uv venv` / `uv sync` call.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PROJECT_DIR/.cache/uv-python}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-$PROJECT_DIR/.cache/uv-tools}"

# Redirect ALL caches to scratch — HOME quota is tiny on JURECA
export XDG_CACHE_HOME="$PROJECT_DIR/.cache/xdg"
export VLLM_CACHE_DIR="$PROJECT_DIR/.cache/vllm"
export FLASHINFER_WORKSPACE_DIR="$PROJECT_DIR/.cache/flashinfer"
export TMPDIR="$PROJECT_DIR/tmp"
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
         "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" \
         "$XDG_CACHE_HOME" "$VLLM_CACHE_DIR" "$FLASHINFER_WORKSPACE_DIR" \
         "$TMPDIR"

# ---------------------------------------------------------------------------
# 1. Install uv (isolated user install — no system packages needed)
# ---------------------------------------------------------------------------
banner "Installing uv"

if command -v uv &>/dev/null; then
    ok "uv already available: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv installed: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# 2. Create Python venv with appropriate Python
# ---------------------------------------------------------------------------
banner "Creating virtual environment"

# vLLM needs Python 3.12. The RHEL system interpreter on JURECA is 3.9, so ask
# uv for 3.12 explicitly instead of reusing whatever `python3` happens to be on
# PATH — a 3.9 venv resolves different (or no) vLLM wheels and only blows up
# much later, on a compute node, after a queue wait.
PY_REQ=3.12

if [ -d "$VENV_DIR" ]; then
    ok "Virtual environment already exists at $VENV_DIR"
else
    uv venv "$VENV_DIR" --python "$PY_REQ"
    ok "Created venv at $VENV_DIR (Python $PY_REQ)"
fi

pin_venv_to_alias "$VENV_DIR"

source "$VENV_DIR/bin/activate"

# Validate CAPABILITY, not presence. `[ -d "$VENV_DIR" ]` and bin/activate both
# pass on a venv whose interpreter symlink dangles, and bash then silently walks
# past the dead symlink to the system python3 — so the venv looks fine right up
# to the first `import vllm`.
if [ ! -x "$VENV_DIR/bin/python" ] || ! "$VENV_DIR/bin/python" --version >/dev/null 2>&1; then
    fail "No working interpreter at $VENV_DIR/bin/python"
    ls -l "$VENV_DIR/bin/python" 2>&1 | sed 's/^/    /'
    exit 1
fi
ok "Interpreter: $("$VENV_DIR/bin/python" --version)"

# ---------------------------------------------------------------------------
# 3. Install vLLM (GPU inference server)
# ---------------------------------------------------------------------------
banner "Installing vLLM"

pip install --upgrade pip
pip install vllm
ok "vLLM installed"

# ---------------------------------------------------------------------------
# 4. Install project dependencies
# ---------------------------------------------------------------------------
banner "Installing project dependencies"

cd "$PROJECT_DIR"
uv sync
ok "Project dependencies installed (uv sync)"

# `uv run` in run_benchmark.sbatch uses the PROJECT venv (.venv) and ignores an
# activated VIRTUAL_ENV, so .venv needs the same alias pinning as .venv-ttbench.
pin_venv_to_alias "$PROJECT_DIR/.venv"

# ---------------------------------------------------------------------------
# 5. Verify GPU access (will only work on compute nodes, but check CUDA libs)
# ---------------------------------------------------------------------------
banner "Verifying installation"

VENV_PY="$VENV_DIR/bin/python"

echo ""
echo "Python:  $("$VENV_PY" --version)"
echo "vLLM:    $("$VENV_PY" -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -n1)"
echo "torch:   $("$VENV_PY" -c 'import torch; print(torch.__version__)' 2>&1 | tail -n1)"
echo "uv:      $(uv --version)"
echo "uv pys:  $UV_PYTHON_INSTALL_DIR"
echo ""

# Fail loudly here rather than 6 hours into a queue wait: this is the last
# moment the import can be checked on a login node for free.
if ! "$VENV_PY" -c 'import vllm' >/dev/null 2>&1; then
    fail "'import vllm' failed in $VENV_PY — fix this BEFORE submitting jobs"
    exit 1
fi
ok "Setup complete"
echo ""
echo "Next: copy your .env with API keys (optional for local models)"
echo "  cp $PROJECT_DIR/.env $PROJECT_DIR/.env.jureca.bak"
echo ""
echo "Then submit jobs:"
echo "  bash jureca/submit_all.sh"
