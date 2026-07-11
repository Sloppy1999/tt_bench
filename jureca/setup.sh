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
# This installs everything in a Python venv at ~/.venv-ttbench so Slurm jobs
# can activate it without module conflicts.
# =============================================================================

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

banner() { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC} $1"; }
fail()  { echo -e "  ${RED}✘${NC} $1"; }

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv-ttbench"

# Move pip cache to scratch to avoid HOME quota
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export UV_CACHE_DIR="$PROJECT_DIR/.cache/uv"

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

# JURECA may have python 3.12 via module; fall back to uv-managed python
MODULE_PYTHON=""
if command -v python3.12 &>/dev/null; then
    MODULE_PYTHON="$(command -v python3.12)"
    ok "System Python 3.12: $MODULE_PYTHON"
elif command -v python3 &>/dev/null; then
    PYVER=$(python3 --version 2>&1 | awk '{print $2}')
    ok "System Python: $PYVER"
    MODULE_PYTHON="$(command -v python3)"
fi

if [ -d "$VENV_DIR" ]; then
    ok "Virtual environment already exists at $VENV_DIR"
else
    if [ -n "$MODULE_PYTHON" ]; then
        uv venv "$VENV_DIR" --python "$MODULE_PYTHON"
    else
        uv venv "$VENV_DIR" --python 3.12
    fi
    ok "Created venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

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

# ---------------------------------------------------------------------------
# 5. Verify GPU access (will only work on compute nodes, but check CUDA libs)
# ---------------------------------------------------------------------------
banner "Verifying installation"

echo ""
echo "Python: $(python --version)"
echo "vLLM:   $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'check on compute node')"
echo "uv:     $(uv --version)"
echo ""
ok "Setup complete"
echo ""
echo "Next: copy your .env with API keys (optional for local models)"
echo "  cp $PROJECT_DIR/.env $PROJECT_DIR/.env.jureca.bak"
echo ""
echo "Then submit jobs:"
echo "  bash jureca/submit_all.sh"
