#!/usr/bin/env bash
# =============================================================================
# Quick smoke test — single model, single challenge (tt-official-ch01-1comp)
# =============================================================================
# Usage:
#   sbatch --gres=gpu:1 jureca/test_single.sh
#
# Or interactively on a compute node:
#   bash jureca/test_single.sh
# =============================================================================
#SBATCH --job-name=tt-test
#SBATCH --partition=dc-hwai
#SBATCH --account=westai0070
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/p/scratch/westai0070/peral1/tt_bench/slurm_logs/tt-test-%j.out
#SBATCH --error=/p/scratch/westai0070/peral1/tt_bench/slurm_logs/tt-test-%j.err
#SBATCH --export=ALL

set -euo pipefail

BOLD='\033[1m'; GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
banner() { echo -e "\n${BOLD}━━━ $1 ━━━${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC} $1"; }
fail()  { echo -e "  ${RED}✘${NC} $1"; }

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Coder-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-test-qwen2.5-7b}"
PROJECT_DIR="${PROJECT_DIR:-/p/scratch/westai0070/peral1/tt_bench}"
CHALLENGE="tt-official-ch01-1comp"
VLLM_PORT=8000
MAX_WAIT=600

# ── Directories & Environment ────────────────────────────────────────────────
VENV_DIR="$PROJECT_DIR/.venv-ttbench"
HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export XDG_CACHE_HOME="$PROJECT_DIR/.cache/xdg"
export VLLM_CACHE_DIR="$PROJECT_DIR/.cache/vllm"
export FLASHINFER_WORKSPACE_DIR="$PROJECT_DIR/.cache/flashinfer"
export TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton"
export TMPDIR="$PROJECT_DIR/tmp"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_DIR" "$FLASHINFER_WORKSPACE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

module load CUDA/13 2>/dev/null || module load CUDA 2>/dev/null || true

banner "Activating environment"
source "$VENV_DIR/bin/activate"
ok "Python: $(python --version)"

# ── Start vLLM ───────────────────────────────────────────────────────────────
banner "Starting vLLM"
echo "  Model:    $MODEL_ID"
echo "  Port:     $VLLM_PORT"

pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

# ── Auto-detect correct --tool-call-parser for this model ─────────────────
# Can be overridden by setting TOOL_PARSER in the environment.
# Both --enable-auto-tool-choice and --tool-call-parser are REQUIRED
# by vLLM when the request includes tools.
if [ -z "${TOOL_PARSER:-}" ]; then
    case "$MODEL_ID" in
        *Qwen2.5*|*Qwen2.5-Coder*)  TOOL_PARSER="hermes" ;;
        *Qwen3*|*Qwen3.6*|*Qwen3-Coder*)  TOOL_PARSER="qwen3_coder" ;;
        *gemma-4*|*Gemma-4*)        TOOL_PARSER="gemma4" ;;
        *functiongemma*)            TOOL_PARSER="functiongemma" ;;
        *DeepSeek-Coder-V2*)        TOOL_PARSER="deepseek_v3" ;;  # experimental
        *DeepSeek*V3.1*)            TOOL_PARSER="deepseek_v31" ;;
        *DeepSeek*V3*)              TOOL_PARSER="deepseek_v3" ;;
        *Llama*|*llama*)            TOOL_PARSER="llama3_json" ;;
        *Mistral*|*mistral*)        TOOL_PARSER="mistral" ;;
        *hermes*|*Hermes*)          TOOL_PARSER="hermes" ;;
        *xlam*|*XLAM*)              TOOL_PARSER="xlam" ;;
        *)                          TOOL_PARSER="hermes" ;;  # safe fallback
    esac
fi
echo "  Tool parser: $TOOL_PARSER"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --served-model-name "$MODEL_ID" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    > "$PROJECT_DIR/slurm_logs/vllm_test.log" 2>&1 &

VLLM_PID=$!

# ── Wait for vLLM ────────────────────────────────────────────────────────────
banner "Waiting for vLLM"
WAITED=0
while ! curl -s --max-time 2 "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [ $WAITED -ge $MAX_WAIT ]; then
        fail "vLLM timeout after ${MAX_WAIT}s"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "  Waiting... ${WAITED}s"
done
ok "vLLM ready after ${WAITED}s"

# ── Quick API smoke test ─────────────────────────────────────────────────────
banner "API smoke test"
RESPONSE=$(curl -s --max-time 30 "http://localhost:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}],\"max_tokens\":10}")
echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  Response:', d['choices'][0]['message']['content'])" 2>/dev/null && \
    ok "API returns valid response" || \
    fail "API test failed: $RESPONSE"

# ── Run single challenge ─────────────────────────────────────────────────────
banner "Running challenge: $CHALLENGE"
cd "$PROJECT_DIR"

CHALLENGES_DIR="data/tasks/challenges_1comp"
OUT_DIR="$PROJECT_DIR/test_results/${MODEL_NAME}"
mkdir -p "$OUT_DIR"

uv run tt-benchmark \
    --provider openai \
    --model "$MODEL_ID" \
    --base-url "http://localhost:${VLLM_PORT}" \
    --api-key none \
    --task-type agentic_synthesis \
    --challenges-dir "$CHALLENGES_DIR" \
    --pattern "$CHALLENGE.json" \
    --output-dir "$OUT_DIR" \
    --save-report \
    --max-turns 25 \
    --max-tokens 8192 \
    --timeout 300 \
    --workers 1 \
    2>&1 | tee "$PROJECT_DIR/slurm_logs/bench_test_${MODEL_NAME}.log"

EXIT_CODE=$?

# ── Check result ─────────────────────────────────────────────────────────────
banner "Result"
if [ $EXIT_CODE -eq 0 ]; then
    # Find the latest report
    REPORT=$(ls -t "$OUT_DIR"/benchmark_*.json 2>/dev/null | head -1)
    if [ -n "$REPORT" ]; then
        echo ""
        python3 -c "
import json
r = json.load(open('$REPORT'))
print(f\"  File:     $REPORT\")
print(f\"  Tasks:    {r['total_tasks']}\")
print(f\"  Success:  {r['successful']}\")
print(f\"  Failed:   {r['failed']}\")
for res in r['results']:
    print(f\"  ─ {res['task_id']}: {'PASS' if res['success'] else 'FAIL'} ({res['metrics']['component_score']*100:.0f}%) tokens={res['tokens_used']}\")
        " 2>/dev/null
    fi
else
    fail "Benchmark exited with code $EXIT_CODE"
fi

# ── Stop vLLM ────────────────────────────────────────────────────────────────
banner "Stopping vLLM"
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
ok "Done"
