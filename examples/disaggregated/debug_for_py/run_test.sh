#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO=1
# Parse tp/pp from yaml configs to build log filename
CTX_TP=$(grep 'tensor_parallel_size:' ctx_config.yaml | awk '{print $2}')
CTX_PP=$(grep 'pipeline_parallel_size:' ctx_config.yaml | awk '{print $2}')
GEN_TP=$(grep 'tensor_parallel_size:' gen_config.yaml | awk '{print $2}')
GEN_PP=$(grep 'pipeline_parallel_size:' gen_config.yaml | awk '{print $2}')
export TLLM_KV_TRANSFER_PERF_LOG_FILE="log_kv_transfer_perf_ctx_tp${CTX_TP:-1}_pp${CTX_PP:-1}_gen_tp${GEN_TP:-1}_pp${GEN_PP:-1}"
# ── Configurable parameters (override via env) ───────────────────
MODEL_DIR="${MODEL_DIR:-/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct}"
CTX_GPUS="${CTX_GPUS:-0,1,2,3}"          # ctx_config.yaml: pp=2, tp=1 → 2 GPUs
GEN_GPUS="${GEN_GPUS:-4,5,6,7}"          # gen_config.yaml: tp=2, pp=1 → 2 GPUs

PREFILL_PORT=8001
DECODE_PORT=8002
DISAGG_PORT=8000

ISL="${ISL:-4096}"
OSL="${OSL:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
NUM_PROMPTS="${NUM_PROMPTS:-320}"
STREAMING="${STREAMING:-true}"
RESULT_DIR="${RESULT_DIR:-./results}"

PREFILL_HEALTH="http://localhost:${PREFILL_PORT}/health"
DECODE_HEALTH="http://localhost:${DECODE_PORT}/health"
DISAGG_HEALTH="http://localhost:${DISAGG_PORT}/health"

POLL_INTERVAL=5
PREFILL_TIMEOUT=600
DECODE_TIMEOUT=600
DISAGG_TIMEOUT=120

# ── Helper functions ─────────────────────────────────────────────
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cleanup() {
    log "Cleaning up background processes..."
    for pid in "${PREFILL_PID:-}" "${DECODE_PID:-}" "${DISAGG_PID:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log "Killing PID $pid"
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    log "Cleanup done."
}
trap cleanup EXIT

wait_for_health() {
    local url="$1"
    local name="$2"
    local timeout="$3"
    local elapsed=0

    log "Waiting for $name to be ready at $url (timeout=${timeout}s)..."
    while (( elapsed < timeout )); do
        if curl -sf "$url" >/dev/null 2>&1; then
            log "$name is ready! (took ${elapsed}s)"
            return 0
        fi
        sleep "$POLL_INTERVAL"
        elapsed=$(( elapsed + POLL_INTERVAL ))
    done
    log "ERROR: $name did not become ready within ${timeout}s"
    return 1
}

# ── Step 1: Start context (prefill) server ───────────────────────
log "Starting context server on GPUs=${CTX_GPUS}, port=${PREFILL_PORT}..."
CUDA_VISIBLE_DEVICES=${CTX_GPUS} trtllm-serve ${MODEL_DIR} \
    --host localhost --port ${PREFILL_PORT} \
    --extra_llm_api_options ./ctx_config.yaml &> log_ctx_0.log &
PREFILL_PID=$!
log "Context server started (PID=${PREFILL_PID}), log -> log_ctx_0.log"

# ── Step 2: Start generation (decode) server ─────────────────────
log "Starting generation server on GPUs=${GEN_GPUS}, port=${DECODE_PORT}..."
CUDA_VISIBLE_DEVICES=${GEN_GPUS} trtllm-serve ${MODEL_DIR} \
    --host localhost --port ${DECODE_PORT} \
    --extra_llm_api_options ./gen_config.yaml &> log_gen_0.log &
DECODE_PID=$!
log "Generation server started (PID=${DECODE_PID}), log -> log_gen_0.log"

# ── Step 3: Wait for both servers to be healthy ──────────────────
wait_for_health "$PREFILL_HEALTH" "Context server"    "$PREFILL_TIMEOUT"
wait_for_health "$DECODE_HEALTH"  "Generation server" "$DECODE_TIMEOUT"

# ── Step 4: Start disaggregated router ───────────────────────────
log "Starting disaggregated router on port=${DISAGG_PORT}..."
trtllm-serve disaggregated -c disagg_config.yaml &> log_disagg.log &
DISAGG_PID=$!
log "Disagg router started (PID=${DISAGG_PID}), log -> log_disagg.log"

wait_for_health "$DISAGG_HEALTH" "Disagg router" "$DISAGG_TIMEOUT"



curl http://localhost:${DISAGG_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "${MODEL_DIR}",
        "prompt": "NVIDIA is a great company because they make the best GPUs for gaming, AI, and deep machine learning workloads in the entire world today",
        "max_tokens": 16,
        "temperature": 0
    }' -w "\n"

# ── Step 5: Run benchmark ────────────────────────────────────────
log "Running benchmark (ISL=${ISL}, OSL=${OSL}, concurrency=${MAX_CONCURRENCY}, prompts=${NUM_PROMPTS})..."
mkdir -p "${RESULT_DIR}"

python -m tensorrt_llm.serve.scripts.benchmark_serving \
    --model "${MODEL_DIR}" \
    --backend openai \
    --host localhost \
    --port "${DISAGG_PORT}" \
    --dataset-name random \
    --random-ids \
    --random-input-len "${ISL}" \
    --random-output-len "${OSL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --ignore-eos \
    --save-result \
    --result-dir "${RESULT_DIR}" \
    --result-filename "result.json" \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    $(if [ "${STREAMING}" = "false" ]; then echo "--non-streaming"; fi) \
    2>&1 | tee benchmark_output.log

BENCH_EXIT=${PIPESTATUS[0]}

if [[ $BENCH_EXIT -eq 0 ]]; then
    log "Benchmark PASSED."
else
    log "Benchmark FAILED (exit code=${BENCH_EXIT})."
fi
pkill -9 trtllm

exit $BENCH_EXIT
