#!/bin/bash
set -euo pipefail

# ── Configurable parameters (override via env) ───────────────────
MODEL_DIR="${MODEL_DIR:-/home/scratch.trt_llm_data/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0}"
DISAGG_PORT="${DISAGG_PORT:-8000}"
ISL="${ISL:-4096}"
OSL="${OSL:-512}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
NUM_PROMPTS="${NUM_PROMPTS:-320}"
STREAMING="${STREAMING:-true}"
RESULT_DIR="${RESULT_DIR:-./results}"

mkdir -p "${RESULT_DIR}"

echo "Benchmark config: ISL=${ISL}, OSL=${OSL}, concurrency=${MAX_CONCURRENCY}, prompts=${NUM_PROMPTS}, streaming=${STREAMING}"

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
    $(if [ "${STREAMING}" = "false" ]; then echo "--non-streaming"; fi)
