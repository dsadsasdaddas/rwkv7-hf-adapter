#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [--quick|--full] [--models-root PATH] [--results PATH]

Runs V100 performance profiling rows for the RWKV-7 HF adapter:
  - fp16 serving-style prefill/decode
  - batch-size sweep
  - TTFT/TPOT
  - bitsandbytes none/8bit/4bit baseline

Default: --quick.
EOF
}

MODE=quick
MODELS_ROOT=/home/data/wangyue/models/rwkv7
RESULTS=/tmp/rwkv7_v100_perf_$(date +%Y%m%d_%H%M%S).jsonl
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) MODE=quick; shift ;;
    --full) MODE=full; shift ;;
    --models-root) MODELS_ROOT="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$(dirname "$RESULTS")"
if [[ -f /home/wzu/anaconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/wzu/anaconda3/etc/profile.d/conda.sh
  conda activate /home/data/wangyue/envs/rwkv7
fi
export PYTHONNOUSERSITE=1
export RWKV_V7_ON=1
export TORCHDYNAMO_DISABLE=1
export DS_IGNORE_CUDA_DETECTION=1
export PYTHONPATH=/home/data/wangyue/projects/flash-linear-attention:${REPO_ROOT}:${PYTHONPATH:-}
cd "$REPO_ROOT"

MODEL_04=${MODEL_04:-${MODELS_ROOT}/rwkv7-g1d-0.4b-hf}
MODEL_15=${MODEL_15:-${MODELS_ROOT}/rwkv7-g1g-1.5b-hf}
MODEL_29=${MODEL_29:-${MODELS_ROOT}/rwkv7-g1g-2.9b-hf}

run_cuda0() {
  echo "+ CUDA_VISIBLE_DEVICES=0 $*" | tee -a "${RESULTS%.jsonl}.commands.log"
  CUDA_VISIBLE_DEVICES=0 "$@"
}

# Small enough to run frequently.
run_cuda0 python bench/bench_speed.py \
  --hf-dir "$MODEL_04" --backend hf --dtype fp16 --attn-mode fused_recurrent \
  --hf-logits-to-keep 1 --fast-cache true --hf-decode-api rwkv7_forward_token \
  --fast-token-backend auto --prompt-tokens 128 --decode-tokens 32 --warmup 1 --runs 2 \
  --results "$RESULTS"

run_cuda0 python bench/bench_batch_sweep.py \
  --hf-dir "$MODEL_04" --dtype fp16 --attn-mode fused_recurrent \
  --fast-cache true --fast-decode-api true --fast-token-backend auto \
  --batch-sizes 1 2 4 --prompt-tokens 128 --decode-tokens 32 --warmup 1 --runs 2 \
  --results "$RESULTS"

run_cuda0 python bench/bench_quantization.py \
  --hf-dir "$MODEL_04" --dtype fp16 --attn-mode fused_recurrent \
  --quantizations none 8bit 4bit --prompt-tokens 128 --decode-tokens 16 \
  --decode-mode compare --warmup 1 --runs 1 --optional --results "$RESULTS"

run_cuda0 python bench/bench_ttft_tpot.py \
  --model "$MODEL_04" --dtype fp16 --attn-mode fused_recurrent \
  --isl 128 512 --batch-sizes 1 2 --reps 5 --warmup 1 \
  --decode-tokens 32 --generate-tokens 16 --generate-warmup 1 --results "$RESULTS"

if [[ "$MODE" == full ]]; then
  for model in "$MODEL_15" "$MODEL_29"; do
    run_cuda0 python bench/bench_speed.py \
      --hf-dir "$model" --backend hf --dtype fp16 --attn-mode fused_recurrent \
      --hf-logits-to-keep 1 --fast-cache true --hf-decode-api rwkv7_forward_token \
      --fast-token-backend auto --prompt-tokens 128 --decode-tokens 16 --warmup 1 --runs 1 \
      --results "$RESULTS"
    run_cuda0 python bench/bench_quantization.py \
      --hf-dir "$model" --dtype fp16 --attn-mode fused_recurrent \
      --quantizations none 8bit 4bit --prompt-tokens 128 --decode-tokens 8 \
      --decode-mode compare --warmup 1 --runs 1 --optional --results "$RESULTS"
  done
fi

echo "V100 performance profiling finished: $RESULTS"
