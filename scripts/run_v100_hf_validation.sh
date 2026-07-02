#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [--quick|--full] [--models-root PATH] [--results-dir PATH]

Runs the V100 HF validation matrix for the RWKV-7 HF adapter.

Modes:
  --quick  Runs representative 0.4B/1.5B/2.9B smoke checks.
  --full   Runs the broader V100 matrix, including quantized inference and
           longer-step smokes where V100 memory permits. Default: --quick.

Environment expected by the V100 server:
  conda env: /home/data/wangyue/envs/rwkv7
  optional FLA path: /home/data/wangyue/projects/flash-linear-attention

EOF
}

MODE=quick
MODELS_ROOT=/home/data/wangyue/models/rwkv7
RESULTS_DIR=/tmp/rwkv7_v100_hf_validation_$(date +%Y%m%d_%H%M%S)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) MODE=quick; shift ;;
    --full) MODE=full; shift ;;
    --models-root) MODELS_ROOT="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$RESULTS_DIR"

if [[ -f /home/wzu/anaconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/wzu/anaconda3/etc/profile.d/conda.sh
  conda activate /home/data/wangyue/envs/rwkv7
fi

export PYTHONNOUSERSITE=1
export RWKV_V7_ON=1
export TORCHDYNAMO_DISABLE=1
export DS_IGNORE_CUDA_DETECTION=1
export RWKV7_NATIVE_MODEL=${RWKV7_NATIVE_MODEL:-1}
export PYTHONPATH=/home/data/wangyue/projects/flash-linear-attention:${REPO_ROOT}:${PYTHONPATH:-}
cd "$REPO_ROOT"

MODEL_04=${MODEL_04:-${MODELS_ROOT}/rwkv7-g1d-0.4b-hf}
MODEL_15=${MODEL_15:-${MODELS_ROOT}/rwkv7-g1g-1.5b-hf}
MODEL_29=${MODEL_29:-${MODELS_ROOT}/rwkv7-g1g-2.9b-hf}
MODEL_72=${MODEL_72:-${MODELS_ROOT}/rwkv7-g1g-7.2b-hf}

run() {
  echo "+ $*" | tee -a "$RESULTS_DIR/commands.log"
  "$@" 2>&1 | tee -a "$RESULTS_DIR/run.log"
}

run_cuda0() {
  CUDA_VISIBLE_DEVICES=0 run "$@"
}

run_cuda01() {
  CUDA_VISIBLE_DEVICES=0,1 run "$@"
}

# 1. Syntax/import-level harness check.
run python -m py_compile \
  tests/test_native_trainer_resume_smoke.py \
  tests/test_native_peft_save_load_merge.py \
  tests/test_deepspeed_resume_smoke.py

# 2. Trainer resume and native TRL/RL matrix.
run_cuda0 python tests/test_native_trainer_resume_smoke.py --model "$MODEL_04" --device cuda --dtype fp32 --first-steps 1 --resume-steps 2 --batch-size 1 --length 32
run_cuda0 python tests/test_native_trainer_resume_smoke.py --model "$MODEL_15" --device cuda --dtype fp32 --first-steps 1 --resume-steps 2 --batch-size 1 --length 16
run_cuda0 python tests/test_native_trainer_resume_smoke.py --model "$MODEL_29" --device cuda --dtype fp32 --first-steps 1 --resume-steps 2 --batch-size 1 --length 8

run_cuda0 python tests/test_native_sft_smoke.py --model "$MODEL_29" --device cuda --dtype fp32 --max-steps 1 --batch-size 1 --max-length 8
run_cuda0 python tests/test_native_dpo_smoke.py --model "$MODEL_29" --dtype fp32 --max-steps 1 --batch-size 1 --max-length 8
run_cuda0 python tests/test_native_grpo_smoke.py --model "$MODEL_29" --dtype fp32 --max-steps 1 --batch-size 2 --max-completion-length 2

# 3. PEFT adapter lifecycle.
run_cuda0 python tests/test_native_peft_save_load_merge.py --model "$MODEL_04" --device cuda --dtype fp32 --steps 1
run_cuda0 python tests/test_native_peft_save_load_merge.py --model "$MODEL_15" --device cuda --dtype fp32 --steps 1
run_cuda0 python tests/test_native_peft_save_load_merge.py --model "$MODEL_29" --device cuda --dtype fp32 --steps 1

# 4. DeepSpeed resume. ZeRO2 is part of quick; ZeRO3 can be enabled by full mode.
run_cuda01 torchrun --standalone --nproc_per_node=2 tests/test_deepspeed_resume_smoke.py \
  --model "$MODEL_04" --zero-stage 2 --train-dtype fp32 --first-steps 1 --resume-steps 2 \
  --batch-size 1 --gradient-accumulation-steps 1 --max-length 16 \
  --results "$RESULTS_DIR/zero_resume_0p4b.jsonl"
run_cuda01 torchrun --standalone --nproc_per_node=2 tests/test_deepspeed_resume_smoke.py \
  --model "$MODEL_15" --zero-stage 2 --train-dtype fp32 --first-steps 1 --resume-steps 2 \
  --batch-size 1 --gradient-accumulation-steps 1 --max-length 8 \
  --results "$RESULTS_DIR/zero_resume_1p5b.jsonl"
run_cuda01 torchrun --standalone --nproc_per_node=2 tests/test_deepspeed_resume_smoke.py \
  --model "$MODEL_29" --zero-stage 2 --train-dtype fp32 --first-steps 1 --resume-steps 2 \
  --batch-size 1 --gradient-accumulation-steps 1 --max-length 8 \
  --results "$RESULTS_DIR/zero_resume_2p9b.jsonl"

if [[ "$MODE" == full ]]; then
  run_cuda01 torchrun --standalone --nproc_per_node=2 tests/test_deepspeed_resume_smoke.py \
    --model "$MODEL_04" --zero-stage 3 --train-dtype fp32 --first-steps 1 --resume-steps 2 \
    --batch-size 1 --gradient-accumulation-steps 1 --max-length 16 \
    --results "$RESULTS_DIR/zero3_resume_0p4b.jsonl"

  run_cuda0 python tests/test_hf_training_smoke.py --model "$MODEL_04" --device cuda --attn-mode fused_recurrent --train-dtype fp32 \
    --max-steps 20 --batch-size 1 --max-length 32 --backend trainer --results "$RESULTS_DIR/long_steps.jsonl"
  run_cuda0 python tests/test_hf_training_smoke.py --model "$MODEL_15" --device cuda --attn-mode fused_recurrent --train-dtype fp32 \
    --max-steps 10 --batch-size 1 --max-length 16 --backend trainer --results "$RESULTS_DIR/long_steps.jsonl"
  run_cuda0 python tests/test_native_sft_smoke.py --model "$MODEL_29" --device cuda --dtype fp32 --max-steps 5 --batch-size 1 --max-length 8
fi

# 5. Quantized inference matrix. 7.2B is included because it is low-risk in bnb 4/8bit on V100.
run_cuda0 python tests/test_native_bnb_quant_smoke.py --model "$MODEL_04" --device cuda --dtype fp16 --quantization both --max-new-tokens 2
run_cuda0 python tests/test_native_bnb_quant_smoke.py --model "$MODEL_15" --device cuda --dtype fp16 --quantization both --max-new-tokens 2
run_cuda0 python tests/test_native_bnb_quant_smoke.py --model "$MODEL_29" --device cuda --dtype fp16 --quantization both --max-new-tokens 2
run_cuda0 python tests/test_native_bnb_quant_smoke.py --model "$MODEL_72" --device cuda --dtype fp16 --quantization both --max-new-tokens 2

echo "V100 HF validation finished. Results: $RESULTS_DIR"
