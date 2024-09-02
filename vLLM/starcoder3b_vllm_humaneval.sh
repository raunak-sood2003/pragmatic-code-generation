#!/bin/bash
#SBATCH --job-name=starcoder3b_vllm_humaneval
#SBATCH --output=starcoder3b_vllm_humaneval.out
#SBATCH --error=starcoder3b_vllm_humaneval.err
#SBATCH --mem=48G
#SBATCH --nodelist=babel-4-28
#SBATCH --time=2:00:00

MODEL_NAME="bigcode/starcoder2-3b"
PORT=8080
TO_GEN_TESTS=0
NUM_GENERATIONS=100
TEMPERATURE=0.8
TOP_P=0.95
MAX_TOKENS=128
INSTRUCTION_TUNED=0
OUTPUT_DIR="/home/rrsood/CodeGen"

python3 -m pragmatic-code-generation.vLLM.vllm_prompting \
        --model_name $MODEL_NAME \
        --port $PORT \
        --to_gen_tests $TO_GEN_TESTS \
        --num_generations $NUM_GENERATIONS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --max_tokens $MAX_TOKENS \
        --instruction_tuned $INSTRUCTION_TUNED \
        --output_dir $OUTPUT_DIR \