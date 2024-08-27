#!/bin/bash
#SBATCH --job-name=vllm_prompting_humaneval
#SBATCH --output=vllm_prompting_humaneval.out
#SBATCH --error=vllm_prompting_humaneval.err
#SBATCH --mem=48G
#SBATCH --nodelist=babel-8-11
#SBATCH --time=5:00:00

MODEL_NAME="codellama/CodeLlama-13b-hf"
PORT=8070
TO_GEN_TESTS=0
NUM_GENERATIONS=1000
TEMPERATURE=0.8
TOP_P=0.95
MAX_TOKENS=128
OUTPUT_DIR="/home/rrsood/CodeGen"

python3 -m pragmatic-code-generation.vLLM.vllm_prompting \
        --model_name $MODEL_NAME \
        --port $PORT \
        --to_gen_tests $TO_GEN_TESTS \
        --num_generations $NUM_GENERATIONS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --max_tokens $MAX_TOKENS \
        --output_dir $OUTPUT_DIR \