#!/bin/bash
#SBATCH --job-name=codellama13b_vllm_mbpp_test
#SBATCH --output=codellama13b_vllm_mbpp_test.out
#SBATCH --error=codellama13b_vllm_mbpp_test.err
#SBATCH --mem=48G
#SBATCH --nodelist=shire-1-10
#SBATCH --time=10:00:00

MODEL_NAME="codellama/CodeLlama-13b-hf"
PORT=8070
TO_GEN_TESTS=0
NUM_GENERATIONS=100
TEMPERATURE=0.8
TOP_P=0.95
MAX_TOKENS=128
MBPP_SPLIT="test"
MBPP_DIR="/home/rrsood/CodeGen/mbpp/mbpp_test.jsonl"
OUTPUT_DIR="/home/rrsood/CodeGen/codellama13b_mbpp_test_programs_k100.jsonl"

python3 -m pragmatic-code-generation.vLLM.vllm_prompting \
        --model_name $MODEL_NAME \
        --port $PORT \
        --to_gen_tests $TO_GEN_TESTS \
        --num_generations $NUM_GENERATIONS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --max_tokens $MAX_TOKENS \
        --mbpp_split $MBPP_SPLIT \
        --mbpp_dir $MBPP_DIR \
        --output_dir $OUTPUT_DIR \