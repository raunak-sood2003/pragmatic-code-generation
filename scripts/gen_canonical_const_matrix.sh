#!/bin/bash
#SBATCH --job-name=gen_canonical_consistency_matrix
#SBATCH --output=gen_canonical_consistency_matrix.out
#SBATCH --error=gen_canonical_consistency_matrix.err
#SBATCH --mem=128G
#SBATCH --time=24:00:00

DATASET="humaneval"
N_TESTS=100
PROGRAMS_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/data/humaneval_canonical_solutions.jsonl"
TESTS_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/data/codellama-13b/generations/codellama_humaneval_tests_k100_temp0.8.jsonl"
SAVE_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/data/codellama-13b/const-matrices/codellama_humaneval_canonical_const_matrix_k100.npy"

python3 -m pragmatic-code-generation.scripts.gen_canonical_const_matrix \
        --dataset $DATASET \
        --num_tests $N_TESTS \
        --programs_dir $PROGRAMS_DIR \
        --tests_dir $TESTS_DIR \
        --save_dir $SAVE_DIR