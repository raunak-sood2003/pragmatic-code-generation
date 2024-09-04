#!/bin/bash
#SBATCH --job-name=gen_consistency_matrix
#SBATCH --output=gen_consistency_matrix.out
#SBATCH --error=gen_consistency_matrix.err
#SBATCH --mem=128G
#SBATCH --time=24:00:00

N_PROGS=100
N_TESTS=100
PROGRAMS_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/data/codellama-13b/generations/codellama_humaneval_programs_k100.jsonl"
TESTS_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/data/codellama-13b/generations/codellama_humaneval_tests_k100.jsonl"
SAVE_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/data/codellama-13b/const-matrices/codellama_humaneval_k100_const_matrix.npy"

python3 -m pragmatic-code-generation.scripts.gen_consistency_matrix \
        --num_programs $N_PROGS \
        --num_tests $N_TESTS \
        --programs_dir $PROGRAMS_DIR \
        --tests_dir $TESTS_DIR \
        --save_dir $SAVE_DIR