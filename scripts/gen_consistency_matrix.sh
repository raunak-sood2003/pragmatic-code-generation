#!/bin/bash
#SBATCH --job-name=gen_consistency_matrix
#SBATCH --output=gen_consistency_matrix.out
#SBATCH --error=gen_consistency_matrix.err
#SBATCH --mem=128G
#SBATCH --time=24:00:00

N_PROGS=100
N_TESTS=100
PROGRAMS_DIR="/home/rrsood/CodeGen/data/mbpp/codellama-13b/generations/codellama13b_mbpp_sanitize_programs_k100.jsonl"
TESTS_DIR="/home/rrsood/CodeGen/data/mbpp/codellama-13b/generations/codellama13b_mbpp_sanitize_tests_k100.jsonl"
CANONICAL_PROGRAMS_DIR="/home/rrsood/CodeGen/data/mbpp/codellama-13b/mbpp_sanitize_canonical_solutions.jsonl"
SAVE_DIR="/home/rrsood/CodeGen/codellama_mbpp_k100_const_matrix.npy"

python3 -m pragmatic-code-generation.scripts.gen_consistency_matrix \
        --num_programs $N_PROGS \
        --num_tests $N_TESTS \
        --programs_dir $PROGRAMS_DIR \
        --tests_dir $TESTS_DIR \
        --canonical_programs_dir $CANONICAL_PROGRAMS_DIR \
        --save_dir $SAVE_DIR