#!/bin/bash
#SBATCH --job-name=gen_canonical_consistency_matrix
#SBATCH --output=gen_canonical_consistency_matrix.out
#SBATCH --error=gen_canonical_consistency_matrix.err
#SBATCH --mem=128G
#SBATCH --time=24:00:00

N_TESTS=100
PROGRAMS_DIR="/home/rrsood/CodeGen/humaneval_canonical_solutions.jsonl"
TESTS_DIR="/home/rrsood/CodeGen/incoder-6b/generations/incoder_humaneval_k100_tests.jsonl"
SAVE_DIR="/home/rrsood/CodeGen/incoder_humaneval_canonical_const_matrix_k100.npy"

python3 -m pragmatic-code-generation.scripts.gen_canonical_const_matrix \
        --num_tests $N_TESTS \
        --programs_dir $PROGRAMS_DIR \
        --tests_dir $TESTS_DIR \
        --save_dir $SAVE_DIR