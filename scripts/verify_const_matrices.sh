#!/bin/bash
#SBATCH --job-name=verify_const_matrices
#SBATCH --output=verify_const_matrices.out
#SBATCH --error=verify_const_matrices.err
#SBATCH --mem=128G
#SBATCH --time=10:00:00


N_PROGRAMS=1000
N_TESTS=1000
PROGRAMS_DIR="/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_programs_k1000.jsonl"
TESTS_DIR="/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_tests_k1000.jsonl"
CONST_MATRIX_DIR="/home/rrsood/CodeGen/codellama_humaneval_k1000_const_matrix_0.npy"

python3 -m pragmatic-code-generation.scripts.verify_const_matrices \
        --num_programs $N_PROGRAMS \
        --num_tests $N_TESTS \
        --programs_dir $PROGRAMS_DIR \
        --tests_dir $TESTS_DIR \
        --const_matrix_dir $CONST_MATRIX_DIR \