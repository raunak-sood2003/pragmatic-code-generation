#!/bin/bash
#SBATCH --job-name=verify_const_matrices
#SBATCH --output=verify_const_matrices.out
#SBATCH --error=verify_const_matrices.err
#SBATCH --mem=128G
#SBATCH --time=10:00:00


N_PROGRAMS=100
N_TESTS=10
PROGRAMS_DIR="/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_programs_k100.jsonl"
TESTS_DIR="/home/rrsood/CodeGen/codellama_runs/rsa-testcases/codellama_humaneval_rsa_100programs_100testcases_k10.jsonl"
CONST_MATRIX_DIR="/home/rrsood/CodeGen/codellama_runs/rsa-testcases/codellama_humaneval_rsa_100programs_100testcases_const_matrix_k10.npy"

python3 -m pragmatic-code-generation.scripts.verify_const_matrices \
        --num_programs $N_PROGRAMS \
        --num_tests $N_TESTS \
        --programs_dir $PROGRAMS_DIR \
        --tests_dir $TESTS_DIR \
        --const_matrix_dir $CONST_MATRIX_DIR \