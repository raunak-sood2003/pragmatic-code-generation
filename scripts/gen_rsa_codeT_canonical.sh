#!/bin/bash
#SBATCH --job-name=gen_rsa_codeT_canonical
#SBATCH --output=gen_rsa_codeT_canonical.out
#SBATCH --error=gen_rsa_codeT_canonical.err
#SBATCH --mem=128G
#SBATCH --time=10:00:00

NUM_PROGRAMS=100
NUM_TESTS=100
NUM_INPUT_TESTS=1
NUM_RANSAC_SAMPLES=100
NUM_OUT_PROGRAMS=100
PROGRAMS_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/codellama-13b/generations/codellama_humaneval_programs_k100.jsonl"
TESTS_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/codellama-13b/generations/codellama_humaneval_tests_k100.jsonl"
CONST_MATRIX_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/codellama-13b/const-matrices/codellama_humaneval_k100_const_matrix.npy"
CANONICAL_CONST_MATRIX_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/codellama-13b/const-matrices/codellama_humaneval_canonical_const_matrix_k100.npy"
RES_DIR="."

python3 -m pragmatic-code-generation.scripts.gen_rsa_codeT_canonical \
        --num_programs $N_PROGRAMS \
        --num_tests $N_TESTS \
        --num_input_tests $NUM_INPUT_TESTS \
        --num_ransac_samples $N_RANSAC_SAMPLES \
        --num_out_programs $N_OUT_PROGRAMS \
        --programs_dir $PROGRAMS_DIR \
        --tests_dir $TESTS_DIR \
        --const_matrix_dir $CONST_MATRIX_DIR \
        --canonical_const_matrix_dir $CANONICAL_CONST_MATRIX_DIR \
        --res_dir $RES_DIR