#!/bin/bash
#SBATCH --job-name=gen_rsa_testcases
#SBATCH --output=gen_rsa_testcases.out
#SBATCH --error=gen_rsa_testcases.err
#SBATCH --mem=128G
#SBATCH --time=10:00:00

N_PROGRAMS=100
N_TESTS=100
N_RSA_TESTS=10
PROGRAMS_DIR="/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_programs_k100.jsonl"
TESTS_DIR="/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_tests_k100.jsonl"
CONST_MATRIX_DIR="/home/rrsood/CodeGen/codellama_runs/const-matrices/codellama_humaneval_k100_const_matrix.npy"
RES_CONST_MATRIX_DIR="/home/rrsood/CodeGen/codellama_runs/rsa-testcases/codellama_humaneval_rsa_${N_PROGRAMS}programs_${N_TESTS}testcases_const_matrix_k${N_RSA_TESTS}.npy"
RES_TESTCASE_DIR="/home/rrsood/CodeGen/codellama_runs/rsa-testcases/codellama_humaneval_rsa_${N_PROGRAMS}programs_${N_TESTS}testcases_k${N_RSA_TESTS}.jsonl"

python3 -m pragmatic-code-generation.scripts.gen_rsa_testcases \
        --num_programs $N_PROGRAMS \
        --num_tests $N_TESTS \
        --num_rsa_tests $N_RSA_TESTS \
        --programs_dir $PROGRAMS_DIR \
        --tests_dir $TESTS_DIR \
        --const_matrix_dir $CONST_MATRIX_DIR \
        --res_const_matrix_dir $RES_CONST_MATRIX_DIR \
        --res_testcase_dir $RES_TESTCASE_DIR