#!/bin/bash
#SBATCH --job-name=gen_rsa_codeT_results
#SBATCH --output=gen_rsa_codeT_results.out
#SBATCH --error=gen_rsa_codeT_results.err
#SBATCH --mem=128G
#SBATCH --time=10:00:00

N_PROGRAMS=100
N_RSA_TESTS=100
N_OUT_PROGRAMS=100
N_RANSAC_SAMPLES=100
PROGRAMS_DIR="/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_programs_k100.jsonl"
RSA_TESTS_DIR="/home/rrsood/CodeGen/codellama_runs/rsa-testcases/codellama_humaneval_rsa_100programs_100testcases_k100.jsonl"
RSA_CONST_MATRIX_DIR="/home/rrsood/CodeGen/codellama_runs/rsa-testcases/codellama_humaneval_rsa_100programs_100testcases_const_matrix_k100.npy"
RES_DIR="/home/rrsood/CodeGen/codellama_runs/rsa-codet-results/codellama_humaneval_codet_rsa_k100.jsonl"

python3 -m pragmatic-code-generation.scripts.gen_rsa_codeT_results \
        --num_programs $N_PROGRAMS \
        --num_rsa_tests $N_RSA_TESTS \
        --num_out_programs $N_OUT_PROGRAMS \
        --num_ransac_samples $N_RANSAC_SAMPLES \
        --programs_dir $PROGRAMS_DIR \
        --rsa_tests_dir $RSA_TESTS_DIR \
        --rsa_const_matrix_dir $RSA_CONST_MATRIX_DIR \
        --res_dir $RES_DIR