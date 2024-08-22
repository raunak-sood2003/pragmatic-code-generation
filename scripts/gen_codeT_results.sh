#!/bin/bash
#SBATCH --job-name=gen_codeT_results
#SBATCH --output=gen_codeT_results.out
#SBATCH --error=gen_codeT_results.err
#SBATCH --mem=128G
#SBATCH --time=10:00:00

N_PROGRAMS=100
N_TESTS=100
N_OUT_PROGRAMS=10
N_RANSAC_SAMPLES=100
PROGRAMS_DIR="/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_programs_k${N_PROGRAMS}.jsonl"
TESTS_DIR="/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_tests_k${N_TESTS}.jsonl"
CONST_MATRIX_DIR="/home/rrsood/CodeGen/codellama_runs/const-matrices/codellama_humaneval_k100_const_matrix.npy"
RES_DIR="/home/rrsood/CodeGen/codellama_runs/codeT-results/codellama_humaneval_codet_k${N_OUT_PROGRAMS}.jsonl"

python3 -m pragmatic-code-generation.scripts.gen_codeT_results \
        --num_programs $N_PROGRAMS \
        --num_tests $N_TESTS \
        --num_out_programs $N_OUT_PROGRAMS \
        --num_ransac_samples $N_RANSAC_SAMPLES \
        --programs_dir $PROGRAMS_DIR \
        --tests_dir $TESTS_DIR \
        --const_matrix_dir $CONST_MATRIX_DIR \
        --res_dir $RES_DIR