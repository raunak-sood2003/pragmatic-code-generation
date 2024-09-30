#!/bin/bash
#SBATCH --job-name=gen_codeT_canonical
#SBATCH --output=gen_codeT_canonical.out
#SBATCH --error=gen_codeT_canonical.err
#SBATCH --mem=128G
#SBATCH --time=8:00:00

NUM_PROGRAMS=100
NUM_TESTS=100
NUM_RANSAC_SAMPLES=100
NUM_OUT_PROGRAMS=100
PROGRAMS_DIR=/home/rrsood/CodeGen/data/mbpp/codellama-13b/generations/codellama13b_mbpp_sanitize_programs_k100.jsonl
CANONICAL_PROGRAMS_DIR=/home/rrsood/CodeGen/data/mbpp/codellama-13b/mbpp_sanitize_canonical_solutions.jsonl
TESTS_DIR=/home/rrsood/CodeGen/data/mbpp/codellama-13b/generations/codellama13b_mbpp_sanitize_tests_k100.jsonl
CONST_MATRIX_DIR=/home/rrsood/CodeGen/data/mbpp/codellama-13b/const-matrices/codellama_mbpp_k100_const_matrix.npy

cd evalplus
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd ..

for n_reps in 1; 
do
    for n_inp_tests in 3 4;
    do
        res_programs_dir="/home/rrsood/CodeGen/prelim-results/mbpp/codellama-13b/codeT/${n_inp_tests}_test/codellama_mbpp_codeT_programs_${n_inp_tests}test_run${n_reps}.jsonl"
        res_json_dir="/home/rrsood/CodeGen/prelim-results/mbpp/codellama-13b/codeT/${n_inp_tests}_test/codellama_mbpp_codeT_result_${n_inp_tests}test_run${n_reps}.json"

        python3 -m pragmatic-code-generation.scripts.gen_codeT_canonical \
                --num_programs $NUM_PROGRAMS \
                --num_tests $NUM_TESTS \
                --num_input_tests $n_inp_tests \
                --num_ransac_samples $NUM_RANSAC_SAMPLES \
                --num_out_programs $NUM_OUT_PROGRAMS \
                --programs_dir $PROGRAMS_DIR \
                --canonical_programs_dir $CANONICAL_PROGRAMS_DIR \
                --tests_dir $TESTS_DIR \
                --const_matrix_dir $CONST_MATRIX_DIR \
                --res_programs_dir $res_programs_dir \
                --res_json_dir $res_json_dir
    done
    
done


