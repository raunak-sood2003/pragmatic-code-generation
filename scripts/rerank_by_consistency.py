import numpy as np
import json
from human_eval.data import write_jsonl
from ..src.utils import subsample_matrix
from tqdm import tqdm
import fire
import random

def rerank_by_consistency(num_programs, num_tests, num_select_tests, programs_dir, tests_dir, const_matrix_dir, res_dir):
    N_HUMAN_EVAL_EXAMPLES = 164
    const_matrices = np.load(const_matrix_dir)

    with open(programs_dir) as f:
        json_programs = [json.loads(line) for line in f]
    with open(tests_dir) as f:
        json_tests = [json.loads(line) for line in f]

    res_programs = []
    for i in range(N_HUMAN_EVAL_EXAMPLES):
        task_id = "HumanEval/%d" % i
        programs = [json_program['completion'] for json_program in json_programs[i * num_programs : i * num_programs + num_programs]]
        tests = np.array([json_test['completion'] for json_test in json_tests[i * num_tests : i * num_tests + num_tests]])
        const_matrix = const_matrices[i]
        
        # De-duplicate programs and test
        unique_programs = {}
        unique_tests = {}
        for j, program in enumerate(programs):
            unique_programs[program] = j
        for j, test in enumerate(tests):
            unique_tests[test] = j
        
        programs = list(unique_programs.keys())
        programs_idx = np.array(list(unique_programs.values()))
        tests = np.array(list(unique_tests.keys()))
        tests_idx = np.array(list(unique_tests.values()))

        # De-duplicated consistency matrix
        const_matrix = const_matrix[tests_idx, :][:, programs_idx]      
        
        # Sub-sample matrix based on num_select_tests
        if const_matrix.shape[0] >= num_select_tests:
            sample_matrix, _, _ = subsample_matrix(const_matrix, num_select_tests, const_matrix.shape[1])
        else:
            sample_matrix = const_matrix
        
        num_consistent_arr = sample_matrix.sum(axis=0).tolist()
        consistent_pair_map = {}
        
        for j in range(len(num_consistent_arr)):
            num_consistent = num_consistent_arr[j]
            program = programs[j]
            if num_consistent in consistent_pair_map:
                consistent_pair_map[num_consistent].append(program)
            else:
                consistent_pair_map[num_consistent] = [program]
        
        unique_consistencies_sorted = sorted(list(consistent_pair_map.keys()), reverse=True)

        for num_consistent in unique_consistencies_sorted:
            shuffled_programs = consistent_pair_map[num_consistent]
            random.shuffle(shuffled_programs)
            for rerank_program in shuffled_programs:
                res_programs.append({"task_id" : task_id, "completion" : rerank_program})
    
    write_jsonl(res_dir, res_programs)

if __name__ == '__main__':
    # fire.Fire(rerank_by_consistency)

    NUM_PROGRAMS=100
    NUM_TESTS=100
    # NUM_SELECT_TESTS=1
    PROGRAMS_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/data/codellama-13b/generations/codellama_humaneval_programs_k100.jsonl"
    TESTS_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/data/codellama-13b/generations/codellama_humaneval_tests_k100_temp0.8.jsonl"
    CONST_MATRIX_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/data/codellama-13b/const-matrices/codellama_humaneval_k100_const_matrix.npy"
    # RES_DIR="/home/rrsood/CodeGen/pragmatic-code-generation/data/codellama-13b/rerank-by-consistency/%dtest/codellama_humaneval_rerank_consistency_%dtest_k100[1].jsonl" % (NUM_SELECT_TESTS, NUM_SELECT_TESTS)

    tests = [1, 2, 3, 4, 5, 10, 25, 50, 75, 100]
    i = 3
    for test in tests:
        RES_DIR="/home/rrsood/CodeGen/rerank-consistencies/codellama-13b/%dtest/codellama_humaneval_rerank_consistency_%dtest_run%d.jsonl" % (test, test, i)
        rerank_by_consistency(NUM_PROGRAMS, NUM_TESTS, test, PROGRAMS_DIR, TESTS_DIR, CONST_MATRIX_DIR, RES_DIR)

    # rerank_by_consistency(NUM_PROGRAMS, NUM_SELECT_TESTS, PROGRAMS_DIR, CONST_MATRIX_DIR, RES_DIR)