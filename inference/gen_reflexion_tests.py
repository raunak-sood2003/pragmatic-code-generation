import numpy as np
import json
from tqdm import tqdm
import random
import os
import fire
from ..src.rsa import RSA
from ..src.utils import create_const_matrix, write_jsonl

def generate_pragmatic_testcases(gen_programs, gen_tests, const_matrix, num_testcases):
    # De-deuplicating programs and tests
    unique_programs = {}
    unique_tests = {}
    for j, program in enumerate(gen_programs):
        unique_programs[program] = j
    for j, test in enumerate(gen_tests):
        unique_tests[test] = j
    
    program_idxs = np.array(list(unique_programs.values()))
    test_idxs = np.array(list(unique_tests.values()))
    
    gen_programs = np.array(gen_programs)[program_idxs].tolist()
    gen_tests = np.array(gen_tests)[test_idxs].tolist()
    const_matrix = np.concatenate((const_matrix[test_idxs, :][:, program_idxs], const_matrix[test_idxs, -1].reshape(-1, 1)), axis = 1)

    # Auto-regressively selecting pragmatic tests
    pragmatic_testcases = []
    const_matrix_update = np.copy(const_matrix)
    for _ in range(num_testcases):
        if const_matrix_update.size == 0:
            break
        P_L0 = RSA.normalize_rows(const_matrix_update)
        P_S1 = RSA.normalize_cols(P_L0)
        P_S1_truth = P_S1[:, -1]
        if P_S1_truth.sum() != 0: 
            rsa_testcase_idx = np.argmax(P_S1_truth)
            pragmatic_testcases.append(gen_tests[rsa_testcase_idx])
            # Auto-regressively update the const matrix
            const_matrix_update = const_matrix_update[:, const_matrix_update[rsa_testcase_idx, :] == 1]
            exclude = np.ones(const_matrix_update.shape[0], dtype = bool)
            exclude[rsa_testcase_idx] = 0
            const_matrix_update = const_matrix_update[exclude, :]
            gen_tests = np.array(gen_tests)[exclude].tolist()
            
    return pragmatic_testcases

def generate_reflexion_tests(num_programs, num_tests, num_out_tests, programs_path, tests_path, canonical_programs_path, save_dir, const_matrix_path = None):
    with open(programs_path) as f:
        json_programs = [json.loads(line) for line in f]
    with open(tests_path) as f:
        json_tests = [json.loads(line) for line in f]
    with open(canonical_programs_path) as f:
        json_canonical_programs = [json.loads(line) for line in f]
    
    task_ids = [program['task_id'] for program in json_canonical_programs]

    const_matrices = np.zeros([len(task_ids), num_tests, num_programs + 1])
    if const_matrix_path is not None:
        const_matrices = np.load(const_matrix_path)
    
    json_save_path = os.path.join(save_dir, 'reflexion_tests.jsonl')
    
    for i, task_id in tqdm(enumerate(task_ids)):
        tests = [json_test['completion'] for json_test in json_tests[i * num_tests : i * num_tests + num_tests]]
        programs = [json_program['completion'] for json_program in json_programs[i *  num_programs : i * num_programs + num_programs]]
        canonical_program = json_canonical_programs[i]['completion']
        programs.append(canonical_program)
        if const_matrix_path is not None:
            const_matrix = const_matrices[i]
        else:
            const_matrix = create_const_matrix(programs, tests)
            const_matrices[i] = const_matrix
            
        pragmatic_tests = generate_pragmatic_testcases(programs, tests, const_matrix, num_out_tests)
        consistent_tests = np.array(tests)[const_matrix[:, -1] == 1].tolist()
        random_tests = random.sample(consistent_tests, num_out_tests) if len(consistent_tests) >= num_out_tests else consistent_tests

        res_json = [{'task_id' : task_id, 'pragmatic_tests' : pragmatic_tests, 'random_tests' : random_tests}]
        write_jsonl(json_save_path, res_json, True)
    
    if const_matrix_path is not None:
        const_matrix_save_path = os.path.join(save_dir, 'const_matrix.npy')
        np.save(const_matrix_save_path, const_matrices)

if __name__ == '__main__':
    fire.Fire(generate_reflexion_tests)
        



