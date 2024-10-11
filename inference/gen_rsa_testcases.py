import numpy as np
from ..src.rsa import RSA
import json
from tqdm import tqdm
from human_eval.data import write_jsonl
import fire

def generate_rsa_testcases(num_programs, num_tests, num_rsa_tests, programs_dir, \
                           tests_dir, const_matrix_dir, res_const_matrix_dir, res_testcase_dir):
    '''
    Uses RSA to select the most informative test cases given the set of generated programs and test cases. 
    '''
    N_HUMAN_EVAL_EXAMPLES = 164
    consistency_matrices = np.load(const_matrix_dir)

    assert(consistency_matrices.shape[0] == N_HUMAN_EVAL_EXAMPLES)
    assert(consistency_matrices.shape[1] == num_tests)
    assert(consistency_matrices.shape[2] == num_programs)

    with open(programs_dir) as f:
        json_programs = [json.loads(line) for line in f]
    with open(tests_dir) as f:
        json_tests = [json.loads(line) for line in f]

    res_testcases = np.zeros([N_HUMAN_EVAL_EXAMPLES, num_rsa_tests, num_programs])
    res_json = []
    
    for i in tqdm(range(N_HUMAN_EVAL_EXAMPLES)):
        const_matrix = consistency_matrices[i]
        
        l0 = RSA.normalize_rows(const_matrix)
        s1 = RSA.normalize_cols(l0)

        prag_testcases = np.zeros([num_tests])

        for j in range(num_programs):
            logprobs = json_programs[i * num_programs + j]['logprobs']
            log_p_c_given_p = sum(logprobs)
            log_p_t_given_c = np.log(s1[:, j])
            prag_testcases += np.exp(log_p_c_given_p + log_p_t_given_c)
        
        prag_testcase_idxs = np.argsort(prag_testcases)[::-1]
        top_k_testcase_idxs = np.sort(prag_testcase_idxs[:num_rsa_tests])

        top_k_const_matrix = const_matrix[top_k_testcase_idxs]
        res_testcases[i] = top_k_const_matrix

        for j in range(top_k_testcase_idxs.shape[0]):
            res_json.append(json_tests[i * num_tests + top_k_testcase_idxs[j]])
    
    np.save(res_const_matrix_dir, res_testcases)
    write_jsonl(res_testcase_dir, res_json)

if __name__ == '__main__':
    fire.Fire(generate_rsa_testcases)

        















