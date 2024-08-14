import numpy as np
from ..src.rsa import RSA
import json
from human_eval.data import write_jsonl

if __name__ == '__main__':
    N_HUMAN_EVAL_EXAMPLES = 164
    N_PROGRAMS = 100
    N_TESTCASES = 100
    
    consistency_matrices_dir = '/home/rrsood/CodeGen/codellama_runs/const-matrices/codellama_humaneval_k100_const_matrix.npy'
    consistency_matrices = np.load(consistency_matrices_dir)

    assert(consistency_matrices.shape[0] == N_HUMAN_EVAL_EXAMPLES)
    assert(consistency_matrices.shape[1] == N_TESTCASES)
    assert(consistency_matrices.shape[2] == N_PROGRAMS)

    with open('/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_programs_k100.jsonl') as f:
        json_programs = [json.loads(line) for line in f]
    with open('/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_tests_k100.jsonl') as f:
        json_tests = [json.loads(line) for line in f]
    
    # How many test cases you want RSA to select
    k = 100

    res_testcases = np.zeros([N_HUMAN_EVAL_EXAMPLES, k, N_PROGRAMS])
    res_json = []
    
    for i in range(N_HUMAN_EVAL_EXAMPLES):
        const_matrix = consistency_matrices[i]
        
        l0 = RSA.normalize_rows(const_matrix)
        s1 = RSA.normalize_cols(l0)

        prag_testcases = np.zeros([N_TESTCASES])

        for j in range(N_PROGRAMS):
            logprobs = json_programs[i * N_PROGRAMS + j]['logprobs']
            log_p_c_given_p = sum(logprobs)
            log_p_t_given_c = np.log(s1[:, j])
            prag_testcases += np.exp(log_p_c_given_p + log_p_t_given_c)
        
        prag_testcase_idxs = np.argsort(prag_testcases)[::-1]
        top_k_testcase_idxs = np.sort(prag_testcase_idxs[:k])

        top_k_const_matrix = const_matrix[top_k_testcase_idxs]
        res_testcases[i] = top_k_const_matrix

        for j in range(top_k_testcase_idxs.shape[0]):
            res_json.append(json_tests[i * N_TESTCASES + top_k_testcase_idxs[j]])
    
    np.save('/home/rrsood/CodeGen/codellama_runs/rsa-testcases/codellama_humaneval_rsa_%dprograms_%dtestcases_const_matrix_k%d.npy' % (N_PROGRAMS, N_TESTCASES, k), res_testcases)
    write_jsonl('/home/rrsood/CodeGen/codellama_runs/rsa-testcases/codellama_humaneval_rsa_%dprograms_%dtestcases_k%d.jsonl' % (N_PROGRAMS, N_TESTCASES, k), res_json)

        















