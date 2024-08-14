import numpy as np
from human_eval.data import write_jsonl
import json
from ..src.codeT import CodeT

if __name__ == '__main__':
    N_HUMAN_EVAL_EXAMPLES = 164
    N_PRAGMATIC_TESTS = 100
        
    rsa_const_matrices_dir = '/home/rrsood/CodeGen/codellama_runs/rsa-testcases/codellama_humaneval_rsa_100programs_100testcases_const_matrix_k%d.npy' % N_PRAGMATIC_TESTS
    rsa_const_matrices = np.load(rsa_const_matrices_dir)

    n_examples, n_testcases, n_programs = rsa_const_matrices.shape[0], rsa_const_matrices.shape[1], rsa_const_matrices.shape[2]
    assert(n_testcases == N_PRAGMATIC_TESTS)
    assert(n_examples == N_HUMAN_EVAL_EXAMPLES)

    # Number of programs to output
    k = 100

    # Number of samples to take in RANSAC
    n = 100

    with open('/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_programs_k100.jsonl') as f:
        json_programs = [json.loads(line) for line in f]
    with open('/home/rrsood/CodeGen/codellama_runs/rsa-testcases/codellama_humaneval_rsa_100programs_100testcases_k%d.jsonl' % n_testcases) as f:
        json_tests = [json.loads(line) for line in f]

    res_programs = []
    
    for i in range(N_HUMAN_EVAL_EXAMPLES):
        rsa_const_matrix = rsa_const_matrices[i]
        rsa_tests = [json_test['completion'] for json_test in json_tests[i * n_testcases : i * n_testcases + n_testcases]]
        programs = [json_program['completion'] for json_program in json_programs[i *  n_programs : i * n_programs + n_programs]]

        assert(len(rsa_tests) == n_testcases)
        assert(len(programs) == n_programs)

        codet = CodeT(programs, rsa_tests, n, k, rsa_const_matrix)

        for program in codet.programs:
            task_id = 'HumanEval/%d' % i
            res_programs.append({'task_id' : task_id, 'completion' : program})
    
    write_jsonl('/home/rrsood/CodeGen/codellama_runs/rsa-codet-results/codellama_humaneval_codet_rsa_k%d.jsonl' % n_testcases, res_programs)



