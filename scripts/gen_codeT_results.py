import numpy as np
import json
from human_eval.data import write_jsonl
from ..src.codeT import CodeT

if __name__ == '__main__':
    N_HUMAN_EVAL_EXAMPLES = 164
    N_PROGRAMS = 100
    N_TESTCASES = 100

    const_matrices_dir = '/home/rrsood/CodeGen/codellama_runs/const-matrices/codellama_humaneval_k100_const_matrix.npy'
    const_matrices = np.load(const_matrices_dir)

    with open('/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_programs_k%d.jsonl' % N_PROGRAMS) as f:
        json_programs = [json.loads(line) for line in f]
    with open('/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_tests_k%d.jsonl' % N_TESTCASES) as f:
        json_tests = [json.loads(line) for line in f]
    

    # Number of programs to output
    k = 10

    # Number of samples to take in RANSAC
    n = 100

    res_programs = []
    for i in range(N_HUMAN_EVAL_EXAMPLES):
        const_matrix = const_matrices[i]
        rsa_tests = [json_test['completion'] for json_test in json_tests[i * N_TESTCASES : i * N_TESTCASES + N_TESTCASES]]
        programs = [json_program['completion'] for json_program in json_programs[i *  N_PROGRAMS : i * N_PROGRAMS + N_PROGRAMS]]

        assert(len(rsa_tests) == N_TESTCASES)
        assert(len(programs) == N_PROGRAMS)

        print("HumanEval/%d" % i)
        codet = CodeT(programs, rsa_tests, n, k, const_matrix)

        for program in codet.programs:
            task_id = 'HumanEval/%d' % i
            res_programs.append({'task_id' : task_id, 'completion' : program})
    
    write_jsonl('/home/rrsood/CodeGen/codellama_runs/codeT-results/codellama_humaneval_codet_k%d.jsonl' % k, res_programs)
