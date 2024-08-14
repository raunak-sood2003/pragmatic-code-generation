import json
from human_eval.data import write_jsonl
import numpy as np
from ..src.utils import verify_const_matrix

if __name__ == '__main__':
    N_HUMANEVAL_EXAMPLES = 164
    n_programs, n_tests = 100, 10

    programs_dir = "/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_programs_k100.jsonl"
    tests_dir = "/home/rrsood/CodeGen/codellama_runs/rsa-testcases/codellama_humaneval_rsa_100programs_100testcases_k10.jsonl"
    const_matrix_dir = "/home/rrsood/CodeGen/codellama_runs/rsa-testcases/codellama_humaneval_rsa_100programs_100testcases_const_matrix_k10.npy"

    with open(programs_dir) as f:
        json_programs = [json.loads(line) for line in f]
    with open(tests_dir) as f:
        json_tests = [json.loads(line) for line in f]

    const_matrices = np.load(const_matrix_dir)

    results = []
    for i in range(N_HUMANEVAL_EXAMPLES):
        const_matrix = const_matrices[i]
        tests = [json_test['completion'] for json_test in json_tests[i * n_tests : i * n_tests + n_tests]]
        programs = [json_program['completion'] for json_program in json_programs[i *  n_programs : i * n_programs + n_programs]]
        if verify_const_matrix(programs, tests, const_matrix):
            results.append(True)
        else:
            results.append(False)

    print("Results:")
    print(results)


