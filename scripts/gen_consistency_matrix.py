import json
import numpy as np
from ..src.utils import valid_program_testcase_pair

if __name__ == '__main__':
    n_progs, n_tests = 100, 100
    programs_dir = '/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_programs_k100.jsonl'
    tests_dir = '/home/rrsood/CodeGen/codellama_runs/generations/codellama_humaneval_tests_k100.jsonl'

    with open(programs_dir) as f:
        json_programs = [json.loads(program) for program in f]

    with open(tests_dir) as f:
        json_tests = [json.loads(test) for test in f]

    const_matrix = np.zeros([164, n_tests, n_progs])

    for l in range(0, 164):
        programs = json_programs[l * n_progs:l*n_progs+n_progs]
        tests = json_tests[l*n_tests:l*n_tests+n_tests]
            
        sample_programs = [programs[j]["completion"] for j in range(n_progs)]
        sample_tests = [tests[j]["completion"] for j in range(n_tests)]

        print("Processing HumanEval/%d" % l)
        
        for i in range(len(sample_tests)):
            for j in range(len(sample_programs)):
                if valid_program_testcase_pair(sample_programs[j], sample_tests[i]):
                    const_matrix[l][i][j] = 1

    np.save('/home/rrsood/CodeGen/const-matrices/codellama_humaneval_k100_const_matrix.npy', const_matrix)

    
