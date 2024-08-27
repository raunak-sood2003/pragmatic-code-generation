import json
import numpy as np
import fire
from tqdm import tqdm
from ..src.utils import valid_program_testcase_pair

'''
Generates consistency matrix (numpy matrix) with model generated tests and HumanEval ground truth programs. 
Consistency matrix will be of shape (164, num_tests, 1).
'''
def generate_canonical_const_matrix(num_tests, programs_dir, tests_dir, save_dir):
    with open(programs_dir) as f:
        json_programs = [json.loads(program) for program in f]

    with open(tests_dir) as f:
        json_tests = [json.loads(test) for test in f]

    N_PROBLEMS = 164
    const_matrix = np.zeros([N_PROBLEMS, num_tests, 1])

    for l in tqdm(range(0, N_PROBLEMS)):
        tests = json_tests[l * num_tests : l * num_tests + num_tests]
            
        sample_program = json_programs[l]["completion"]
        sample_tests = [tests[j]["completion"] for j in range(num_tests)]

        print("Processing HumanEval/%d" % l)
        
        for i in range(len(sample_tests)):
            if valid_program_testcase_pair(sample_program, sample_tests[i]):
                const_matrix[l][i][0] = 1

    np.save(save_dir, const_matrix)

if __name__ == '__main__':
    fire.Fire(generate_canonical_const_matrix)