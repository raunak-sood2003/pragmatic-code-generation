import json
import numpy as np
import fire
from tqdm import tqdm
from ..src.utils import valid_program_testcase_pair

'''
Generates consistency matrix (numpy matrix) with model generated tests and HumanEval/MBPP ground truth programs. 
Consistency matrix will be of shape (164, num_tests, 1) for HumanEval and (1000, num_tests, 1) for MBPP.
'''
def generate_canonical_const_matrix(dataset, num_tests, canonical_programs_dir, generated_tests_dir, save_dir):
    with open(canonical_programs_dir) as f:
        json_programs = [json.loads(program) for program in f]

    with open(generated_tests_dir) as f:
        json_tests = [json.loads(test) for test in f]

    if dataset == 'humaneval':
        N_PROBLEMS = 164
    elif dataset == 'mbpp':
        N_PROBLEMS = 1000
    else:
        print("Invalid dataset")
        assert(False)
    
    const_matrix = np.zeros([N_PROBLEMS, num_tests, 1])

    for l in tqdm(range(0, N_PROBLEMS)):
        tests = json_tests[l * num_tests : l * num_tests + num_tests]
            
        sample_program = json_programs[l]["completion"]
        sample_tests = [tests[j]["completion"] for j in range(num_tests)]
        
        for i in range(len(sample_tests)):
            if valid_program_testcase_pair(sample_program, sample_tests[i]):
                const_matrix[l][i][0] = 1

    np.save(save_dir, const_matrix)

if __name__ == '__main__':
    fire.Fire(generate_canonical_const_matrix)