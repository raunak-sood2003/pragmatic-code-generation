import json
import numpy as np
import fire
from tqdm import tqdm
from ..src.utils import valid_program_testcase_pair

'''
Generates consistency matrix (numpy matrix) given JSON files of programs and test cases 
generated from HumanEval/MBPP prompts. JSON files should have generated code in a "completion" entry.
'''
def generate_const_matrix(dataset, num_programs, num_tests, programs_dir, tests_dir, save_dir):
    with open(programs_dir) as f:
        json_programs = [json.loads(program) for program in f]
    with open(tests_dir) as f:
        json_tests = [json.loads(test) for test in f]

    if dataset == 'humaneval':
        N_PROBLEMS = 164
    elif dataset == 'mbpp':
        N_PROBLEMS = 1000
    else:
        print("Invalid dataset")
        assert(False)
    
    const_matrix = np.zeros([N_PROBLEMS, num_tests, num_programs])

    for l in tqdm(range(0, N_PROBLEMS)):
        programs = json_programs[l * num_programs : l * num_programs + num_programs]
        tests = json_tests[l * num_tests : l * num_tests + num_tests]
            
        sample_programs = [programs[j]["completion"] for j in range(num_programs)]
        sample_tests = [tests[j]["completion"] for j in range(num_tests)]
        
        for i in range(len(sample_tests)):
            for j in range(len(sample_programs)):
                if valid_program_testcase_pair(sample_programs[j], sample_tests[i]):
                    const_matrix[l][i][j] = 1

    np.save(save_dir, const_matrix)

if __name__ == '__main__':
    fire.Fire(generate_const_matrix)

    
