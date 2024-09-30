import json
import numpy as np
import fire
from tqdm import tqdm
from ..src.utils import valid_program_testcase_pair

'''
Generates consistency matrix (numpy matrix) given JSON files of programs, test cases and canonical programs
generated from HumanEval/MBPP prompts. JSON files should have generated code in a "completion" entry.
'''
def generate_const_matrix(num_programs, num_tests, programs_dir, tests_dir, canonical_programs_dir, save_dir):
    with open(programs_dir) as f:
        json_programs = [json.loads(program) for program in f]
    with open(tests_dir) as f:
        json_tests = [json.loads(test) for test in f]
    with open(canonical_programs_dir) as f:
        json_canonical_programs = [json.loads(program) for program in f]

    N_PROBLEMS = len(json_canonical_programs)
    const_matrix = np.zeros([N_PROBLEMS, num_tests, num_programs + 1])

    for l in range(N_PROBLEMS):
        print("Processing task_id %d" % l)
        programs = json_programs[l * num_programs : l * num_programs + num_programs]
        tests = json_tests[l * num_tests : l * num_tests + num_tests]
            
        sample_programs = [programs[j]["completion"] for j in range(num_programs)]
        sample_tests = [tests[j]["completion"] for j in range(num_tests)]
        
        canonical_program = json_canonical_programs[l]['completion']
        sample_programs.append(canonical_program)
        
        for i in range(len(sample_tests)):
            for j in range(len(sample_programs)):
                if valid_program_testcase_pair(sample_programs[j], sample_tests[i]):
                    const_matrix[l][i][j] = 1

    np.save(save_dir, const_matrix)

if __name__ == '__main__':
    fire.Fire(generate_const_matrix)

    
