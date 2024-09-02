import json
from human_eval.data import write_jsonl
import numpy as np
from tqdm import tqdm
import fire
from ..src.utils import verify_const_matrix

'''
Verifies whether the consistency matrix between programs and test cases is correct for all HumanEval problems. 
Programs and tests JSON files must be in HumanEval submission format. Consistency matrix must be of shape (164, n_tests, n_programs).
'''
def verify_consistency_matrix(num_programs, num_tests, programs_dir, tests_dir, const_matrix_dir):
    with open(programs_dir) as f:
        json_programs = [json.loads(line) for line in f]
    with open(tests_dir) as f:
        json_tests = [json.loads(line) for line in f]

    const_matrices = np.load(const_matrix_dir)
    N_HUMANEVAL_EXAMPLES = 164

    results = []
    for i in tqdm(range(N_HUMANEVAL_EXAMPLES)):
        const_matrix = const_matrices[i]
        tests = [json_test['completion'] for json_test in json_tests[i * num_tests : i * num_tests + num_tests]]
        programs = [json_program['completion'] for json_program in json_programs[i *  num_programs : i * num_programs + num_programs]]
        if verify_const_matrix(programs, tests, const_matrix):
            results.append(True)
            print("HumanEval/%d PASS" % i)
        else:
            results.append(False)
            print("HumanEval/%d FAIL" % i)

    print("Results:")
    print(results)


if __name__ == '__main__':
    fire.Fire(verify_consistency_matrix)

