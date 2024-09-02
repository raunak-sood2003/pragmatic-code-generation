import numpy as np
import json
from human_eval.data import write_jsonl
from tqdm import tqdm
import fire
from ..src.codeT import CodeT
from ..src.rsa import RSA
from ..src.utils import verify_const_matrix

def generate_rsa_codeT_canonical_results(num_programs, num_tests, num_input_tests, num_ransac_samples, num_out_programs, \
                                     programs_dir, tests_dir, const_matrix_dir, canonical_const_matrix_dir, res_dir):
    N_HUMAN_EVAL_EXAMPLES = 164
    const_matrices = np.load(const_matrix_dir)
    canonical_const_matrices = np.load(canonical_const_matrix_dir)

    with open(programs_dir) as f:
        json_programs = [json.loads(line) for line in f]
    with open(tests_dir) as f:
        json_tests = [json.loads(line) for line in f]

    res_programs = []
    for i in tqdm(range(N_HUMAN_EVAL_EXAMPLES)):
        canonical_const_matrix = canonical_const_matrices[i]
        const_matrix = const_matrices[i]
        const_matrix_concat = np.concatenate((const_matrix, canonical_const_matrix), axis = 1)
        
        tests = np.array([json_test['completion'] for json_test in json_tests[i * num_tests : i * num_tests + num_tests]])
        programs = [json_program['completion'] for json_program in json_programs[i * num_programs : i * num_programs + num_programs]]

        l0 = RSA.normalize_rows(const_matrix_concat)
        s1 = RSA.normalize_cols(l0)

        test_idxs = np.argsort(s1[:, -1])[::-1][:num_input_tests]
        tests = tests[test_idxs].tolist()
        const_matrix = const_matrix[test_idxs]

        codet = CodeT(programs, tests, num_ransac_samples, num_out_programs, const_matrix)

        for program in codet.programs:
            task_id = 'HumanEval/%d' % i
            res_programs.append({'task_id' : task_id, 'completion' : program})
    
    write_jsonl(res_dir, res_programs)

if __name__ == '__main__':
    fire.Fire(generate_rsa_codeT_canonical_results)