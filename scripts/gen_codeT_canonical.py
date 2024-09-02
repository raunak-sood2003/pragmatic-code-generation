import numpy as np
import json
from human_eval.data import write_jsonl
from tqdm import tqdm
import fire
from ..src.codeT import CodeT
from ..src.utils import verify_const_matrix

def generate_codeT_canonical_results(num_programs, num_tests, num_input_tests, num_ransac_samples, num_out_programs, \
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
        canonical_const_matrix = canonical_const_matrices[i].reshape(-1,)
        canonical_const_idxs = np.argwhere(canonical_const_matrix == 1).reshape(-1,)
        tests = np.array([json_test['completion'] for json_test in json_tests[i * num_tests : i * num_tests + num_tests]])
        programs = [json_program['completion'] for json_program in json_programs[i * num_programs : i * num_programs + num_programs]]
        
        if canonical_const_idxs.size >= num_input_tests:
            canonical_const_idxs = np.random.permutation(canonical_const_idxs)[:num_input_tests]
            const_matrix = const_matrices[i][canonical_const_idxs].reshape(num_input_tests, num_programs)
            tests = tests[canonical_const_idxs].reshape(-1).tolist()
        else:
            canonical_const_idxs = np.random.permutation(np.arange(0, num_tests))[:num_input_tests]
            const_matrix = const_matrices[i][canonical_const_idxs].reshape(num_input_tests, num_programs)
            tests = tests[canonical_const_idxs].reshape(-1).tolist()

        codet = CodeT(programs, tests, num_ransac_samples, num_out_programs, const_matrix)

        for program in codet.programs:
            task_id = 'HumanEval/%d' % i
            res_programs.append({'task_id' : task_id, 'completion' : program})
    
    write_jsonl(res_dir, res_programs)

if __name__ == '__main__':
    fire.Fire(generate_codeT_canonical_results)