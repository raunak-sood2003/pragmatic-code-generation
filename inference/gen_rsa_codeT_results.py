import numpy as np
from human_eval.data import write_jsonl
import json
import fire
from tqdm import tqdm
from ..src.codeT import CodeT

def generate_rsa_codeT_results(num_programs, num_rsa_tests, num_out_programs, num_ransac_samples, programs_dir, rsa_tests_dir, rsa_const_matrix_dir, res_dir):
    """
    Selects test cases based on RSA inference time algorithm and runs CodeT to re-rank programs with the test cases.
    """
    N_HUMAN_EVAL_EXAMPLES = 164
    rsa_const_matrices = np.load(rsa_const_matrix_dir)

    N_EXAMPLES, N_TESTS, N_PROGRAMS = rsa_const_matrices.shape[0], rsa_const_matrices.shape[1], rsa_const_matrices.shape[2]
    assert(num_programs == N_PROGRAMS)
    assert(num_rsa_tests == N_TESTS)
    assert(N_EXAMPLES == N_HUMAN_EVAL_EXAMPLES)

    with open(programs_dir) as f:
        json_programs = [json.loads(line) for line in f]
    with open(rsa_tests_dir) as f:
        json_tests = [json.loads(line) for line in f]

    res_programs = []
    
    for i in tqdm(range(N_HUMAN_EVAL_EXAMPLES)):
        rsa_const_matrix = rsa_const_matrices[i]
        rsa_tests = [json_test['completion'] for json_test in json_tests[i * num_rsa_tests : i * num_rsa_tests + num_rsa_tests]]
        programs = [json_program['completion'] for json_program in json_programs[i *  num_programs : i * num_programs + num_programs]]

        assert(len(rsa_tests) == num_rsa_tests)
        assert(len(programs) == num_programs)

        codet = CodeT(programs, rsa_tests, num_ransac_samples, num_out_programs, rsa_const_matrix)

        for program in codet.programs:
            task_id = 'HumanEval/%d' % i
            res_programs.append({'task_id' : task_id, 'completion' : program})
    
    write_jsonl(res_dir, res_programs)

if __name__ == '__main__':
    fire.Fire(generate_rsa_codeT_results)



