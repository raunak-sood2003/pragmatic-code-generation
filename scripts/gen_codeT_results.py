import numpy as np
import json
from human_eval.data import write_jsonl
from tqdm import tqdm
import fire
from ..src.codeT import CodeT

def generate_codeT_results(num_programs, num_tests, num_out_programs, num_ransac_samples, programs_dir, tests_dir, const_matrix_dir, res_dir):
    N_HUMAN_EVAL_EXAMPLES = 164
    const_matrices = np.load(const_matrix_dir)

    with open(programs_dir) as f:
        json_programs = [json.loads(line) for line in f]
    with open(tests_dir) as f:
        json_tests = [json.loads(line) for line in f]

    res_programs = []
    for i in tqdm(range(N_HUMAN_EVAL_EXAMPLES)):
        const_matrix = const_matrices[i]
        rsa_tests = [json_test['completion'] for json_test in json_tests[i * num_tests : i * num_tests + num_tests]]
        programs = [json_program['completion'] for json_program in json_programs[i *  num_programs : i * num_programs + num_programs]]

        assert(len(rsa_tests) == num_tests)
        assert(len(programs) == num_programs)

        print("HumanEval/%d" % i)
        codet = CodeT(programs, rsa_tests, num_ransac_samples, num_out_programs, const_matrix)

        for program in codet.programs:
            task_id = 'HumanEval/%d' % i
            res_programs.append({'task_id' : task_id, 'completion' : program})
    
    write_jsonl(res_dir, res_programs)


if __name__ == '__main__':
    fire.Fire(generate_codeT_results)
