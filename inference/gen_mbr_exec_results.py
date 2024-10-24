import numpy as np
import json
from human_eval.data import write_jsonl
from tqdm import tqdm
import fire
import os
from ..src.mbr_exec import MBRExec

def generate_mbr_exec_results(num_programs, num_tests, programs_dir, tests_dir, \
                              res_dir, res_json_dir, value_matrix_dir = None):
    """
    Uses MBRExec algorithm to re-rank model generated programs based on test execution results.
    """
    N_HUMAN_EVAL_EXAMPLES = 164
    if value_matrix_dir is not None:
        value_matrices = np.load(value_matrix_dir)
    else:
        value_matrices = None

    with open(programs_dir) as f:
        json_programs = [json.loads(line) for line in f]
    with open(tests_dir) as f:
        json_tests = [json.loads(line) for line in f]

    res_programs = []
    for i in tqdm(range(N_HUMAN_EVAL_EXAMPLES)):
        task_id = 'HumanEval/%d' % i
        tests = np.array([json_test['completion'] for json_test in json_tests[i * num_tests : i * num_tests + num_tests]])
        programs = np.array([json_program['completion'] for json_program in json_programs[i *  num_programs : i * num_programs + num_programs]])

        # De-deuplicating programs and tests
        unique_programs = {}
        unique_tests = {}
        for j, program in enumerate(programs):
            unique_programs[program] = j
        for j, test in enumerate(tests):
            unique_tests[test] = j
        
        program_idxs = np.array(list(unique_programs.values()))
        test_idxs = np.array(list(unique_tests.values()))
        
        
        # Update the programs, tests and value_matrix
        if value_matrices is not None:
            value_matrix = value_matrices[i][test_idxs, :][:, program_idxs]
        else:
            value_matrix = None
     
        programs = programs[program_idxs].reshape(-1).tolist()
        tests = tests[test_idxs].reshape(-1).tolist()

        mbr_exec = MBRExec(programs, tests, value_matrix)
        for program in mbr_exec.reranked_programs:
            res_programs.append({'task_id' : task_id, 'completion' : program})
        
    
    write_jsonl(res_dir, res_programs)
    os.system("python3 -m evalplus.evaluate --dataset humaneval --samples %s" % (res_dir))

    res_result_dir = res_dir[:-6] + "_eval_results" + ".json"
    with open(res_result_dir) as f:
        eval_res_json = json.load(f)
    
    res_json = {}
    for i in range(N_HUMAN_EVAL_EXAMPLES):
        task_id = 'HumanEval/%d' % i
        res_program_json = eval_res_json["eval"][task_id]
        base_res = []
        plus_res = []

        for j in range(len(res_program_json)):
            if res_program_json[j]["base_status"] == "pass":
                base_res.append(1)
            else:
                base_res.append(0)
            
            if res_program_json[j]["plus_status"] == "pass":
                plus_res.append(1)
            else:
                plus_res.append(0)
        
        res_json[task_id] = {'base_results' : base_res, 'plus_results' : plus_res}

    with open(res_json_dir, 'w') as f:
        json.dump(res_json, f)


if __name__ == '__main__':
    fire.Fire(generate_mbr_exec_results)
