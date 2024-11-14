import numpy as np
import json
from human_eval.data import write_jsonl
from tqdm import tqdm
import fire
import os
from ..src.codeT import CodeT
from ..src.rsa import RSA

def generate_rsa_codeT_results(num_programs, num_tests, num_input_tests, num_ransac_samples, num_out_programs, \
                           programs_dir, tests_dir, canonical_programs_dir, const_matrix_dir, res_dir, res_json_dir):
    """
    Uses CodeT algorithm with RSA to re-rank model generated programs based on dual-execution agreement.
    """
    const_matrices = np.load(const_matrix_dir)
    const_matrices = const_matrices[:, :, :-1]

    with open(programs_dir) as f:
        json_programs = [json.loads(line) for line in f]
    with open(tests_dir) as f:
        json_tests = [json.loads(line) for line in f]
    with open(canonical_programs_dir) as f:
        json_canonical_programs = [json.loads(line) for line in f]
    
    task_ids = [program['task_id'] for program in json_canonical_programs]

    # Determine if the dataset is HumanEval or MBPP (sanitized)
    n_humaneval_problems = 164
    n_mbpp_sanitize_problems = 427
    if len(task_ids) == n_humaneval_problems:
        dataset = "humaneval"
    else:
        assert(len(task_ids) == n_mbpp_sanitize_problems)
        dataset = "mbpp"

    res_programs = []
    for i, task_id in tqdm(enumerate(task_ids)):
        const_matrix = const_matrices[i]
        tests = np.array([json_test['completion'] for json_test in json_tests[i * num_tests : i * num_tests + num_tests]])
        programs = np.array([json_program['completion'] for json_program in json_programs[i *  num_programs : i * num_programs + num_programs]])
        program_prob_map = {json_program['completion'] : json_program['logprobs'] for json_program in json_programs[i *  num_programs : i * num_programs + num_programs]}

        # De-deuplicating programs and tests
        unique_programs = {}
        unique_tests = {}
        for j, program in enumerate(programs):
            unique_programs[program] = j
        for j, test in enumerate(tests):
            unique_tests[test] = j
        
        program_idxs = np.array(list(unique_programs.values()))
        test_idxs = np.array(list(unique_tests.values()))
        
        # Update the programs, tests and const_matrix
        const_matrix = const_matrix[test_idxs, :][:, program_idxs]
        programs = programs[program_idxs].reshape(-1).tolist()
        tests = tests[test_idxs].reshape(-1).tolist()

        # Inference time RSA step
        L0 = RSA.normalize_rows(const_matrix)
        S1 = RSA.normalize_cols(L0)

        prag_testcases = np.zeros([len(tests)])
        for j in range(len(programs)):
            logprobs = program_prob_map[programs[j]]
            log_p_c_given_p = sum(logprobs)
            log_p_t_given_c = np.log(S1[:, j].reshape(prag_testcases.shape))
            prag_testcases += log_p_c_given_p + log_p_t_given_c
        
        prag_testcase_idxs = np.argsort(prag_testcases)[::-1]

        to_run_codet = True
        if prag_testcase_idxs.size >= num_input_tests:
            # We have at least num_input_tests amount of tests
            prag_testcase_idxs = prag_testcase_idxs[:num_input_tests]
        elif prag_testcase_idxs.size < num_input_tests and prag_testcase_idxs.size >= 1:
            # We have enough tests for CodeT but not num_input_tests amount
            pass
        else:
            to_run_codet = False
        
        tests = np.array(tests)[prag_testcase_idxs].tolist()
        const_matrix = const_matrix[prag_testcase_idxs, :]

        if to_run_codet:
            codet = CodeT(programs, tests, num_ransac_samples, num_out_programs, const_matrix)
            for program in codet.programs:
                res_programs.append({'task_id' : task_id, 'completion' : program})
        else:
            # If no CodeT, then just output the unique generated programs
            for program in programs:
                res_programs.append({'task_id' : task_id, 'completion' : program})
    
    write_jsonl(res_dir, res_programs)
    os.system("python3 -m evalplus.evaluate --dataset %s --samples %s" % (dataset, res_dir))

    res_result_dir = res_dir[:-6] + "_eval_results" + ".json"
    with open(res_result_dir) as f:
        eval_res_json = json.load(f)
    
    task_ids = list(eval_res_json['eval'].keys())
    
    res_json = {}
    for i, task_id in enumerate(task_ids):
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
    fire.Fire(generate_rsa_codeT_results)



