import numpy as np
import json
from human_eval.data import write_jsonl
from tqdm import tqdm
import fire
from ..src.codeT import CodeT
import os

"""
Runs CodeT using the model generated programs and test cases (need to precompute consistency matrix first). 
Saves results in json file consisting of evalplus results and CodeT clusters.
"""
def generate_codeT_canonical_results(num_programs, num_tests, num_input_tests, num_ransac_samples, num_out_programs, \
                                     programs_dir, canonical_programs_dir, tests_dir, const_matrix_dir, res_programs_dir, res_json_dir):
    
    combined_const_matrices = np.load(const_matrix_dir)
    const_matrices = combined_const_matrices[:, :, :-1]
    canonical_const_matrices = combined_const_matrices[:, :, -1, None]

    with open(programs_dir) as f:
        json_programs = [json.loads(line) for line in f]
    with open(canonical_programs_dir) as f:
        json_canonical_programs = [json.loads(line) for line in f]
    with open(tests_dir) as f:
        json_tests = [json.loads(line) for line in f]
    
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
    res_json = {}
    for i, task_id in enumerate(task_ids):
        print("Processing task_id %d" % i)
        json_res = {}
        const_matrix = const_matrices[i]
        canonical_const_matrix = canonical_const_matrices[i]

        tests = np.array([json_test['completion'] for json_test in json_tests[i * num_tests : i * num_tests + num_tests]])
        programs = [json_program['completion'] for json_program in json_programs[i * num_programs : i * num_programs + num_programs]]
        canonical_program = json_canonical_programs[i]['completion']

        # De-deuplicating programs and tests
        unique_programs = {}
        unique_tests = {}
        for j, program in enumerate(programs):
            unique_programs[program] = j
        for j, test in enumerate(tests):
            unique_tests[test] = j
        
        programs = list(unique_programs.keys())
        programs_idx = np.array(list(unique_programs.values()))
        tests = np.array(list(unique_tests.keys()))
        tests_idx = np.array(list(unique_tests.values()))
        
        # New const matrices with unique programs and tests
        const_matrix = const_matrix[tests_idx, :][:, programs_idx]
        canonical_const_matrix = canonical_const_matrix[tests_idx, :].reshape(-1,)
        canonical_const_idxs = np.argwhere(canonical_const_matrix == 1).reshape(-1,)

        json_res['gen_programs'] = programs
        json_res['canonical_program'] = canonical_program
        
        to_run_codet = True
        if canonical_const_idxs.size >= num_input_tests:
            # We have at least num_input_tests amount of tests
            canonical_const_idxs = np.random.permutation(canonical_const_idxs)[:num_input_tests]
            const_matrix = const_matrix[canonical_const_idxs].reshape(num_input_tests, len(programs))
            tests = tests[canonical_const_idxs].reshape(-1)
        elif canonical_const_idxs.size < num_input_tests and canonical_const_idxs.size >= 1:
            # We have enough tests for CodeT but not num_input_tests amount
            canonical_const_idxs = np.random.permutation(canonical_const_idxs)
            const_matrix = const_matrix[canonical_const_idxs].reshape(canonical_const_idxs.size, len(programs))
            tests = tests[canonical_const_idxs].reshape(-1)
        else:
            # We have no tests -> can't run CodeT
            to_run_codet = False
        
        tests = tests.tolist()
        
        # Saving data to json
        json_res['codeT_tests'] = tests
        json_res['ran_codeT'] = to_run_codet
        json_res['clusters'] = []
        json_res["reranked_programs"] = programs

        if to_run_codet:
            # Running CodeT and saving re-ranked programs
            codet = CodeT(programs, tests, num_ransac_samples, num_out_programs, const_matrix)
            for program in codet.programs:
                res_programs.append({'task_id' : task_id, 'completion' : program})
            json_res['clusters'] = codet.save_clusters
            json_res["reranked_programs"] = codet.programs
        else:
            # If no CodeT, then just output the unique generated programs
            for program in programs:
                res_programs.append({'task_id' : task_id, 'completion' : program})
        
        res_json[task_id] = json_res
    
    # Save the re-ranked programs and evaluate
    write_jsonl(res_programs_dir, res_programs)
    os.system("export PYTHONPATH=$PYTHONPATH:/home/rrsood/CodeGen/evalplus")
    os.system("python3 -m evalplus.evaluate --dataset %s --samples %s" % (dataset, res_programs_dir))

    # Save eval results to json
    res_programs_result_dir = res_programs_dir[:-6] + "_eval_results" + ".json"
    with open(res_programs_result_dir) as f:
        eval_res_json = json.load(f)
    
    task_ids = list(eval_res_json['eval'].keys())

    for i, task_id in enumerate(task_ids):
        print("Processing task_id %d" % i)
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
        
        res_json[task_id]["base_results"] = base_res
        res_json[task_id]["plus_results"] = plus_res


    with open(res_json_dir, 'w') as f:
        json.dump(res_json, f)

if __name__ == '__main__':
    fire.Fire(generate_codeT_canonical_results)