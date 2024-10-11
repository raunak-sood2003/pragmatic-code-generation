import numpy as np
import json
from human_eval.data import write_jsonl
from tqdm import tqdm
import fire
from ..src.codeT import CodeT
from ..src.rsa import RSA
from ..src.utils import verify_const_matrix
import os

def generate_rsa_codeT_canonical_results(num_programs, num_tests, num_input_tests, num_ransac_samples, num_out_programs, programs_dir, \
                                         canonical_programs_dir, tests_dir, const_matrix_dir, res_programs_dir, res_json_dir):
    """
    Runs CodeT using the model generated programs and informatively chosen test cases (need to precompute consistency matrix first). 
    Saves results in json file consisting of evalplus results and CodeT clusters.
    """
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
        canonical_const_matrix = canonical_const_matrices[i]
        const_matrix = const_matrices[i]
        json_res = {}
        
        tests = [json_test['completion'] for json_test in json_tests[i * num_tests : i * num_tests + num_tests]]
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
        canonical_const_matrix = canonical_const_matrix[tests_idx, :]
        const_matrix_concat = np.concatenate((const_matrix, canonical_const_matrix), axis = 1)

        json_res['gen_programs'] = programs
        json_res['canonical_program'] = canonical_program

        # RSA procedure: getting pragmatic speaker distribution
        l0 = RSA.normalize_rows(const_matrix_concat)
        s1 = RSA.normalize_cols(l0)

        # Selecting ony consistent test cases in speaker distribution
        consistent_idxs = set(list(np.argwhere(canonical_const_matrix.reshape(-1,) == 1).reshape(-1,)))
        rsa_idxs = list(np.argsort(s1[:, -1])[::-1])
        rsa_test_idxs = np.array([rsa_idxs[j] for j in range(len(rsa_idxs)) if rsa_idxs[j] in consistent_idxs])

        to_run_codet = True
        if rsa_test_idxs.size >= num_input_tests:
            # We have at least num_input_tests amount of tests
            rsa_test_idxs = rsa_test_idxs[:num_input_tests]
            tests = tests[rsa_test_idxs]
            const_matrix = const_matrix[rsa_test_idxs]
        elif rsa_test_idxs.size < num_input_tests and rsa_test_idxs.size >= 1:
            # We have enough tests for CodeT but not num_input_tests amount
            tests = tests[rsa_test_idxs]
            const_matrix = const_matrix[rsa_test_idxs]
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
                res_programs.append({'task_id' : task_id, 'completion' : program, "n_tests": len(tests)})
            json_res['clusters'] = codet.save_clusters
            json_res["reranked_programs"] = codet.programs

        else:
            # If no CodeT, then just output the unique generated programs
            for program in programs:
                res_programs.append({'task_id' : task_id, 'completion' : program})
        
        res_json[task_id] = json_res

    # Save the re-ranke programs and evaluate
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
    fire.Fire(generate_rsa_codeT_canonical_results)