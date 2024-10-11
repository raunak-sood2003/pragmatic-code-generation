import json
import matplotlib.pyplot as plt

def pass_at_k(res_json_dir, ids_to_exclude):
    """
    Calculates exact pass@k score for JSON result file from CodeT results.
    """
    with open(res_json_dir) as f:
        res_json = json.load(f)
    task_ids = list(res_json.keys())
    base_res = [0 for _ in range(100)]
    plus_res = [0 for _ in range(100)]
    num_eval = 0
    
    for i, task_id in enumerate(task_ids):
        if 'base_results' not in res_json[task_id]:
            continue
        if task_id in ids_to_exclude:
            continue
        
        num_eval += 1
        
        base_prob_res = res_json[task_id]['base_results']
        plus_prob_res = res_json[task_id]['plus_results']

        for i in range(1, len(base_prob_res)):
            if base_prob_res[i-1]:
                base_prob_res[i] = 1
            if plus_prob_res[i-1]:
                plus_prob_res[i] = 1
        
        for i in range(len(base_prob_res)):
            base_res[i] += base_prob_res[i]
            plus_res[i] += plus_prob_res[i]

    base_res  = [x / num_eval for x in base_res]
    plus_res = [x / num_eval for x in plus_res]

    return base_res, plus_res

def generate_icl_plot_data(passk_values):
    """
    Takes in pass@k info (from pass_at_k) for each ICL experiment, and generates plot data.
    """
    res = {}
    icl_res_dir = '/data/user_data/rrsood/icl_results/llama3-70b/'
    prompt_types_exclude = {
        '1shot' : 'HumanEval/18', 
        '3shot' : ['HumanEval/18', 'HumanEval/12', 'HumanEval/127'], 
        '5shot' : ['HumanEval/18', 'HumanEval/12', 'HumanEval/127', 'HumanEval/140', 'HumanEval/52'], 
        '10shot' : ['HumanEval/18', 'HumanEval/12', 'HumanEval/127', 'HumanEval/140', 'HumanEval/52', 'HumanEval/54', 'HumanEval/71', 'HumanEval/67', 'HumanEval/144', 'HumanEval/158']
    }

    num_input_tests = [1, 2, 3, 4, 5, 10, 25, 50, 75, 100]
    input_test_idx_map = {key : i for i, key in enumerate(num_input_tests)}

    for prompt_type in prompt_types_exclude:
        prompt_dir = icl_res_dir + prompt_type
        exclude_list = prompt_types_exclude[prompt_type]
    
        passk_scores_random_base = {k : [0 for _ in range(len(num_input_tests))] for k in passk_values}
        passk_scores_random_plus = {k : [0 for _ in range(len(num_input_tests))] for k in passk_values}
        passk_scores_informative_base = {k : [0 for _ in range(len(num_input_tests))] for k in passk_values}
        passk_scores_informative_plus = {k : [0 for _ in range(len(num_input_tests))] for k in passk_values}

        for test in num_input_tests:
            res_json_random_dir1 = '%s/random/%dtest/llama3_70b_humaneval_codet_random_icl_result_dump_%dtest_run1.json' % (prompt_dir, test, test)
            res_json_random_dir2 = '%s/random/%dtest/llama3_70b_humaneval_codet_random_icl_result_dump_%dtest_run2.json' % (prompt_dir, test, test)
            res_json_random_dir3 = '%s/random/%dtest/llama3_70b_humaneval_codet_random_icl_result_dump_%dtest_run3.json' % (prompt_dir, test, test)

            res_json_informative_dir1 = '%s/informative/%dtest/llama3_70b_humaneval_codet_informative_icl_result_dump_%dtest_run1.json' % (prompt_dir, test, test)
            res_json_informative_dir2 = '%s/informative/%dtest/llama3_70b_humaneval_codet_informative_icl_result_dump_%dtest_run2.json' % (prompt_dir, test, test)
            res_json_informative_dir3 = '%s/informative/%dtest/llama3_70b_humaneval_codet_informative_icl_result_dump_%dtest_run3.json' % (prompt_dir, test, test)
        
            
            base1_random, plus1_random = pass_at_k(res_json_random_dir1, exclude_list)
            base2_random, plus2_random = pass_at_k(res_json_random_dir2, exclude_list)
            base3_random, plus3_random = pass_at_k(res_json_random_dir3, exclude_list)
            base_random = [(base1_random[i] + base2_random[i] + base3_random[i]) / 3 for i in range(len(base1_random))]
            plus_random = [(plus1_random[i] + plus2_random[i] + plus3_random[i]) / 3 for i in range(len(plus1_random))]

            base1_informative, plus1_informative = pass_at_k(res_json_informative_dir1, exclude_list)
            base2_informative, plus2_informative = pass_at_k(res_json_informative_dir2, exclude_list)
            base3_informative, plus3_informative = pass_at_k(res_json_informative_dir3, exclude_list)
            base_informative = [(base1_informative[i] + base2_informative[i] + base3_informative[i]) / 3 for i in range(len(base1_informative))]
            plus_informative = [(plus1_informative[i] + plus2_informative[i] + plus3_informative[i]) / 3 for i in range(len(plus1_informative))]

            for k in passk_values:
                passk_scores_random_base[k][input_test_idx_map[test]] = base_random[k - 1]
                passk_scores_random_plus[k][input_test_idx_map[test]] = plus_random[k - 1]

                passk_scores_informative_base[k][input_test_idx_map[test]] = base_informative[k - 1]
                passk_scores_informative_plus[k][input_test_idx_map[test]] = plus_informative[k - 1]
            
        
        res[prompt_type] = {
            'random_base' : passk_scores_random_base,
            'random_plus' : passk_scores_random_plus,
            'informative_base' : passk_scores_informative_base,
            'informative_plus' : passk_scores_informative_plus
        }
    
    return res

def plot_results(passk_results, passk_values, num_input_tests):
    """
    Plots pass@k data from pass_at_k and generate_icl_plot_data.
    """ 
    ver_idx_map = {k : i for i, k in enumerate(passk_values)}
    hor_idx_map = {prompt_type : i for i, prompt_type in enumerate(passk_results.keys())}
    
    fig, ax = plt.subplots(len(hor_idx_map), len(ver_idx_map), figsize = (15, 15))
    fig.suptitle("ICL + CodeT Pass@k Scores on HumanEval").set_y(0.93)

    for prompt_type in passk_results:
        passk_scores_random_base = passk_results[prompt_type]['random_base']
        passk_scores_random_plus = passk_results[prompt_type]['random_plus']
        passk_scores_informative_base = passk_results[prompt_type]['informative_base']
        passk_scores_informative_plus = passk_results[prompt_type]['informative_plus']

        for k in passk_values:
            random_base = passk_scores_random_base[k]
            random_plus = passk_scores_random_plus[k]
            informative_base = passk_scores_informative_base[k]
            informative_plus = passk_scores_informative_plus[k]

            row_idx, col_idx = hor_idx_map[prompt_type], ver_idx_map[k]

            ax[row_idx, col_idx].plot(num_input_tests, random_base, label = "Base w/ Random Tests")
            ax[row_idx, col_idx].plot(num_input_tests, random_plus, label = "Plus w/ Random Tests")
            ax[row_idx, col_idx].plot(num_input_tests, informative_base, label = "Base w/ Informative Tests")
            ax[row_idx, col_idx].plot(num_input_tests, informative_plus, label = "Plus w/ Informative Tests")
            ax[row_idx, col_idx].set_title("%s Pass@%d" % (prompt_type, k))
            ax[row_idx, col_idx].set
    
    fig.text(0.5, 0.08, 'Number of Tests', ha='center')
    fig.text(0.08, 0.5, 'Pass@k Score', va='center', rotation='vertical')
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    
    plt.show()