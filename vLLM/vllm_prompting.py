from openai import OpenAI
import os
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems
from ..src.utils import extract_function, extract_testcase
import json
import fire

"""
Prompting LLM at vLLM server

model_name: (string) name of the model at the server
samples: (List[string]) list of prompts
api_base: (string) IP addres and port of vLLM server
params: (dict) LLM params
"""
def prompt_vllm(model_name, samples, api_base, params, dataset, to_gen_tests, output_dir):
    api_key = "EMPTY"
    client = OpenAI(
        api_key=api_key,
        base_url=api_base)
    
    responses = []
    logprobs = []

    if dataset != 'humaneval' and dataset != 'mbpp':
        print("Invalid data set")
        return
    
    for i in tqdm(range(len(samples))):
        sample = samples[i]
        if dataset == 'humaneval':
            task_id = "HumanEval/%d" % i
        else:
            task_id = "Mbpp/%d" % (i+1)
       
        print("Prompt:")
        print(sample)
        try:
            response = client.completions.create(
                model=model_name,
                prompt=sample,
                logprobs=False,
                stream=False,
                **params)

           
            completions = [choice.text for choice in response.choices]
            probs = [choice.logprobs.token_logprobs for choice in response.choices]
            responses.append(completions)
            logprobs.append(probs)

            res = []
            for j in range(len(responses[i])):
                gen_code = sample + responses[i][j]
                if to_gen_tests:
                    gen_code = extract_testcase(gen_code)
                else:
                    gen_code = extract_function(gen_code)
                res.append({"task_id" : task_id, "completion" : gen_code, "logprobs" : logprobs[i][j]})

            write_jsonl(output_dir, res, True)
        
        except Exception as e:
            print(e)
    
    return responses, logprobs

"""
Reads and formats HumanEval prompts for non-instruction tuned models. Returns prompts 
that can be used to generate programs and test cases based on HumanEval problems.
"""
def read_humaneval_examples():
    # Removes test cases from HumanEval docstrings
    def remove_testcase_from_prompt(problem):
        prompt = problem["prompt"]
        function_name = problem["entry_point"] + "("
        testcase_idx1 = prompt.find(">>>")
        function_idxs = [i for i in range(len(prompt)) if prompt.startswith(function_name, i)]
        if len(function_idxs) > 1:
            testcase_idx2 = function_idxs[1]
        else:
            testcase_idx2 = testcase_idx1
        
        testcase_min = min(testcase_idx1, testcase_idx2)
        testcase_max = max(testcase_idx1, testcase_idx2)

        if testcase_min == -1:
            testcase_use_idx = testcase_max
        else:
            testcase_use_idx = testcase_min
        
        res_prompt = prompt[:testcase_use_idx] + "\"\"\"" + "\n"
        return res_prompt
    
    problems = read_problems()
    program_prompts = []
    test_prompts = []
    for task_id in problems:                
        test_prefix = "# Write test cases for the following function.\n"
        test_suffix = "    pass\n\nassert"
        test_init_prompt = remove_testcase_from_prompt(problems[task_id])
        test_prompt = test_prefix + test_init_prompt + test_suffix
        test_prompts.append(test_prompt)

        program_prefix = "# Complete the following function.\n"
        program_prompt = program_prefix + problems[task_id]["prompt"]
        program_prompts.append(program_prompt)

    return program_prompts, test_prompts


'''
Main function for prompting an LLM hosted on a vLLM server with HumanEval data. This function allows you 
to pass in the model name and parameters to run all HumanEval prompts and stores the responses in the 
output directory as a jsonl file.
'''
def vllm_prompt_humaneval(model_name, port, to_gen_tests, num_generations, temperature, top_p, max_tokens, output_dir):
    api_base = "http://localhost:%d/v1" % port
    params = {
        "temperature" : temperature,
        "top_p" : top_p,
        "max_tokens" : max_tokens,
        "n" : num_generations
    }

    program_prompts, test_prompts = read_humaneval_examples()
    if to_gen_tests:
        prompts = test_prompts
    else:
        prompts = program_prompts
    _, _ = prompt_vllm(model_name, prompts, api_base, params, "humaneval", to_gen_tests, output_dir)

"""
Reads and formats MBPP prompts for non-instruction tuned models. Function takes in path to MBPP problems 
json file and returns prompts for generating programs and test cases based on MBPP problems.
"""
def read_mbpp_examples(data_path):
    def extract_function_header(code):
        header = code[:code.find(')')+2]
        if header[-1] == " ":
            header = header[:-1] + ":"
        return header
    
    def format_program_prompt(q, tests, code):
        instruction = "# Complete the following function\n"
        prompt = "{}\n    \"\"\"    \n    {}\n    >>> {}    \n    \"\"\"".format(extract_function_header(code), q.strip(), "\n    >>> ".join(tests))
        return instruction + prompt
    
    def format_test_prompt(q, code):
        instruction = "# Write test cases for the following function\n"
        prompt = "{}\n    \"\"\"    \n    {}\n    \"\"\"\n    pass\n\nassert(".format(extract_function_header(code), q.strip())
        return instruction + prompt

    examples = [json.loads(x) for x in open(data_path)]  

    program_prompts = []
    test_prompts = []
    for i in range(len(examples)):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        
        program_prompt_format = format_program_prompt(q, test, code)
        test_prompt_format = format_test_prompt(q, code)

        program_prompts.append(program_prompt_format)
        test_prompts.append(test_prompt_format)
    return program_prompts, test_prompts

'''
Main function for prompting an LLM hosted on a vLLM server with MBPP data. This function allows you to pass in the model 
name and parameters to run a particular split of MBPP prompts and stores the responses in the output directory as a jsonl file.
'''
def vllm_prompt_mbpp(model_name, port, to_gen_tests, num_generations, temperature, top_p, max_tokens, mbpp_dir, output_dir):
    api_base = "http://localhost:%d/v1" % port
    
    params = {
        "temperature" : temperature,
        "top_p" : top_p,
        "max_tokens" : max_tokens,
        "n" : num_generations
    }

    program_prompts, test_prompts = read_mbpp_examples(mbpp_dir)
    assert(len(program_prompts) == len(test_prompts))

    if to_gen_tests:
        prompts = test_prompts
    else:
        prompts = program_prompts

    _, _ = prompt_vllm(model_name, prompts, api_base, params, "mbpp", to_gen_tests, output_dir)

if __name__ == '__main__':
    fire.Fire(vllm_prompt_mbpp)