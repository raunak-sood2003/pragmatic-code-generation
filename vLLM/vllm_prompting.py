from openai import OpenAI
import os
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems
from ..src.utils import extract_function, extract_testcase, extract_function_name
import json
import fire

"""
Prompting LLM at VLLM server

model_name: (string) name of the model at the server
samples: (List[string]) list of prompts
output_dir: (string) directory for results (stored at /output_dir/results.json)
params: (dict) LLM params
"""
def prompt_vllm(model_name, samples, output_dir, api_base, params):
    api_key = "EMPTY"
    client = OpenAI(
        api_key=api_key,
        base_url=api_base)
    
    responses = []
    logprobs = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in tqdm(range(len(samples))):
        sample = samples[i]
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
        
        except Exception as e:
            print(e)
    
    return responses, logprobs

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)

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

'''
Main function for prompting an LLM hosted on a vLLM server with HumanEval data. This function allows you 
to pass in the model name and parameters to run all HumanEval prompts and stores the responses in the 
output directory as a json file.
'''
def vllm_prompt_humaneval(model_name, port, to_gen_tests, num_generations, temperature, top_p, max_tokens, instruction_tuned, output_dir):
    api_base = "http://localhost:%d/v1" % port
    
    params = {
        "temperature" : temperature,
        "top_p" : top_p,
        "max_tokens" : max_tokens,
        "n" : num_generations
    }

    problems = read_problems()

    if to_gen_tests:
        prompts = []
        for i, task_id in enumerate(problems):                
            prefix = "# Write test cases for the following function.\n" if instruction_tuned else ""
            suffix = "    pass\n\nassert"
            init_prompt = remove_testcase_from_prompt(problems[task_id])
            prompt = prefix + init_prompt + suffix
            prompts.append(prompt)
        
        responses, logprobs = prompt_vllm(model_name, prompts, output_dir, api_base, params)

        res = []
        for i, task_id in enumerate(problems):    
            for j in range(len(responses[i])):
                prompt = problems[task_id]["prompt"]
                gen_code = prompt + responses[i][j]
                gen_test = extract_testcase(gen_code)
                res.append({"task_id" : task_id, "completion" : gen_test, "logprobs" : logprobs[i][j]})

        path_to_save = os.path.join(output_dir, model_name)
        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save)
        write_jsonl("%s/%s/humaneval_tests_k%d.jsonl" % (output_dir, model_name, num_generations), res)
    
    else:
        prompts = []
        for i, task_id in enumerate(problems):
            prefix = "# Complete the following function.\n" if instruction_tuned else ""
            prompt = prefix + problems[task_id]["prompt"]
            prompts.append(prompt)
            break
        
        responses, logprobs = prompt_vllm(model_name, prompts, output_dir, api_base, params)

        res = []
        for i, task_id in enumerate(problems):
            for j in range(len(responses[i])):
                prompt = problems[task_id]["prompt"]
                gen_code = prompt + responses[i][j]
                gen_function = extract_function(gen_code)
                res.append({"task_id" : task_id, "completion" : gen_function, "logprobs" : logprobs[i][j]})
            break
        
        path_to_save = os.path.join(output_dir, model_name)
        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save)
        write_jsonl("%s/%s/humaneval_programs_k%d.jsonl" % (output_dir, model_name, num_generations), res)

if __name__ == '__main__':
    fire.Fire(vllm_prompt_humaneval)

    # problems = read_problems()

    # for i, task_id in enumerate(problems):
    #     # if i != 0:
    #     #     continue
        
    #     init_prompt = problems[task_id]["prompt"]
    #     new_prompt = remove_testcase_from_prompt(problems[task_id])

    #     print("Init prompt:")
    #     print(init_prompt)
    #     print("Last char:", init_prompt[-1])
    #     print("New prompt:")
    #     print(new_prompt)
    #     print("Last char:", new_prompt[-1])

