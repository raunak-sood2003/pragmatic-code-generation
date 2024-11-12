from openai import OpenAI
import os
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems
from ..src.utils import extract_function, extract_testcase
from ..src.data import format_humaneval_examples, format_mbpp_examples
import json
import fire

def prompt_vllm(model_name, samples, api_base, params, dataset, to_gen_tests, output_dir):
    """
    Prompting LLM at vLLM server

    model_name: (string) name of the model at the server
    samples: (List[string]) list of prompts
    api_base: (string) IP addres and port of vLLM server
    params: (dict) LLM params
    """
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
                gen_code = responses[i][j] # NEED TO MAKE THIS GENERALIZED
                # if to_gen_tests:
                #     gen_code = extract_testcase(gen_code)
                # else:
                #     gen_code = extract_function(gen_code)
                res.append({"task_id" : task_id, "completion" : gen_code, "logprobs" : logprobs[i][j]})

            write_jsonl(output_dir, res, True)
        
        except Exception as e:
            print(e)
    
    return responses, logprobs

def remove_testcase_from_prompt(problem):
    """
    Removes test cases from HumanEval prompts (for unbiased test synthesis)
    """
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

def vllm_prompt_humaneval(model_name, port, to_gen_tests, num_generations, temperature, top_p, max_tokens, output_dir):
    """
    Main function for prompting an LLM hosted on a vLLM server with HumanEval data. This function allows you 
    to pass in the model name and parameters to run all HumanEval prompts and stores the responses in the 
    output directory as a jsonl file.
    """
    api_base = "http://localhost:%d/v1" % port
    params = {
        "temperature" : temperature,
        "top_p" : top_p,
        "max_tokens" : max_tokens,
        "n" : num_generations
    }

    program_prompt_template = "# Complete the following function.\n{}"
    test_prompt_template = "# Write test cases for the following function.\n{}    pass\n\nassert"
    program_prompts, test_prompts = format_humaneval_examples(read_problems(), program_prompt_template, test_prompt_template)
    if to_gen_tests:
        prompts = test_prompts
    else:
        prompts = program_prompts
    _, _ = prompt_vllm(model_name, prompts, api_base, params, "humaneval", to_gen_tests, output_dir)

def vllm_prompt_mbpp(model_name, port, to_gen_tests, num_generations, temperature, top_p, max_tokens, mbpp_dir, output_dir):
    """
    Main function for prompting an LLM hosted on a vLLM server with MBPP data. This function allows you to pass in the model 
    name and parameters to run a particular split of MBPP prompts and stores the responses in the output directory as a jsonl file.
    """
    api_base = "http://localhost:%d/v1" % port
    
    params = {
        "temperature" : temperature,
        "top_p" : top_p,
        "max_tokens" : max_tokens,
        "n" : num_generations
    }

    program_prompt_template = "{}"
    test_prompt_template = "{}"
    program_prompts, test_prompts = format_mbpp_examples(mbpp_dir, program_prompt_template, test_prompt_template)
    assert(len(program_prompts) == len(test_prompts))

    if to_gen_tests:
        prompts = test_prompts
    else:
        prompts = program_prompts

    _, _ = prompt_vllm(model_name, prompts, api_base, params, "mbpp", to_gen_tests, output_dir)

if __name__ == '__main__':
    fire.Fire(vllm_prompt_mbpp)