from .vllm_prompting import prompt_vllm
from human_eval.data import write_jsonl, read_problems
from ..src.utils import extract_function, extract_testcase
import json
 
if __name__ == '__main__':
    model_name = "codellama/CodeLlama-13b-hf"
    output_dir = "."
    api_base = "http://localhost:8010/v1"
    n = 1000
    
    params = {
        "temperature" : 0.8,
        "top_p" : 0.95,
        "max_tokens" : 128,
        "n" : n
    }

    problems = read_problems()
    
    to_gen_tests = True

    prob_lo, prob_hi = 0, 164

    if to_gen_tests:
        prompts = []
        for i, task_id in enumerate(problems):                
            prefix = "# Write test cases for the following function.\n"
            suffix = "    pass\n\nassert"
            prompt = prefix + problems[task_id]["prompt"] + suffix
            prompts.append(prompt)
        
        responses, logprobs = prompt_vllm(model_name, prompts, output_dir, api_base, params)

        res = []
        for i, task_id in enumerate(problems):    
            for j in range(len(responses[i])):
                prompt = problems[task_id]["prompt"]
                gen_code = prompt + responses[i][j]
                gen_test = extract_testcase(gen_code)
                res.append({"task_id" : task_id, "completion" : gen_test, "logprobs" : logprobs[i][j]})

        write_jsonl("codellama_humaneval_tests_k%d_%d-%d_with_probs.jsonl" % (n, prob_lo, prob_hi - 1), res)
    
    else:
        prompts = []
        for i, task_id in enumerate(problems):          
            if i < prob_lo:
                continue
            if i == prob_hi:
                break
            prefix = "# Complete the following function.\n"
            prompt = prefix + problems[task_id]["prompt"]
            prompts.append(prompt)
        
        responses, logprobs = prompt_vllm(model_name, prompts, output_dir, api_base, params)

        res = []
        for i in range(len(responses)):
            task_id = 'HumanEval/%d' % (prob_lo + i)
            for j in range(len(responses[i])):
                prompt = problems[task_id]["prompt"]
                gen_code = prompt + responses[i][j]
                gen_function = extract_function(gen_code)
                res.append({"task_id" : task_id, "completion" : gen_function, "logprobs" : logprobs[i][j]})
        
        write_jsonl("codellama_humaneval_programs_k%d_%d-%d_with_probs.jsonl" % (n, prob_lo, prob_hi - 1), res)

