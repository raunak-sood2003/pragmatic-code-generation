from .prompt_templates import *
from .vLLM.vllm_prompting import prompt_vllm, remove_testcase_from_prompt
from human_eval.data import read_problems
import fire

def read_humaneval_examples(prompt_template_fn):
    problems = read_problems()
    test_prompts = []
    for task_id in problems:                
        test_init_prompt = remove_testcase_from_prompt(problems[task_id])
        test_prompt = prompt_template_fn(test_init_prompt)
        test_prompts.append(test_prompt)

    return test_prompts

def icl_prompt_tests(model_name, port, num_generations, temperature, top_p, max_tokens, num_shots, informative, output_dir):
    api_base = "http://localhost:%d/v1" % port
    params = {
        "temperature" : temperature,
        "top_p" : top_p,
        "max_tokens" : max_tokens,
        "n" : num_generations
    }

    if informative:
        if num_shots == 1:
            prompts = read_humaneval_examples(format_prompt_informative_1shot)
        elif num_shots == 3:
            prompts = read_humaneval_examples(format_prompt_informative_3shot)
        elif num_shots == 5:
            prompts = read_humaneval_examples(format_prompt_informative_5shot)
        else:
            assert(num_shots == 10)
            prompts = read_humaneval_examples(format_prompt_informative_10shot)
    else:
        if num_shots == 1:
            prompts = read_humaneval_examples(format_prompt_random_1shot)
        elif num_shots == 3:
            prompts = read_humaneval_examples(format_prompt_random_3shot)
        elif num_shots == 5:
            prompts = read_humaneval_examples(format_prompt_random_5shot)
        else:
            assert(num_shots == 10)
            prompts = read_humaneval_examples(format_prompt_random_10shot)
    
    _, _ = prompt_vllm(model_name, prompts, api_base, params, "humaneval", True, output_dir)

if __name__ == '__main__':
    fire.Fire(icl_prompt_tests)