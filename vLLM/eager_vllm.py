from .prompt_templates import *
from .vLLM.vllm_prompting import read_humaneval_examples
from .src.utils import extract_testcase
from human_eval.data import write_jsonl
from vllm import LLM, SamplingParams
import re

if __name__ == '__main__':
    model = "/data/models/huggingface/meta-llama/Meta-Llama-3-70B-Instruct"
    llm = LLM(model, tensor_parallel_size=4)
    sampling_params = SamplingParams(temperature = 0.8, max_tokens = 64, n = 100, top_p = 0.95)

    prompts_to_run = {
    1 : [format_prompt_informative_1shot, format_prompt_random_1shot, 'llama3_70b_humaneval_informative_icl_1shot.jsonl', 'llama3_70b_humaneval_random_icl_1shot.jsonl'],
    # 3 : [format_prompt_informative_3shot, format_prompt_random_3shot, 'llama3_70b_humaneval_informative_icl_3shot.jsonl', 'llama3_70b_humaneval_random_icl_3shot.jsonl'],
    # 5 : [format_prompt_informative_5shot, format_prompt_random_5shot, 'llama3_70b_humaneval_informative_icl_5shot.jsonl', 'llama3_70b_humaneval_random_icl_5shot.jsonl'],
    # 10 : [format_prompt_informative_10shot, format_prompt_random_10shot, 'llama3_70b_humaneval_informative_icl_10shot.jsonl', 'llama3_70b_humaneval_random_icl_10shot.jsonl']
    }

    for num_shot in prompts_to_run:
        informative_format_fn = prompts_to_run[num_shot][0]
        informative_dir_name = prompts_to_run[num_shot][2]
        informative_prompts = read_humaneval_examples(informative_format_fn)
        res_informative_dir = '/home/rrsood/CodeGen/%s' % informative_dir_name
        informative_outputs = llm.generate(informative_prompts, sampling_params=sampling_params, use_tqdm=True)
        res_informative = []
        for i, output in enumerate(informative_outputs):
            task_id = 'HumanEval/%d' % i
            for j in range(len(output.outputs)):
                generated_text = output.outputs[j].text
                generated_test = extract_testcase(generated_text)
                res_informative.append({'task_id' : task_id, 'completion' : generated_test})
        write_jsonl(res_informative_dir, res_informative)

        random_format_fn = prompts_to_run[num_shot][1]
        random_dir_name = prompts_to_run[num_shot][3]
        random_prompts = read_humaneval_examples(random_format_fn)
        res_random_dir = '/home/rrsood/CodeGen/%s' % random_dir_name
        random_outputs = llm.generate(random_prompts, sampling_params=sampling_params, use_tqdm=True)
        res_random = []
        for i, output in enumerate(random_outputs):
            task_id = 'HumanEval/%d' % i
            for j in range(len(output.outputs)):
                generated_text = output.outputs[j].text
                generated_test = extract_testcase(generated_text)
                res_random.append({'task_id' : task_id, 'completion' : generated_test})
        write_jsonl(res_random_dir, res_random)