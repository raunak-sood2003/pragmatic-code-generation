import json
from src.utils import write_jsonl

def format_humaneval_examples(problems, program_prompt_template, test_prompt_template):
    """
    Reads and formats HumanEval prompts based on program and test prompt templates. 
    Note that the HumanEval docstrings contain test cases, so these are removed for test 
    case generation prompts to avoid redundancy. Returns prompts that can be used to generate 
    programs and test cases based on HumanEval problems.

    problems: object returns from humaneval read_problems() function
    program_prompt_template: f string for program prompt formatting
    test_prompt_template: f string for test prompt formatting
    """

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

    program_prompts = []
    test_prompts = []
    for task_id in problems:                
        test_init_prompt = remove_testcase_from_prompt(problems[task_id])
        test_prompt = test_prompt_template.format(test_init_prompt)
        test_prompts.append(test_prompt)

        program_init_prompt = problems[task_id]["prompt"]
        program_prompt = program_prompt_template.format(program_init_prompt)
        program_prompts.append(program_prompt)

    return program_prompts, test_prompts


def format_mbpp_examples(data_path, program_prompt_template, test_prompt_template):
    """
    Reads and formats MBPP prompts based on passed in prompt templates; returns prompts 
    for generating programs and test cases based on MBPP problems.

    data_path: jsonl file with the following fields
        program_ctx: extracted function header used for generating programs 
                     (may include relevant test cases in docstring)
        test_ctx: extracted function header used for generating tests 
                    (SHOULD NOT include relevant test cases in docstring)
    program_prompt_template: f string for program prompt formatting
    test_prompt_template: f string for test prompt formatting
    """
    with open(data_path) as f:
        examples = [json.loads(line) for line in f]  

    program_prompts = []
    test_prompts = []
    for i in range(len(examples)):
        ex = examples[i]
        program_ctx, test_ctx = ex['program_ctx'], ex['test_ctx']
        program_prompt_format = program_prompt_template.format(program_ctx)
        test_prompt_format = test_prompt_template.format(test_ctx)
        program_prompts.append(program_prompt_format)
        test_prompts.append(test_prompt_format)
    return program_prompts, test_prompts

def create_mbpp_dataset(program_prompt_template, test_prompt_template, context_dir, mbpp_dir, res_dir):
    """
    Creates an MBPP data set for testing training pipeline. Formats program and test context with the 
    respective prompt templates and outputs a jsonl file with the following format:
    {
        program_ctx : <context to generate program>,
        test_ctx : <context to generate tests>
        program : <ground truth program>,
        tests : <ground truth tests from MBPP>
    }
    """
    with open(context_dir) as f:
        contexts = [json.loads(line) for line in f]
    with open(mbpp_dir) as f:
        mbpp = [json.loads(line) for line in f]
    
    program_ctxs, test_ctxs = format_mbpp_examples(context_dir, program_prompt_template, test_prompt_template)

    assert(len(program_ctxs) == len(test_ctxs))
    assert(len(program_ctxs) == len(mbpp))

    res = []
    for i in range(len(mbpp)):
        program = mbpp[i]['code']
        test_list = mbpp[i]['test_list']
        tests = "\n".join(test_list)
        program_ctx = program_ctxs[i]
        test_ctx = test_ctxs[i]

        res.append({
            'program_ctx' : program_ctx,
            'test_ctx' : test_ctx,
            'program' : program,
            'tests' : tests
        })
    
    write_jsonl(res_dir, res)