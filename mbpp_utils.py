import json
import ast
import re
from datasets import load_dataset
from prompts import (
    CODE_PROMPT_BASE_TEMPLATE,
    TEST_PROMPT_BASE_TEMPLATE,
    CODE_PROMPT_INSTRUCT_TEMPLATE,
    TEST_PROMPT_INSTRUCT_TEMPLATE,
)


def get_entry_point(mbpp_problem):
    try:
        function_name = [
            ast.parse(t).body[0].test.left.func.id for t in mbpp_problem["test_list"]
        ]
        if len(set(function_name)) == 1:
            return function_name[0]
        else:
            return None
    except:
        return None


def make_test_function(test, entry_point=None, idx=None):
    function_name = "test"
    if entry_point:
        function_name += f"_{entry_point}"

    if not idx is None:
        function_name += f"_{idx}"

    return f"""def {function_name}():\n\t{test}"""


def process_mbpp_instance(mbpp_problem, incremental=False):
    entry_point = get_entry_point(mbpp_problem)
    if entry_point is None:
        return {
            **mbpp_problem,
            "code_prompt_base": None,
            "test_prompt_base": None,
            "code_prompt_instruct": None,
            "test_prompt_instruct": None,
            "entry_point": None,
            "test_output_instruct": None,
            "test_functions": None,
        }

    if not mbpp_problem["text"].lower().startswith("write a"):
        return {
            **mbpp_problem,
            "code_prompt_base": None,
            "test_prompt_base": None,
            "code_prompt_instruct": None,
            "test_prompt_instruct": None,
            "entry_point": None,
            "test_output_instruct": None,
            "test_functions": None,
        }

    docstring = mbpp_problem["text"].lower().replace("write a", "").strip()

    test_functions = [
        make_test_function(t, entry_point, i + 1)
        for i, t in enumerate(mbpp_problem["test_list"])
    ]

    pattern = r"(.*?def\s+" + entry_point + r"\s*\([^)]*\):)"

    try:
        before, fn_signature, after = re.split(
            pattern, mbpp_problem["code"], maxsplit=1
        )
    except:
        return {
            **mbpp_problem,
            "code_prompt_base": None,
            "test_prompt_base": None,
            "code_prompt_instruct": None,
            "test_prompt_instruct": None,
            "entry_point": None,
            "test_output_instruct": None,
            "test_functions": None,
        }

    return {
        **mbpp_problem,
        "test_functions": test_functions,
        "code_prompt_base": CODE_PROMPT_BASE_TEMPLATE.format(
            before=before, function_signature=fn_signature, docstring=docstring
        ),
        "test_prompt_base": TEST_PROMPT_BASE_TEMPLATE.format(
            before=before,
            function_signature=fn_signature,
            docstring=docstring,
            function_body=after,
            # test_setup_code=mbpp_problem["test_setup_code"],
            entry_point=entry_point,
        ),
        "code_prompt_instruct": CODE_PROMPT_INSTRUCT_TEMPLATE.format(
            instruction=docstring, test_case=mbpp_problem["test_list"][0]
        ),
        "test_prompt_instruct": TEST_PROMPT_INSTRUCT_TEMPLATE.format(
            before=before,
            function_signature=fn_signature,
            docstring=docstring,
            function_body=after,
            entry_point=entry_point,
        ),
        "test_output_instruct": f"```python\n{'\n\n'.join(test_functions)}\n```",
        "entry_point": entry_point,
    }


def main():
    mbpp = load_dataset("google-research-datasets/mbpp")

    mbpp = mbpp.map(process_mbpp_instance)

    from vllm import LLM, SamplingParams

    model = LLM(model="Qwen/Qwen2.5-Coder-7B-Instruct", tensor_parallel_size=1)
    response = model.chat(
        [{"role": "user", "content": mbpp["test"][0]["test_prompt_instruct"]}],
        SamplingParams(max_tokens=256, n=100, temperature=0.7, top_p=0.95),
    )

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    main()
