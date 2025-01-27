import os
import openai
import aiohttp
import json
import asyncio
from typing import List, Tuple, Dict
import time
import re
import pdb
import ast
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
import itertools


class APIAgent:
    def __init__(
        self,
        model,
        api_key,
        base_url="https://api.together.xyz/v1",
    ):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.model = model

    def __call__(self, messages, sampling_params=None):
        default_sampling_params = {
            "max_tokens": 1024,
            "temperature": 0.8,
            "top_p": 0.7,
            "stop": ["<|eot_id|>", "<|eom_id|>"],
        }

        if sampling_params:
            default_sampling_params.update(sampling_params)

        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **default_sampling_params
        )
        return response.choices[0].message.content


class vLLMAgent:
    def __init__(self, model_name_or_path, tensor_parallel_size):
        self.model = LLM(model_name_or_path, tensor_parallel_size=tensor_parallel_size)

    def __call__(self, messages, sampling_params=None):
        default_sampling_params = {
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        if sampling_params:
            default_sampling_params.update(sampling_params)

        response = self.model.chat(
            messages,
            sampling_params=(SamplingParams(**default_sampling_params)),
        )
        return [x.text for x in response[0].outputs]


def extract_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from text that are wrapped in ```python ... ``` markers"""
    pattern = r"```python\n(.*?)```"
    matches = re.finditer(pattern, text, re.DOTALL)
    return [match.group(1).strip() for match in matches]


def extract_test_functions(text: str) -> List[str]:
    """Extract Python test functions from text"""
    test_functions = list()
    code_blocks = extract_code_blocks(text)
    if len(code_blocks) == 0:
        return []

    for block in code_blocks:
        parsed = ast.parse(block)
        for raw_func in parsed.body:
            # only consider test functions, and not individual statements
            if not isinstance(raw_func, ast.FunctionDef):
                continue

            # ensure the function is a test function
            if not raw_func.name.startswith("test"):
                continue

            # ensure the function has exactly one assert statement
            if not (len([x for x in raw_func.body if isinstance(x, ast.Assert)]) == 1):
                continue

            # remove trailing test indices in the output so functions can be reindexed after deduplication
            raw_func.name = re.sub(r"_[0-9]+$", "", raw_func.name)

            test_functions.append(ast.unparse(raw_func).strip())

    return test_functions


class Solver:
    def __init__(self, agent, eval_url):
        self.agent = agent
        self.eval_url = eval_url

    def generate_solutions(self, prompt: str, n_samples: int) -> List[str]:
        raw_code = [
            x
            for x in self.agent(
                [{"role": "user", "content": prompt}], sampling_params={"n": n_samples}
            )
        ]
        code_blocks = [extract_code_blocks(x) for x in raw_code]

        return list(set(x[0] for x in code_blocks if len(x) > 0))

    def generate_tests(self, prompt: str, n_samples: int) -> list[str]:
        outputs = self.agent(
            [{"role": "user", "content": prompt}],
            sampling_params={"n": n_samples},
        )

        return [
            re.sub(r"def test_([^(]+)", f"def test_\g<1>_{i + 1}", t)
            for i, t in enumerate(
                set(
                    itertools.chain.from_iterable(
                        [extract_test_functions(x) for x in outputs]
                    )
                )
            )
        ]

    async def evaluate_solution(self, code: str, test_code: str):
        # Returns JSON report from pytest
        request_code = f"{code}\n{test_code}"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.eval_url, json={"codes": [request_code]}
            ) as response:
                try:
                    result = await response.json()
                    return result[0]
                except Exception as e:
                    print(f"Error evaluating solution: {e}")
                    return False

    async def compute_test_solution_matrix(
        self, solutions: list[str], test_functions: list[str]
    ):
        tasks = []
        indices = []

        test_suite = "\n\n".join(test_functions)

        for i, solution in enumerate(solutions):
            tasks.append(self.evaluate_solution(solution, test_suite))
            indices.append(i)

        # Wait for all tasks to complete concurrently
        results = await asyncio.gather(*tasks)

        return results

        # # Process results
        # for (i, j), report in zip(indices, results):
        #     if report and "tests" in report:
        #         successes = sum(
        #             [test["outcome"] == "passed" for test in report["tests"]]
        #         )
        #         total = len(report["tests"])
        #         M[i, j] = successes / total

        # return M


if __name__ == "__main__":
    from datasets import load_dataset
    from mbpp_utils import process_mbpp_instance
    import time
    import os
    from dotenv import load_dotenv

    load_dotenv()

    mbpp = load_dataset("google-research-datasets/mbpp")
    mbpp = mbpp.map(process_mbpp_instance, load_from_cache_file=False)

    agent = vLLMAgent("Qwen/Qwen2.5-Coder-7B-Instruct", 1)
    solver = Solver(agent, os.getenv("MODAL_URL"))

    code_time = list()
    test_time = list()
    results_time = list()
    results = list()

    for x in mbpp["train"]:
        code = solver.generate_solutions(x["code_prompt_instruct"], 100)

        tests = solver.generate_tests(x["test_prompt_instruct"], 20)

        execution_results = asyncio.run(
            solver.compute_test_solution_matrix(code, tests)
        )

        gt_execution_results = asyncio.run(
            solver.compute_test_solution_matrix([x["code"]], tests)
        )

        eval_execution_results = asyncio.run(
            solver.compute_test_solution_matrix(code, x["test_functions"])
        )

        eval_gt_execution_results = asyncio.run(
            solver.compute_test_solution_matrix([x["code"]], x["test_functions"])
        )

        results.append(
            {
                "mbpp_instance": x,
                "code": code,
                "tests": tests,
                "execution_results": execution_results,
                "gt_execution_results": gt_execution_results,
                "eval_execution_results": eval_execution_results,
                "eval_gt_execution_results": eval_gt_execution_results,
            }
        )

        with open("mbpp_train_testcases.json", "w") as f:
            json.dump(results, f)

    print(f"Code generation time: {np.mean(code_time)}")
    print(f"Test generation time: {np.mean(test_time)}")
    print(f"Results computation time: {np.mean(results_time)}")
