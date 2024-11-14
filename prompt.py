import os
import together
import requests
import fire
import json
from typing import List, Tuple, Dict
import time
import re
import pdb
import numpy as np
from datasets import load_dataset


def extract_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from text that are wrapped in ```python ... ``` markers"""
    pattern = r"```python\n(.*?)```"
    matches = re.finditer(pattern, text, re.DOTALL)
    return [match.group(1).strip() for match in matches]


def convert_humaneval_tests(test_code, entrypoint):
    # Split the input into lines and clean up
    lines = test_code.strip().split("\n")

    # Find all assert lines
    assert_lines = [line for line in lines if line.lstrip().startswith("assert")]

    # Generate individual test functions
    test_functions = [f"candidate = {entrypoint}"]
    for i, assert_line in enumerate(assert_lines, 1):
        test_func = f"def test{i}():\n{assert_line}"
        test_functions.append(test_func)

    return "\n\n".join(test_functions)


def convert_mbpp_tests(assert_list, entrypoint):
    # Generate individual test functions
    test_functions = []
    for i, assert_line in enumerate(assert_list, 1):
        test_func = f"def test{i}():\n{assert_line}"
        test_functions.append(test_func)

    return "\n\n".join(test_functions)


class Solver:
    def __init__(self):
        self.client = together.Together()
        #self.model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        self.model = "Qwen/Qwen2.5-7B-Instruct-Turbo"
        self.eval_url = "https://justinchiu--runtest-dev.modal.run"

    def generate_solutions(self, prompt: str, n_samples: int) -> List[str]:
        prompt = self.SOLUTION_PROMPT.format(prompt=prompt)
        solutions = []
        for _ in range(n_samples):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=1.0,
                top_p=0.7,
                top_k=50,
                stop=["<|eot_id|>","<|eom_id|>"],
            )
            #print(response.choices[0].message.content)
            #import pdb; pdb.set_trace()
            solutions.append(extract_code_blocks(response.choices[0].message.content)[0])
        return solutions

    def generate_tests(self, prompt: str, n_samples: int) -> str:
        prompt = self.TEST_PROMPT.format(prompt=prompt)
        test_suites = []
        for _ in range(n_samples):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0,
                top_p=0.7,
                top_k=50,
                stop=["<|eot_id|>","<|eom_id|>"],
            )
            #print(response.choices[0].message.content)
            test_suites.append(extract_code_blocks(response.choices[0].message.content)[0])

        return test_suites

    def evaluate_solution(self, code: str, test_code: str):
        # Returns JSON report from pytest
        request_code = f"{code}\n{test_code}"
        response = requests.post(self.eval_url, json={"codes": [request_code]})
        try:
            return response.json()[0]
        except Exception as e:
            print(f"Error evaluating solution: {e}")
            return False

    def compute_test_solution_matrix(
        self, solutions: list[str], test_suites: list[str]
    ):
        M = np.zeros((len(test_suites), len(solutions)))

        for i, test_code in enumerate(test_suites):
            for j, solution in enumerate(solutions):
                report = self.evaluate_solution(solution, test_code)
                successes = sum([test["outcome"] == "passed" for test in report["tests"]])
                total = len(report["tests"])
                scored_solutions.append((solution, successes / total))
                M[i,j] = successes / total

        return M

    def solve_problem(
        self, problem: Dict, n_samples: int = 5
    ) -> List[Tuple[str, float]]:
        # Generate test cases
        test_code = self.generate_tests(problem, n_samples)

        # Generate multiple solutions
        solutions = self.generate_solutions(problem, n_samples)

        # Rerank solutions based on test performance
        M = self.compute_test_solution_matrix(solutions, test_code)
        import pdb; pdb.set_trace()
        return
        #return self.rerank_solutions(solutions, test_code)


class HumanEvalSolver(Solver):
    TEST_PROMPT = """Write comprehensive test cases for the following function:
{prompt}
    ...

Return only the test cases in Python code format, wrapped like
```python
code
```
Each test case should get its own function.

Do not repeat the original function.
"""
    SOLUTION_PROMPT = """Write a Python implementation for the following function:

{prompt}

Return only the implementation code, no explanations. Be sure to include the relevant import statements:
```python
code
```
"""

class MbppSolver(Solver):
    TEST_PROMPT = """Write comprehensive test cases for the following prompt:
{prompt}
    ...

Return only the test cases in Python code format, wrapped like
```python
code
```
Each test case should get its own function.

Do not repeat the original function.
"""
    SOLUTION_PROMPT = """Write a Python function implementation for the following prompt:

{prompt}

Return only the implementation code, no explanations. Be sure to include the relevant import statements:
```python
code
```
"""


def main(dataset="openai_humaneval"):
    assert dataset in ["openai_humaneval", "mbpp"]
    if dataset == "openai_humaneval":
        # Load HumanEval dataset
        dataset = load_dataset("openai_humaneval", split="test")
        test_code = convert_humaneval_tests(dataset[0]["test"], dataset[0]["entry_point"])
        solver = HumanEvalSolver()
    elif dataset == "mbpp":
        dataset = load_dataset("openai_humaneval", split="test")
        test_code = convert_mbpp_tests(dataset[0]["test"], dataset[0]["entry_point"])
        solver = MbppSolver()
    ranked_solutions = solver.solve_problem(dataset[0])
    for solution, score in ranked_solutions:
        print(f"\nScore: {score}")
        print("Solution:")
        print(solution)

    report = solver.evaluate_solution(solution, test_code)
    passed = all(test["outcome"] == "passed" for test in report["tests"])
    print(passed)
    pdb.set_trace()


if __name__ == "__main__":
    fire.Fire(main)
