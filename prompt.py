import os
import openai
import aiohttp
import json
import asyncio
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


def convert_mbpp_tests(assert_list):
    # Generate individual test functions
    test_functions = []
    for i, assert_line in enumerate(assert_list, 1):
        test_func = f"def test{i}():\n    {assert_line}"
        test_functions.append(test_func)

    return "\n\n".join(test_functions)


class Solver:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )
        #self.model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        self.model = "Qwen/Qwen2.5-7B-Instruct-Turbo"
        #self.model = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        #self.model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        #self.model = "gpt-4o"
        self.eval_url = "https://justinchiu--runtest-dev.modal.run"

    def generate_solutions(self, prompt: str, n_samples: int) -> List[str]:
        prompt = self.SOLUTION_PROMPT.format(prompt=prompt)
        solutions = []
        for _ in range(n_samples):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.8,
                top_p=0.7,
                #top_k=50,
                stop=["<|eot_id|>","<|eom_id|>"],
            )
            #print(response.choices[0].message.content)
            #import pdb; pdb.set_trace()
            solutions.append(extract_code_blocks(response.choices[0].message.content)[0])
        return solutions

    def generate_tests(self, prompt: str, solutions: list[str], n_samples: int) -> list[str]:
        #prompt = self.TEST_PROMPT.format(prompt=prompt)
        test_suites = []
        for solution in solutions:
            prompt = self.TEST_PROMPT_SOLUTION.format(prompt=prompt, solution=solution)
            for _ in range(n_samples):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.8,
                    top_p=0.7,
                    #top_k=50,
                    stop=["<|eot_id|>","<|eom_id|>"],
                )
                test_suites.append(extract_code_blocks(response.choices[0].message.content)[0])
        return test_suites

    async def evaluate_solution(self, code: str, test_code: str):
        # Returns JSON report from pytest
        request_code = f"{code}\n{test_code}"
        async with aiohttp.ClientSession() as session:
            async with session.post(self.eval_url, json={"codes": [request_code]}) as response:
                try:
                    result = await response.json()
                    return result[0]
                except Exception as e:
                    print(f"Error evaluating solution: {e}")
                    return False

    async def compute_test_solution_matrix(
        self, solutions: list[str], test_suites: list[str]
    ):
        M = np.zeros((len(test_suites), len(solutions)))
        tasks = []
        indices = []
        
        # Create all evaluation tasks
        for i, test_code in enumerate(test_suites):
            for j, solution in enumerate(solutions):
                tasks.append(self.evaluate_solution(solution, test_code))
                indices.append((i, j))
        
        # Wait for all tasks to complete concurrently
        results = await asyncio.gather(*tasks)
        
        # Process results
        for (i, j), report in zip(indices, results):
            if report and "tests" in report:
                successes = sum([test["outcome"] == "passed" for test in report["tests"]])
                total = len(report["tests"])
                M[i,j] = successes / total
        
        return M

    def rerank_solutions(self, test_code: list[str], solutions: list[str], M: np.array):
        # naive sum over tests passed in total
        scores = M.sum(0)
        # from worst to best
        ordering = scores.argsort()
        return [(solutions[x], scores[x]) for x in ordering]

    async def solve_problem(
        self, prompt: str, n_samples: int = 5
    ) -> List[Tuple[str, float]]:
        # Generate multiple solutions
        solutions = self.generate_solutions(prompt, n_samples)
        # Generate test cases
        test_code = self.generate_tests(prompt, solutions, n_samples)

        # Rerank solutions based on test performance
        # Rows are test suites
        # Columns are solutions
        M = await self.compute_test_solution_matrix(solutions, test_code)
        return self.rerank_solutions(test_code, solutions, M)


class HumanEvalSolver(Solver):
    TEST_PROMPT = """Write comprehensive test cases for the following function:
```
{prompt}
    ...
```

Return only the test cases in Python code format, wrapped like
```python
def test_description1():
    assert ...

def test_description2():
    assert ...
```
Each test case should get its own function and have a descriptive name.

Do not repeat the original function.
"""
    TEST_PROMPT_SOLUTION = """Write comprehensive test cases for the following function:
```
{solution}
```

Return only the test cases in Python code format, wrapped like
```python
def test_description1():
    assert ...

def test_description2():
    assert ...
```
Each test case should get its own function and have a descriptive name.

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

Return only the test cases in Python code format, wrapped like
```python
def test_description1():
    assert ...

def test_description2():
    assert ...
```
Each test case should get its own function and have a descriptive name.

Do not repeat the original function.
"""
    TEST_PROMPT_SOLUTION = """Write comprehensive test cases for the following function:
```
{solution}
```

Return only the test cases in Python code format, wrapped like
```python
def test_description1():
    assert ...

def test_description2():
    assert ...
```
Each test case should get its own function and have a descriptive name.

Do not repeat the original function.
"""
    SOLUTION_PROMPT = """Write a Python function implementation for the following prompt:

{prompt}

Return only the implementation code, no explanations. Be sure to include the relevant import statements:
```python
code
```
"""


async def benchmark(dataset, example, solver):
    if dataset == "openai_humaneval":
        test_code = convert_humaneval_tests(example["test"], example["entry_point"])
        prompt = example["prompt"].rstrip()
    elif dataset == "mbpp":
        test_code = convert_mbpp_tests(example["test_list"] + example["challenge_test_list"])
        prompt_tests = convert_mbpp_tests(example["test_list"])
        prompt = example["text"] + "\n" + prompt_tests

    ranked_solutions = await solver.solve_problem(prompt)
    for solution, score in ranked_solutions:
        print(f"\nScore: {score}")
        print("Solution:")
        print(solution)

    report = await solver.evaluate_solution(solution, test_code)
    passed = all(test["outcome"] == "passed" for test in report["tests"])
    print(passed)
    pdb.set_trace()


async def collect(dataset, example, solver):
    if dataset == "openai_humaneval":
        test_code = convert_humaneval_tests(example["test"], example["entry_point"])
        prompt = example["prompt"].rstrip()
        true_code = None
        raise NotImplementedError
    elif dataset == "mbpp":
        test_code = convert_mbpp_tests(example["test_list"] + example["challenge_test_list"])
        prompt_tests = convert_mbpp_tests(example["test_list"])
        prompt = example["text"] + "\n" + prompt_tests
        true_code = example["code"]

    # should probably make these async as well
    solutions = [true_code] + solver.generate_solutions(prompt, n_samples=5)
    tests = [test_code] + solver.generate_tests(prompt, solutions, n_samples=5)

    reports = await asyncio.gather(*[
        solver.evaluate_solution(solution, test_suite)
        for solution in solutions
        for test_suite in tests
    ])

    num_tests = [len(re.findall("def.*:", test)) for test in tests]
    pass_matrix = np.zeros((len(solutions), sum(num_tests)))

    # populate pass matrix
    idx = 0
    for i, solution in enumerate(solutions):
        for j, test in enumerate(tests):
        # Discriminative test: pass on ground truth and fail on incorrect
            report = reports[idx]
            j_start = sum(num_tests[:j])
            for t, test_result in enumerate(report["tests"]):
                pass_matrix[i,j_start+t] = test_result["outcome"] == "passed"
            idx += 1

    np.save("pass_matrix.npy", pass_matrix)
    with open("outputs/solutions.json", "w") as f:
        f.write(json.dumps(solutions))
    with open("outputs/tests.json", "w") as f:
        f.write(json.dumps(tests))
    with open("outputs/num_tests.json", "w") as f:
        f.write(json.dumps(num_tests))
    with open("outputs/reports.json", "w") as f:
        f.write(json.dumps(reports))


async def main(
    dataset="mbpp",
    mode="benchmark",
):
    # argcheck
    assert dataset in ["openai_humaneval", "mbpp"]
    assert mode in ["benchmark", "collect"]

    example = None
    solver = None
    if dataset == "openai_humaneval":
        # Load HumanEval dataset
        examples = load_dataset("openai_humaneval", split="test")
        example = examples[0]
        solver = HumanEvalSolver()
    elif dataset == "mbpp":
        examples = load_dataset("mbpp", split="test")
        example = examples[0]
        solver = MbppSolver()

    # can async map over all examples in future
    if mode == "collect":
        await collect(dataset, example, solver)
    elif mode == "benchmark":
        await benchmark(dataset, example, solver)


if __name__ == "__main__":
    import strictfire
    strictfire.StrictFire(lambda *args, **kwargs: asyncio.run(main(*args, **kwargs)))
