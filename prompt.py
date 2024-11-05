import os
import anthropic
import requests
import json
from typing import List, Tuple, Dict
import time
import pdb
from datasets import load_dataset


class HumanEvalSolver:
    def __init__(self):
        self.client = anthropic.Client()
        self.model = "claude-3-5-haiku-20241022"
        self.eval_url = "https://justinchiu--runtest-dev.modal.run"

    def generate_tests(self, problem: Dict) -> str:
        prompt = f"""
        Write comprehensive test cases for the following function:

        {problem['prompt']}

        Return only the test cases in Python code format.
        """

        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def generate_solutions(self, problem: Dict, n_samples: int) -> List[str]:
        solutions = []

        for _ in range(n_samples):
            prompt = f"""
            Write a Python implementation for the following function:

            {problem['prompt']}

            Return only the implementation code, no explanations.
            """

            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            solutions.append(response.content[0].text.strip())
            time.sleep(1)  # Rate limiting

        return solutions

    def evaluate_solution(self, code: str, entry_point: str, test_code: str) -> bool:
        request_code = f"{code}\n{test_code}"
        response = requests.post(self.eval_url, json={"codes": [request_code]})
        import pdb; pdb.set_trace()
        try:
            return response.json()[0]["success"]
        except Exception as e:
            print(f"Error evaluating solution: {e}")
            return False

    def rerank_solutions(
        self, solutions: List[str], entry_point: str, test_code: str
    ) -> List[Tuple[str, float]]:
        scored_solutions = []

        for solution in solutions:
            success = self.evaluate_solution(solution, entry_point, test_code)
            scored_solutions.append((solution, float(success)))

        return sorted(scored_solutions, key=lambda x: x[1], reverse=True)

    def solve_problem(
        self, problem: Dict, n_samples: int = 5
    ) -> List[Tuple[str, float]]:
        # Generate test cases
        test_code = self.generate_tests(problem)

        # Generate multiple solutions
        solutions = self.generate_solutions(problem, n_samples)

        # Rerank solutions based on test performance
        ranked_solutions = self.rerank_solutions(
            solutions, problem["entry_point"], test_code
        )

        return ranked_solutions


def main():
    # Load HumanEval dataset
    dataset = load_dataset("openai_humaneval")
    problems = dataset["test"]
    
    solver = HumanEvalSolver()

    # Process first problem as an example
    problem = {
        "prompt": problems[0]["prompt"],
        "entry_point": problems[0]["entry_point"],
        "test": problems[0]["test"]
    }

    ranked_solutions = solver.solve_problem(problem)

    print(f"Solutions for problem: {problems[0]['entry_point']}")
    for solution, score in ranked_solutions:
        print(f"\nScore: {score}")
        print("Solution:")
        print(solution)


if __name__ == "__main__":
    main()
