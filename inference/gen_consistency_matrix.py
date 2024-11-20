import asyncio
import aiohttp
import json
import numpy as np
import fire
from tqdm import tqdm


eval_url = "https://justinchiu--runtest-dev.modal.run"


async def evaluate_solution(session, semaphore, code: str, test_code: str):
    # Returns JSON report from pytest
    request_code = f"{code}\n{test_code}"
    async with semaphore:
        async with session.post(eval_url, json={"codes": [request_code]}) as response:
            try:
                result = await response.json()
                return result[0]
            except Exception as e:
                print(f"Error evaluating solution: {e}")
                return False


async def generate_const_matrix(num_programs, num_tests, programs_dir, tests_dir, canonical_programs_dir, save_dir):
    """
    Generates consistency matrix (numpy matrix) given JSON files of programs, test cases and canonical programs
    generated from HumanEval/MBPP prompts. JSON files should have generated code in a "completion" entry.
    """
    with open(programs_dir) as f:
        json_programs = [json.loads(program) for program in f]
    with open(tests_dir) as f:
        json_tests = [json.loads(test) for test in f]
    with open(canonical_programs_dir) as f:
        json_canonical_programs = [json.loads(program) for program in f]

    N_PROBLEMS = len(json_canonical_programs)
    const_matrix = np.zeros([N_PROBLEMS, num_tests, num_programs + 1])

    tasks = []
    indices = []

    semaphore = asyncio.Semaphore(32)

    async with aiohttp.ClientSession() as session:
        for l in range(N_PROBLEMS):
            print("Processing task_id %d" % l)
            programs = json_programs[l * num_programs : l * num_programs + num_programs]
            tests = json_tests[l * num_tests : l * num_tests + num_tests]

            sample_programs = [programs[j]["completion"] for j in range(num_programs)]
            sample_tests = [tests[j]["completion"] for j in range(num_tests)]

            canonical_program = json_canonical_programs[l]['completion']
            sample_programs.append(canonical_program)

            for i in range(len(sample_tests)):
                for j in range(len(sample_programs)):
                    tasks.append(evaluate_solution(session, semaphore, sample_programs[j], sample_tests[i]))
                    indices.append((l,i,j))

        results = await asyncio.gather(*tasks)
        for result, idx in zip(results, indices):
            import pdb; pdb.set_trace()
            if result:
                l,i,j = idx
                const_matrix[l][i][j] = 1

    np.save(save_dir, const_matrix)

if __name__ == '__main__':
    fire.Fire(generate_const_matrix)

