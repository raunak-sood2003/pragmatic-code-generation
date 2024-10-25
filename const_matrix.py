import requests
from tqdm import tqdm
import numpy as np

# URL for Modal server for sandboxed code execution
MODAL_URL = "https://justinchiu--runtest.modal.run/"
# Maximum number of concurrent requests
MAX_BATCH_SIZE = 64
# Number of concurrent requests
BATCH_SIZE = 50

def send_request(url, codes):
    response = requests.post(url, json={"codes": codes})
    print(response)
    return response.json()

def combine_code_solution(program, test):
    return program + "\n" + test

# Creates consistency matrix of shape (len(tests), len(programs))
def create_const_matrix(programs, tests):
    program_test_map = {
        (i, j) : combine_code_solution(programs[j], tests[i])
        for j in range(len(programs)) for i in range(len(tests))
    }

    idxs = list(program_test_map.keys())
    codes = list(program_test_map.values())
    results = []
    for i in range(0, len(codes), BATCH_SIZE):
        batch_codes = codes[i : min(i + BATCH_SIZE, len(codes))]
        batch_results = send_request(MODAL_URL, batch_codes)
        results.extend(batch_results)
    
    const_matrix = np.zeros([len(tests), len(programs)])
    for idx, result in zip(idxs, results):
        if result == ['passed']:
            const_matrix[idx[0], idx[1]] = 1
    return const_matrix

def extract_assert_values(assert_statement):
    first_idx = assert_statement.find("assert") + 6
    last_idx = len(assert_statement)
    
    if first_idx >= len(assert_statement):
        return "1", "2"
    
    if assert_statement[first_idx] == "(":
        first_idx += 1
        last_idx = len(assert_statement) - 1
    
    if first_idx >= last_idx:
        return "1", "2"
    
    assert_statement = assert_statement[first_idx : last_idx]
    equals_idx = assert_statement.find("==")
    if equals_idx == -1:
        return "1", "2"

    return assert_statement[:equals_idx - 1], assert_statement[equals_idx + 3:]
    
    
def assert_to_unittest(assert_statement):
    eq1, eq2 = extract_assert_values(assert_statement)
    return """
import unittest

class Test(unittest.TestCase):
    def test(self):
        self.assertEqual({}, {})
""".format(eq1, eq2)

if __name__ == '__main__':
    import json
    
    programs_dir = '/Users/raunaksood/Desktop/data/humaneval/codellama-13b/generations/codellama_humaneval_programs_k100.jsonl'
    tests_dir = '/Users/raunaksood/Desktop/data/humaneval/codellama-13b/generations/codellama_humaneval_tests_k100_temp0.8.jsonl'

    num_programs, num_tests = 100, 100

    with open(programs_dir) as f:
        json_programs = [json.loads(line) for line in f]
    with open(tests_dir) as f:
        json_tests = [json.loads(line) for line in f]
    
    task_ids = ['HumanEval/%d' % i for i in range(164)]
    const_matrices = np.zeros([164, num_programs, num_tests])
    for i, task_id in tqdm(enumerate(task_ids)):
        if i <= 1:
            continue
        tests = [assert_to_unittest(json_test['completion']) for json_test in json_tests[i * num_tests : i * num_tests + num_tests]]
        programs = [json_program['completion'] for json_program in json_programs[i * num_programs : i * num_programs + num_programs]]
        const_matrix = create_const_matrix(programs, tests)
        print("Num consistent: %d" % const_matrix.sum())
        const_matrices[i] = const_matrix
    np.save('./modal_const_matrices.npy', const_matrices)

    # p1 = "def add(x, y): return x + y"
    # p2 = "def sub(x, y): return x - y"
    # p3 = "def mul(x, y): return x * y"
    # p4 = "def div(x, y): return x / y"
    # p5 = "def exp(x, y): return x ** y"

    # t1 = "assert(add(1, 2) == 3)"
    # t2 = "assert(sub(1, 2) == -1)"
    # t3 = "assert(mul(1, 2) == 2)"
    # t4 = "assert(div(1, 2) == 0.5)"
    # t5 = "assert(exp(1, 2) == 1)"

    # programs = [p1, p2, p3, p4, p5]
    # tests = [assert_to_unittest(test) for test in [t1, t2, t3, t4, t5]]


    # print(create_const_matrix(programs, tests))


