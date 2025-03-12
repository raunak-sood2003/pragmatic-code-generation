from src.rsa import RSA
from copy import deepcopy
import ast


def prepare_mbpp_rsa_dataset_instance(
    mbpp_instance,
    code_samples,
    test_cases,
    execution_results,
    answer_execution_results,
    mode="pragmatic_greedy",
    num_test_cases=10,
):
    rsa_test_cases = list()
    rsa = RSA.from_execution_results(
        [*code_samples, mbpp_instance["code"]],
        test_cases,
        [*execution_results, answer_execution_results],
    )
    while len(rsa.tests) > 0 and len(rsa_test_cases) < num_test_cases:
        if len(rsa.code) == 0:
            break
        next_test_return = rsa.select_tests(1, get_index=True, mode=mode)
        if next_test_return is None:
            break
        next_test, next_test_idx = next_test_return
        rsa_test_cases.append(next_test)
        rsa = rsa.filter(next_test_idx)

    rsa_mbpp_instance = deepcopy(mbpp_instance)
    if len(rsa_test_cases) == 0:
        rsa_mbpp_instance["test_output_instruct"] = None
        return rsa_mbpp_instance

    reindexed_test_cases = list()
    for idx, test in enumerate(rsa_test_cases):
        parsed = ast.parse(test)
        parsed.body[0].name = f"test_{mbpp_instance['entry_point']}_{idx + 1}"
        reindexed_test_cases.append(ast.unparse(parsed))

    rsa_mbpp_instance["test_functions"] = reindexed_test_cases
    rsa_mbpp_instance["test_output_instruct"] = (
        f"```python\n{'\n\n'.join(reindexed_test_cases)}\n```"
    )

    return rsa_mbpp_instance


def prepare_mbpp_rsa_dataset(
    execution_results_path, save_path, num_test_cases=10, mode="pragmatic_greedy"
):
    import json
    from tqdm import tqdm

    with open(execution_results_path, "r") as f:
        data = json.load(f)

    rsa_dataset = list()
    for x in tqdm(data):
        if (
            x["execution_results"]
            and x["gt_execution_results"]
            and x["code"]
            and x["tests"]
        ):
            rsa_instance = prepare_mbpp_rsa_dataset_instance(
                x["mbpp_instance"],
                x["code"],
                x["tests"],
                x["execution_results"],
                x["gt_execution_results"][0],
                mode=mode,
                num_test_cases=num_test_cases,
            )
            rsa_dataset.append(rsa_instance)

    with open(save_path, "w") as f:
        json.dump(rsa_dataset, f)
