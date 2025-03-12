import io
import modal
from utils import get_input_output_pair

app = modal.App("modal-exec")


@app.function(concurrency_limit=100)
def run_all_tests(code, tests):
    import subprocess
    import ast
    import json

    test_function_names = [ast.parse(test).body[0].name for test in tests]

    function_calls = [
        f'try:\n    result["{test_function_name}"] = {test_function_name}()\nexcept Exception as e:\n    result["{test_function_name}"] = None'
        for test_function_name in test_function_names
    ]

    result = subprocess.run(
        [
            "python",
            "-c",
            f"""import json\n{code}\n{'\n'.join(tests)}\nresult={{}}\n{'\n'.join(function_calls)}\nprint(json.dumps(result))""",
        ],
        capture_output=True,
    )

    return json.loads(result.stdout), result.stderr.decode("utf-8")


def run_fn(codes, tests):
    try:
        tests_io = [get_input_output_pair(test) for test in tests]
    except ValueError:
        return list()

    return modal.functions.gather(
        *[
            run_all_tests.spawn(
                codes[i],
                [t[0] for t in tests_io],
            )
            for i in range(len(codes))
        ]
    )


def main():
    import json

    from tqdm import tqdm
    import time
    from multiprocessing import Pool
    from modal.functions import gather

    with open("../mbpp_prototype_data/mbpp_train_testcases.json") as f:
        data = json.load(f)

    x = data[0]

    tests = [get_input_output_pair(test) for test in x["tests"]]

    start = time.time()

    with app.run():
        with Pool() as pool:
            outcomes = pool.starmap(
                run_fn,
                zip([x["code"] for x in data[:100]], [x["tests"] for x in data[:100]]),
            )

    print(f"Time taken: {time.time() - start}")
    with open("exec_results.json", "w") as f:
        json.dump(outcomes, f)

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    main()
    # import asyncio

    # asyncio.run(main())
