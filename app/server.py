
import modal

app = modal.App("humaneval-server")

image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.9")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "America/New_York"})
    .run_commands("apt update")
    .apt_install(
        "clang", "git", "g++", "python3-tk", "zip", "unzip"
    )
    .pip_install("uv")
    .run_commands(
        "uv pip install --system pytest pytest-json-report",
    )
    .workdir("/test")
)


@app.function(
    image=image,
    concurrency_limit=64,
)
def run(code):
    import subprocess
    import json
    from pathlib import Path

    with Path("run.py").open("w") as f:
        f.write(code)

    result = subprocess.run(
        f"pytest run.py --json-report --json-report-file=report.json",
        shell=True,
        capture_output=True,
        text=True,
    )
    print("Stdout:", result.stdout)
    print("Stderr", result.stderr)

    with Path("report.json").open("r") as f:
        result = json.loads(f.read())
        return result
        #outcomes = [test["outcome"] for test in result["tests"]]
        #return outcomes

def convert_example(output, entry_point, test):
    import re
    inputs = ast.literal_eval(re.findall(r"inputs = (.*)", test)[0])
    results = ast.literal_eval(re.findall(r"results = (.*)", test)[0])
    test_text = "\n".join([
        TEST_SINGLE.format(i=i, input=repr(input), result=repr(result))
        for i, (input, result) in enumerate(zip(inputs, results))
    ])
    test_string = f"""{output}
import pytest
@pytest.fixture
def candidate():
    return {entry_point}

{TEST_PREFIX}

{test_text}
"""
    return test_string, inputs, results

@app.local_entrypoint()
async def main():
    import asyncio
    import datasets
    from debugger.analysis import convert_example

    dataset = datasets.load_dataset("evalplus/humanevalplus", split="test[:10]")

    futures = []
    for example in dataset:
        code = convert_example(
            example["prompt"] + example["canonical_solution"],
            example["entry_point"],
            example["test"],
        )
        futures.append(run.remote.aio(code))
    all_outcomes = await asyncio.gather(*futures)
    print(all_outcomes)


from pydantic import BaseModel
import asyncio
from typing import Any


class Request(BaseModel):
    codes: list[str]


web_image = modal.Image.debian_slim(python_version="3.10")


@app.function(image=web_image, timeout=60 * 20)
@modal.web_endpoint(
    method="POST",
    label=f"runtest",
)
async def runtest(data: Request) -> list[Any]:
    """Generate responses to a batch of prompts, optionally with custom inference settings."""
    futures = []
    for code in data.codes:
        futures.append(run.remote.aio(code))
    return await asyncio.gather(*futures)
