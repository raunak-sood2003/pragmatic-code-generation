# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use Ray Data for running offline batch inference
distributively on a multi-nodes cluster.

Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
"""

from typing import Any, Dict, List

import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams
import json


def distributed_inference(
    model,
    input_path,
    output_path,
    tensor_parallel_size=1,
    num_instances=1,
    batch_size=1,
    temperature=0.8,
    top_p=0.9,
    num_samples=128,
    timeline_path=None,
):
    # Create a class to do batch inference.
    class LLMPredictor:
        def __init__(self):
            # Create an LLM.
            self.llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
            )
            self.sampling_params = sampling_params

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            # Generate texts from the prompts.
            # The output is a list of RequestOutput objects that contain the prompt,
            # generated text, and other information.
            outputs = self.llm.chat(
                [m.tolist() for m in batch["messages"]], self.sampling_params
            )
            batch["samples"] = [
                json.dumps([o.text for o in output.outputs]) for output in outputs
            ]
            batch["messages"] = [json.dumps(m.tolist()) for m in batch["messages"]]
            return batch

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, n=num_samples, max_tokens=512
    )

    # Read one text file from S3. Ray Data supports reading multiple files
    # from cloud storage (such as JSONL, Parquet, CSV, binary format).
    ds = ray.data.read_json(input_path)

    # For tensor_parallel_size > 1, we need to create placement groups for vLLM
    # to use. Every actor has to have its own placement group.
    def scheduling_strategy_fn():
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{"GPU": 1, "CPU": 1}] * tensor_parallel_size,
            strategy="STRICT_PACK",
        )
        return dict(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                pg, placement_group_capture_child_tasks=True
            )
        )

    resources_kwarg: Dict[str, Any] = {}
    if tensor_parallel_size == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

    # Apply batch inference for all input data.
    ds = ds.map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=num_instances,
        # Specify the batch size for inference.
        batch_size=batch_size,
        **resources_kwarg,
    )

    ds.write_json(output_path)

    if timeline_path:
        # Save the timeline data.
        ray.timeline(timeline_path)
