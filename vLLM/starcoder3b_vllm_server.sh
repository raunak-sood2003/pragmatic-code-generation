#!/bin/bash
#SBATCH --job-name=starcoder3b_vllm
#SBATCH --output=starcoder3b_vllm.out
#SBATCH --error=starcoder3b_vllm.err
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00

python -m vllm.entrypoints.openai.api_server --model bigcode/starcoder2-3b --port 8080 --max-num-seqs 2000