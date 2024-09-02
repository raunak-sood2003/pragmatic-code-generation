#!/bin/bash
#SBATCH --job-name=codellama7b_vllm
#SBATCH --output=codellama7b_vllm.out
#SBATCH --error=codellama7b_vllm.err
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=128G
#SBATCH --time=10:00:00

python3 -m vllm.entrypoints.openai.api_server --model codellama/CodeLlama-7b-hf --port 8070 --tensor-parallel-size 2 --max-num-seqs 2000