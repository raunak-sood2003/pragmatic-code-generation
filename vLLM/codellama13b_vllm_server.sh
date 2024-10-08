#!/bin/bash
#SBATCH --job-name=codellama_vllm
#SBATCH --output=codellama_vllm.out
#SBATCH --error=codellama_vllm.err
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=128G
#SBATCH --time=10:00:00

python3 -m vllm.entrypoints.openai.api_server --model codellama/CodeLlama-13b-hf --port 8070 --tensor-parallel-size 2 --max-num-seqs 2000