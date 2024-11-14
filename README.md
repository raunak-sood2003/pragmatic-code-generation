# Pragmatic Neural Program/Test Case Synthesis

## Overview
Recently, LLMs have shown remarkable success on code generation tasks such as natural language -> program/test case tasks. However, it often takes several attempts for an LLM to produce the user's intended program. Accordingly, this project aims to tackle the problem of program synthesis by using informatively chosen test cases to filter out spuriously generated programs.

## Repo Layout
### src
This contains implementations of CodeT, RSA and (soon) MBR-Exec. Also contains utils file with functions to generate consistency matrices.

### inference
Contains scripts that run various inference-time experiments. 

- gen_consistency_matrix.py: this is a standalone script that generates consistency matrices as numpy arrays given model generations for HumanEval/MBPP.
- gen_codeT_results: runs CodeT on model generated programs and tests (HumanEval/MBPP) and evaluates the pass@k scores
- gen_rsa_codeT_results: runs CodeT by selecting test cases from the inference-time RSA algorithm
- gen_codeT_canonical: runs CodeT by using test cases that are consistent with the ground truth program for the problem
- gen_rsa_codeT_canonical: runs CodeT by using test cases selected by RSA from the ground truth program


### vllm
Contains scripts that prompt vLLM for generating programs and tests from language models.

### finetune
Contains scripts for supervised fine-tuning of speaker models on test cases.

- sft_trainer: trains a HuggingFace model on test cases using supervised fine-tuning
- pragmatic_trainer: creates the informative program/test pair by running RSA with the ground truth example and auto-regressively choosing test cases
- agent: contains LLM implementation of Speaker and Listener models

