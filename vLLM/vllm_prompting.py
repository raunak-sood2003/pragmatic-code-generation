from openai import OpenAI
import os
from tqdm import tqdm
import json

"""
Prompting LLM at VLLM server

model_name: (string) name of the model at the server
samples: (List[string]) list of prompts
output_dir: (string) directory for results (stored at /output_dir/results.json)
params: (dict) LLM params
"""
def prompt_vllm(model_name, samples, output_dir, api_base, params):
    api_key = "EMPTY"
    client = OpenAI(
        api_key=api_key,
        base_url=api_base)
    
    responses = []
    logprobs = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in tqdm(range(len(samples))):
        sample = samples[i]
        print("Prompt:")
        print(sample)
        try:
            response = client.completions.create(
                model=model_name,
                prompt=sample,
                logprobs=False,
                stream=False,
                **params)

           
            completions = [choice.text for choice in response.choices]
            probs = [choice.logprobs.token_logprobs for choice in response.choices]
            responses.append(completions)
            logprobs.append(probs)
        
        except Exception as e:
            print(e)
    
    return responses, logprobs
