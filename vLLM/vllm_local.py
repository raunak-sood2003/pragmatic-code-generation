from vllm import LLM, SamplingParams

class vLLMAgent:
    def __init__(self, 
                 model_name_or_path,
                 program_prompt,
                 test_prompt,
                 extract_program_fn,
                 extract_test_fn,
                 sampling_params, 
                 num_gpus):
        self.model = LLM(model_name_or_path, tensor_parallel_size=num_gpus)
        self.sampling_params = SamplingParams(**sampling_params)
        self.program_prompt = program_prompt
        self.test_prompt = test_prompt
        self.extract_program_fn = extract_program_fn
        self.extract_test_fn = extract_test_fn

    def api_call(self, messages):
        return self.model.generate(messages, self.sampling_params)
    
    def post_process(self, responses, extract_fn):
        results = []
        for response in responses:
            result = []
            prompt = response.prompt
            for i in range(len(response.outputs)):
                completion = response.outputs[i].text
                generation = prompt + completion
                result.append(extract_fn(generation))
            results.append(result)
        return results
    
    def generate_programs(self, inputs):
        inputs_formatted = [self.program_prompt.format(inp) for inp in inputs]
        response = self.api_call(inputs_formatted)
        return self.post_process(response, self.extract_program_fn)
    
    def generate_tests(self, inputs):
        inputs_formatted = [self.test_prompt.format(inp) for inp in inputs]
        response = self.api_call(inputs_formatted)
        return self.post_process(response, self.extract_test_fn)

if __name__ == '__main__':
    model_name = 'codellama/CodeLlama-7b-hf'
    sampling_params = {
        'temperature' : 1,
        'max_tokens' : 128,
        'top_p' : 0.95,
        'n' : 5
    }
    num_gpus = 1

    # speaker = Agent(
    #     model_name,
    #     sampling_params,
    #     num_gpus,
    #     extract_testcase
    # )

    listener = vLLMAgent(
        model_name,
        sampling_params,
        num_gpus,
        extract_function
    )

    message1 = """
def sum_array(arr):
    \"\"\" Returns the sum of the elements in the array. \"\"\"
"""

    message2 = """
def fibonacci(n):
"""

    messages = [
        message1,
        message2
    ]

    responses = listener.generate(messages)

    for response in responses:
        for i in range(len(response)):
            print(response[i])