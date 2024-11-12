from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig
)
import json

class Agent:
    def __init__(self,
                model_name_or_path : str,
                gen_config : dict,
                device : str
            ):
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map = 'auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.gen_config = GenerationConfig(**gen_config)
        self.device = device
    
    def generate(self, inputs, num_generations):
        input_is_str = False
        if isinstance(inputs, str):
            input_is_str = True
        
        if input_is_str:
            inputs = [inputs]

        tokenized_input = self.tokenizer(inputs, padding = True, return_tensors = 'pt').to(self.device)
        outputs = self.model.generate(
            **tokenized_input,
            generation_config = self.gen_config,
            num_return_sequences = num_generations,
        )

        outputs_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)
        outputs_grouped = [outputs_decoded[i : i + num_generations] for i in range(0, len(outputs_decoded), num_generations)]
        return outputs_grouped[0] if input_is_str else outputs_grouped

class Speaker(Agent):
    def __init__(self,
                model_name_or_path : str,
                gen_config : dict,
                device : str
            ):

        super().__init__(
            model_name_or_path,
            gen_config,
            device
        )

class Listener(Agent):
    def __init__(self,
                model_name_or_path : str,
                gen_config : dict,
                device : str
            ):

        super().__init__(
            model_name_or_path,
            gen_config,
            device
        )