from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    GenerationConfig
)
import json

class Agent:
    def __init__(self,
                model_name_or_path : str,
                prompt_template_dir : str,
                save_path : str,
                gen_config : dict,
                training_args : dict
            ):

        # For generation
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.gen_config = GenerationConfig(**gen_config)
        
        with open(prompt_template_dir) as f:
            prompt_template_json = json.load(f)
        self.prompt_template = prompt_template_json['prompt_template']
        
        # For training
        self.save_path = save_path
        self.training_args = TrainingArguments(**training_args, output_dir = save_path)
    
    def generate(self, inputs, num_generations):
        if isinstance(inputs, str):
            inputs_formatted = [self.prompt_template.format(inputs)]
        else:
            inputs_formatted = [self.prompt_template.format(inp) for inp in inputs]
        tokenized_input = self.tokenizer(inputs_formatted, padding = True, return_tensors = 'pt')
        outputs = self.model.generate(
            **tokenized_input,
            generation_config = self.gen_config,
            num_return_sequences = num_generations,
        )

        outputs_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)
        outputs_grouped = [outputs_decoded[i : i + num_generations] for i in range(0, len(outputs_decoded), num_generations)]
        return outputs_grouped[0] if num_generations == 1 else outputs_grouped

class Speaker(Agent):
    def __init__(self,
                model_name_or_path : str,
                prompt_template_dir : str,
                save_path : str,
                gen_config : dict,
                training_args : dict
            ):

        super().__init__(
            model_name_or_path,
            prompt_template_dir,
            save_path,
            gen_config,
            training_args
        )

class Listener(Agent):
    def __init__(self,
                model_name_or_path : str,
                prompt_template_dir : str,
                save_path : str,
                gen_config : dict,
                training_args : dict
            ):

        super().__init__(
            model_name_or_path,
            prompt_template_dir,
            save_path,
            gen_config,
            training_args
        )