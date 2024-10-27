from datasets import load_dataset
from dataclasses import dataclass, field
import json
from typing import Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    Trainer
)

@dataclass
class DataTrainingArguments:
    """ 
    Arguments for the training/evaluation data we are using.
    """
    train_file : str = field(
        default = None, 
        metadata = {
            "help": "The input training data file as a json. Should have columns for input programs and output test cases."
            }
        )
    
    validation_file : str = field(
        default = None,
        metadata= {
            "help": "The input validation data file as a json. Should have columns for input programs and output test cases."
        }
    )

    prompt_template_file : Optional[str] = field(
        default = None,
        metadata= {
            "help": "The json file with the prompt template used for training. File should have a single entry mapping 'prompt_template' to the template."
        }
    )


@dataclass
class ModelArguments:
    """ 
    Arguments used for model/tokenizer we are using.
    """

    model_name_or_path : str = field(
        metadata = {
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    
    tokenizer_name : str = field(
        default = None, 
        metadata = {
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        }
    )

    max_length : Optional[int] = field(
        default = 128,
        metadata= {
            "help": "The maximum length to pad the input/output sequences"
        }
    )

    cache_dir : Optional[str] = field(
        default = None,
        metadata = {
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )

    # output_dir : str = field(
    #     default = None,
    #     metadata = {
    #         "help": "Where to store the model checkpoints during training"
    #     },
    # )

def sft_trainer(model_args, data_args, training_args):
    dataset = load_dataset(
        "json", 
        data_files = {
            "train" : data_args.train_file, 
            "validation" : data_args.validation_file}, 
        cache_dir = model_args.cache_dir
    )
    
    train_dataset, val_dataset = dataset['train'], dataset['validation']

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if data_args.prompt_template_file is not None:
        with open(data_args.prompt_template_file) as f:
            json_template = json.load(f)
        prompt_template = json_template['prompt_template']
    else:
        # Default prompt template is the program itself
        prompt_template = "{}"
    
    # def preprocess_fn(example):
    #     # Apply prompt template to input program and tokenize
    #     prompt = prompt_template.format(example['program'])
    #     return tokenizer(prompt, text_target = example['test'])

    def preprocess_fn(example):
        # Apply prompt template to input program and tokenize
        prompt = prompt_template.format(example['program'])
        input_tokens = tokenizer(prompt, max_length = model_args.max_length, padding = 'max_length')
        output_tokens = tokenizer(example['test'], max_length = model_args.max_length, padding = 'max_length')
        input_tokens['labels'] = output_tokens['input_ids']
        return input_tokens
    
    tokenized_train_dataset = train_dataset.map(preprocess_fn)
    tokenized_val_dataset = val_dataset.map(preprocess_fn)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config = config, cache_dir = model_args.cache_dir)
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_train_dataset,
        eval_dataset = tokenized_val_dataset,
        tokenizer = tokenizer,
        data_collator = data_collator
    )

    trainer.train()

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses() 
    sft_trainer(model_args, data_args, training_args)