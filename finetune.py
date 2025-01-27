from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from dataclasses import replace


def prepare_model_for_finetuning(model_name_or_path, **kwargs):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name_or_path, load_in_4bit=False, use_gradient_checkpointing="unsloth"
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=kwargs.get("lora_r", 16),
        lora_alpha=kwargs.get("lora_alpha", 16),
        lora_dropout=kwargs.get("lora_dropout", 0),
        bias="none",
        random_state=412,
    )

    return model, tokenizer


default_training_args = dict(
    eval_strategy="steps",
    eval_steps=10,
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=10,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    bf16=True,
    optim="adamw_torch",
    report_to="wandb",
    learning_rate=1e-4,
    lr_scheduler_type="constant",
    warmup_steps=0,
    seed=412,
    eval_on_start=True,
)


def finetune(model, tokenizer, dataset, **kwargs):
    training_args = default_training_args.copy()
    training_args.update(kwargs)
    print(training_args["output_dir"])
    sft_config = SFTConfig(output_dir=training_args["output_dir"])
    for k, v in training_args.items():
        if hasattr(sft_config, k):
            setattr(sft_config, k, v)

    sft_config.remove_unused_columns = False
    sft_config.dataset_kwargs = {"skip_prepare_dataset": True}

    print(sft_config)

    FastLanguageModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        data_collator=DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template=kwargs.get("instruction_template", "<|im_start|>user"),
            response_template=kwargs.get("response_template", "<|im_start|>assistant"),
        ),
        train_dataset=dataset["train"],
        eval_dataset=(
            dataset["validation"] if sft_config.eval_strategy != "no" else None
        ),
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)


def mbpp_finetune(**kwargs):
    from mbpp_utils import process_mbpp_instance
    from datasets import load_dataset

    model, tokenizer = prepare_model_for_finetuning(**kwargs)

    mbpp = load_dataset("google-research-datasets/mbpp")
    mbpp = mbpp.map(process_mbpp_instance, load_from_cache_file=False)
    mbpp = mbpp.filter(
        lambda x: x["test_prompt_instruct"] is not None
        and x["test_output_instruct"] is not None
    )
    mbpp = mbpp.map(
        lambda x: {
            "messages": [
                {"role": "user", "content": x["test_prompt_instruct"]},
                {"role": "assistant", "content": x["test_output_instruct"]},
            ]
        },
        remove_columns=mbpp["train"].column_names,
    )
    mbpp = mbpp.map(
        lambda x: {
            k: v[0]
            for k, v in tokenizer(
                tokenizer.apply_chat_template(
                    x["messages"], tokenize=False, add_generation_prompt=False
                ),
                return_tensors="pt",
            ).items()
        },
        remove_columns=["messages"],
    )

    finetune(model, tokenizer, mbpp, **kwargs)
