from datetime import datetime
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import json
from datasets import Dataset
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel

#test model

base_model = "meta-llama/Llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
eval_prompt = """Extract Name,Age,Profession,and Hobby information from the following text description: As the sun dipped below the horizon,
 Eliza Matthews, 32, closed her laptop with a sigh. Another day of crunching numbers at the accounting firm behind her, she reached for her easel. 
 Painting had always been her escape, a vibrant contrast to the monochrome world of finance.
"""
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))


tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()
    return result


# dataset

with open('CustomeData.json', 'r') as f:
    data = json.load(f)
dataset = Dataset.from_list(data)
first_datapoint = dataset[0]
print(first_datapoint)
print(type(first_datapoint))
train_dataset = dataset.train_test_split(test_size=0.1)["train"] 
eval_dataset = dataset.train_test_split(test_size=0.1)["test"]
  
def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""You are a AI Assistant. You will extract Name,Age,Profession,and Hobby information from the given text description.
### text description:
{data_point["input"]}
### Response:
{data_point["output"]}
"""
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)



from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

model.train() # put model back into training mode
model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
wandb_project = "Llama3-3B-Finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

batch_size = 128
per_device_train_batch_size = 32
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = "Llama3-3B-finetuned"

training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        max_steps=400,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=20,
        save_steps=20,
        output_dir=output_dir,
        # save_total_limit=3,
        load_best_model_at_end=False,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="wandb", # if use_wandb else "none",
        run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
    )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
model = torch.compile(model)  

# trainer.train()

##inference after finetuning

base_model = "meta-llama/Llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
model = PeftModel.from_pretrained(model, output_dir+"/checkpoint-400")
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))