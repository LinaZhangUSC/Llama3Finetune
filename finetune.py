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
from transformers import TextDataset
import pandas as pd

#test model

base_model = "meta-llama/Llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")


#finetune
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

with open('/home/lina/finetuneLLama/Llama3finetune/dataProcess/CustomeData.json', 'r') as f:
    data = json.load(f)
dataset = Dataset.from_list(data)
first_datapoint = dataset[0]
print(first_datapoint)
print(type(first_datapoint))
train_dataset = dataset.train_test_split(test_size=0.1)["train"] 
eval_dataset = dataset.train_test_split(test_size=0.1)["test"]



model.eval()
def generate_inference_prompt(data_point):
    full_prompt =f"""You are an AI Assistant. You will extract Name, Age, Profession, and Hobby information from the given text description.
### text description:
{data_point["input"]}
### Response:
"""
    return full_prompt

results_before_finetune = []
for data_point in eval_dataset:
    text_input = data_point["input"]
    origin_output = data_point["output"]
    inference_input = generate_inference_prompt(data_point)
    model_input = tokenizer(inference_input, return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated_output_before_finetune = tokenizer.decode(
            model.generate(**model_input, max_new_tokens=100)[0],
            skip_special_tokens=True
        ) 
    print(generated_output_before_finetune)       
    results_before_finetune.append(
        generated_output_before_finetune.lstrip(inference_input),
    )
    
  
def generate_trainning_prompt(data_point):
    full_prompt =f"""You are a AI Assistant. You will extract Name,Age,Profession,and Hobby information from the given text description.
### text description:
{data_point["input"]}
### Response:
{data_point["output"]}
"""
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_trainning_prompt)
tokenized_val_dataset = eval_dataset.map(generate_trainning_prompt)



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

base_model = "meta-llama/Llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, output_dir+"/checkpoint-400")

model.eval()
results = []
for data_point in eval_dataset:
    text_input = data_point["input"]
    origin_output = data_point["output"]
    inference_input = generate_inference_prompt(data_point)
    model_input = tokenizer(inference_input, return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated_output_after_finetune = tokenizer.decode(
            model.generate(**model_input, max_new_tokens=100)[0],
            skip_special_tokens=True
        )    
    
    # Append the result to the list
    results.append({
        "text": text_input,
        "chatgpt output": origin_output,
        "output after finetune": generated_output_after_finetune.lstrip(inference_input),
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df['output before finetune'] = results_before_finetune
results_df.to_csv("llama_finetune_inference_results4.csv", index=False)
