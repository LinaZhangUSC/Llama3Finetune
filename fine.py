import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import prepare_model_for_kbit_training, LoraConfig
from trl import setup_chat_format
import os
from datetime import datetime
import pandas as pd

model_id =  "meta-llama/Llama-3.2-3B"

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
tokenizer.model_max_length = 2048


import json
import pandas as pd
import json
from datasets import Dataset
from datasets import load_dataset
# dataset
with open('/home/lina/finetuneLLama/Llama3finetune/dataProcess/CustomeData.json', 'r') as f:
    data = json.load(f)


def generate_trainning_prompt(data_point):
    #full_prompt =f"""Extract Name, Age, Profession, and Hobby information from the given text. Text: {data_point["input"]}  The output:{data_point["output"]}""" 
    full_prompt =f"""Text: {data_point["input"]}  The output:{data_point["output"]}""" + tokenizer.eos_token
    return full_prompt
train_data = [{"text":generate_trainning_prompt(j) } for j in data]
dataset = Dataset.from_list(train_data)

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Model setup
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    quantization_config=bnb_config
)

model, tokenizer = setup_chat_format(model, tokenizer)



def generate_inference_prompt(data_point):
    full_prompt =f"""Text: {data_point["input"]}  The output:"""

    #full_prompt =f"""Extract Name, Age, Profession, and Hobby information from the given text. Text: {data_point["input"]}  The output:"""
    return full_prompt

results_before_finetune = []
for data_point in data[0:20]:
    prompt = generate_inference_prompt(data_point)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    outputs = pipe(prompt, max_new_tokens=120, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    results_before_finetune.append(outputs[0]["generated_text"].lstrip(prompt))



model = prepare_model_for_kbit_training(model)

# LoRA configuration

peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules=["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "k_proj", "v_proj"],
    task_type="CAUSAL_LM",)

from transformers import TrainingArguments

wandb_project = "Llama3-3B-SFT"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project


args = TrainingArguments(
    output_dir="sft_model_path2",
    num_train_epochs=100,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",

    report_to="wandb", # if use_wandb else "none",
    run_name=f"Llama3-SFT-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
    )

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=512,
    tokenizer=tokenizer,
    dataset_text_field="text",
    packing=False,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    }
)
trainer.train()

new_model = "./llama3-sft-100823"
trainer.save_model(new_model)



from peft import PeftModel
base_model_reload = AutoModelForCausalLM.from_pretrained(
    model_id,
    return_dict=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)

model = PeftModel.from_pretrained(base_model_reload, new_model)
model = model.merge_and_unload()
model = model.to("cuda")
model.save_pretrained("llama-3-3b-SFT2")
tokenizer.save_pretrained("llama-3-3b-SFT2")

results = []
for data_point in data[0:20]:
    prompt = generate_inference_prompt(data_point)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    outputs = pipe(prompt, max_new_tokens=120, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    results.append({
        "text": data_point["input"],
        "chatgpt output": data_point["output"],
        "output after finetune": outputs[0]["generated_text"].lstrip(prompt),
    })
   
# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df['output before finetune'] = results_before_finetune
results_df.to_csv("llama_finetune_inference_results22.csv", index=False)

