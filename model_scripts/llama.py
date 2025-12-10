import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import math

# Job id
job_id = os.getenv('SLURM_JOB_ID')

# Faulty step select corresponding checkpoint
falutyStepFP = f"/home/yuhangl/control_{job_id}/faulty_step.txt"
with open(falutyStepFP, 'r') as file:
    faulty_step = int(file.readline())
faulty_epoch = math.floor(faulty_step / 250)
local_faulty_step = faulty_step % 250

logFP = f"/home/yuhangl/control_{job_id}/output.log"
with open(logFP, "a") as file:
    file.write(f"{faulty_step}, ({faulty_epoch}, {local_faulty_step})\n")

# faulty_epoch = 0

# if faulty_epoch == 0:
checkpoint = "meta-llama/Llama-3.2-1B"
# else:
    # checkpoint = f"./llama1b-finetuned/checkpoint-{faulty_epoch * 250}"

# print(f"faulty step: {faulty_step}, faulty epoch: {faulty_epoch}, local faulty step: {local_faulty_step}, selected checkpoint: {faulty_epoch * 250}")

# 1. load model
# model_name = "meta-llama/Llama-3.2-1B"
model_name = checkpoint

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,   # H100 推荐 bf16
    # torch_dtype=torch.float32,
    # torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager" 
)

tokenizer.pad_token = tokenizer.eos_token

# 2. load data
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

train_dataset = dataset["train"].select(range(2000))
eval_dataset  = dataset["validation"].select(range(500))

# tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset  = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# tokenized_datasets = dataset.map(
#     tokenize_function,
#     batched=True,
#     remove_columns=["text"],
# )

# 3. data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    seed = 123
)

# 4. Training agruments
# training_args = TrainingArguments(
#     output_dir="./llama1b-finetuned",
#     overwrite_output_dir=True,
#     num_train_epochs=1,
#     per_device_train_batch_size=8,
#     gradient_accumulation_steps=1,
#     learning_rate=2e-5,
#     # max_steps = 10,
#     # warmup_steps=100,
#     # logging_steps=50,
#     save_strategy="no",
#     # evaluation_strategy="epoch",
#     logging_strategy="epoch",
#     bf16=False,             
#     fp16=False,
#     gradient_checkpointing=True,  
#     # save_total_limit=2,
#     # report_to="none",     
# )

# # 5. Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )

# 6. begin training
controlFP = f"/home/yuhangl/control_{job_id}/perform.txt"
cutlassFP = f"/home/yuhangl/control_{job_id}/cutlass.txt"
SMChkFP = f"/home/yuhangl/control_{job_id}/split.txt"

# controlFP = "/home/yuhangl/control/perform.txt"
# cutlassFP = "/home/yuhangl/control/cutlass.txt"
# SMChkFP = "/home/yuhangl/control/split.txt"

loss = 0
grad_norm = 0
Iter = 1
fault_free_loss = 2.5013

num_epoch = 10

for i in range(Iter):

    training_args = TrainingArguments(
        output_dir="./llama1b",
        overwrite_output_dir=True,
        # num_train_epochs=20,
        # per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        max_steps = 1,
        # warmup_steps=100,
        # logging_steps=50,
        save_strategy="no",
        # evaluation_strategy="epoch",
        logging_strategy="epoch",
        bf16=True,            
        fp16=False,
        gradient_checkpointing=True,   # reduce memory usage
        # save_total_limit=2,
        # report_to="none",      
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, 
        # torch_dtype=torch.float32,
        # torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager" 
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    with open(controlFP, 'w') as file:
        file.truncate(0)
        file.write("t")

    with open(cutlassFP, 'w') as file:
        file.truncate(0)
        file.write("f")

    with open(SMChkFP, 'w') as file:
        file.truncate(0)
        file.write("f")

    trainer.train()

    log = trainer.state.log_history

    smchk_loss = [str(e['loss']) for e in log[:-1]]
    grad_norm = [str(e['grad_norm']) for e in log[:-1]]
    # loss += smchk_loss[0]

    torch.cuda.empty_cache()

    with open(logFP, "a") as file:
        file.write(", ".join(smchk_loss))
        file.write("\n")
        file.write(", ".join(grad_norm))

    with open(falutyStepFP, "r") as file:
        lines = file.readlines()
    lines.pop(0)
    with open(falutyStepFP, "w") as file:
        file.writelines(lines)

# print("Loss of SM-Checker: ")
# print(f"1st epoch: {loss/Iter}")
    
# print("1st epoch: ", smchk_loss[0], ", 2nd epoch: ", smchk_loss[1], ", 3rd epoch: ", smchk_loss[2])
