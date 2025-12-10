import torch
import copy
import logging
import gc
import os
import math

from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback  # <--- 新增导入
)
from tqdm import tqdm

# ==========================================
# 1. 配置 (Configuration)
# ==========================================

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
    
# cutlass control file
controlFP = f"/home/yuhangl/control_{job_id}/perform.txt"
cutlassFP = f"/home/yuhangl/control_{job_id}/cutlass.txt"
SMChkFP = f"/home/yuhangl/control_{job_id}/split.txt"

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TEMP_OUTPUT_DIR = "./tmp_mmlu_epoch_eval_logs"

PREFERRED_TRAIN_SPLIT = "test"       
PREFERRED_EVAL_SPLIT = "validation"  

NUM_SUBJECTS_TO_LOAD = 20
NUM_TRAIN_SAMPLES = 4000 
SEED = 42

BATCH_SIZE = 32
GRAD_ACCUMULATION = 2
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 256 
NUM_EPOCHS = 3          # <--- 跑 3 个 epoch 以观察变化

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 2. 数据加载函数
# ==========================================

def get_best_available_split(dataset_dict, preferred_split):
    available_splits = list(dataset_dict.keys())
    if preferred_split in available_splits:
        return dataset_dict[preferred_split]
    priority = ["auxiliary_train", "test", "train", "validation", "dev"]
    for p in priority:
        if p in available_splits:
            logger.warning(f"Preferred split '{preferred_split}' not found. Fallback to '{p}'.")
            return dataset_dict[p]
    return dataset_dict[available_splits[0]]

def load_mmlu_subset(preferred_split, num_subjects=20):
    try:
        all_subjects = get_dataset_config_names("cais/mmlu")
    except Exception:
        all_subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge"] # Fallback

    all_subjects = [s for s in all_subjects if s != "all"]
    selected_subjects = all_subjects[:num_subjects]
    
    dataset_list = []
    for subject in tqdm(selected_subjects, desc=f"Loading subjects"):
        try:
            ds_dict = load_dataset("cais/mmlu", subject)
            target_ds = get_best_available_split(ds_dict, preferred_split)
            dataset_list.append(target_ds)
        except Exception as e:
            logger.error(f"Failed to load subject {subject}: {e}")
            
    if not dataset_list:
        raise ValueError("No data found.")
    full_ds = concatenate_datasets(dataset_list)
    return full_ds

# ==========================================
# 3. 预处理逻辑
# ==========================================

def format_mmlu_prompt(example):
    question = example['question']
    choices = example['choices']
    answer_idx = example['answer'] 
    
    # 防止脏数据导致 crash
    if answer_idx is None or not isinstance(answer_idx, int) or not (0 <= answer_idx <= 3):
        return None, None

    answer_char = ["A", "B", "C", "D"][answer_idx]
    options_text = f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Analyze the following multiple-choice question and provide the correct answer. Output ONLY the corresponding letter (A, B, C, or D)."},
        {"role": "user", "content": f"{question}\n\n{options_text}\n\nAnswer:"},
        {"role": "assistant", "content": answer_char}
    ]
    return messages, answer_char

def preprocess_function(examples, tokenizer):
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for i in range(len(examples['question'])):
        single_example = {
            'question': examples['question'][i],
            'choices': examples['choices'][i],
            'answer': examples['answer'][i]
        }
        messages, ans_char = format_mmlu_prompt(single_example)
        if messages is None: continue # 跳过脏数据
        
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized_full = tokenizer(
            full_text, truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length", add_special_tokens=False
        )
        input_ids = tokenized_full["input_ids"]
        labels = copy.deepcopy(input_ids)
        
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        tokenized_prompt = tokenizer(
            prompt_text, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False, add_special_tokens=False
        )
        prompt_len = len(tokenized_prompt["input_ids"])
        
        if prompt_len < len(labels):
            labels[:prompt_len] = [-100] * prompt_len
        else:
            labels = [-100] * len(labels)
            
        for j in range(len(labels)):
            if tokenized_full["attention_mask"][j] == 0: labels[j] = -100

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(tokenized_full["attention_mask"])
        model_inputs["labels"].append(labels)
        
    return model_inputs

# ==========================================
# 4. 评估逻辑 (封装为函数以便 Callback 调用)
# ==========================================

def run_accuracy_evaluation(model, tokenizer, eval_dataset, num_samples=200):
    with open(controlFP, 'w') as file:
        file.truncate(0)
        file.write("f")

    print(f"\n[Callback] Starting evaluation on {num_samples} samples...")
    model.eval()
    
    # 显存清理：防止 OOM
    torch.cuda.empty_cache()
    
    # 临时开启 cache 加速推理
    original_use_cache = model.config.use_cache
    model.config.use_cache = True
    
    correct_count = 0
    total_count = 0
    actual_samples = min(num_samples, len(eval_dataset))
    subset = eval_dataset.shuffle(seed=SEED).select(range(actual_samples))
    
    for item in tqdm(subset, desc="Evaluating"):
        messages, ground_truth_char = format_mmlu_prompt(item)
        if messages is None: continue 
        
        prompt_messages = messages[:-1] 
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1, 
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=False
            )
        new_token_id = outputs[0][inputs.input_ids.shape[1]:]
        generated_char = tokenizer.decode(new_token_id, skip_special_tokens=True).strip().upper()
        
        prediction = generated_char[0] if len(generated_char) > 0 else "NULL"
        if prediction == ground_truth_char:
            correct_count += 1
        total_count += 1
        
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\n[Callback] Epoch Result | Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    
    # 恢复训练状态
    with open(controlFP, 'w') as file:
        file.truncate(0)
        file.write("t")
    model.config.use_cache = original_use_cache
    model.train()
    
    # 再次清理显存
    torch.cuda.empty_cache()
    gc.collect()
    
    return accuracy

# ==========================================
# 5. 自定义 Callback 类
# ==========================================

class EpochEvalCallback(TrainerCallback):
    """
    在每个 Epoch 结束时触发 MMLU 评估
    """
    def __init__(self, tokenizer, eval_dataset, num_samples=200):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Trainer 会在每个 epoch 结束时自动调用此方法
        """
        print(f"\n\n*** Epoch {state.epoch:.1f} Finished. Running Evaluation... ***")
        model = kwargs['model']
        run_accuracy_evaluation(model, self.tokenizer, self.eval_dataset, self.num_samples)

# ==========================================
# 6. 主程序
# ==========================================

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # --- A. 加载数据 ---
    print(f"Loading training data (Preferred: {PREFERRED_TRAIN_SPLIT})...")
    partial_train_dataset = load_mmlu_subset(PREFERRED_TRAIN_SPLIT, num_subjects=NUM_SUBJECTS_TO_LOAD)
    
    # 过滤脏数据
    print("Filtering invalid samples...")
    partial_train_dataset = partial_train_dataset.filter(
        lambda x: x['answer'] is not None and isinstance(x['answer'], int) and 0 <= x['answer'] <= 3
    )

    print(f"Randomly selecting {NUM_TRAIN_SAMPLES} samples for training...")
    actual_train_samples = min(NUM_TRAIN_SAMPLES, len(partial_train_dataset))
    train_subset = partial_train_dataset.shuffle(seed=SEED).select(range(actual_train_samples))

    # --- B. 预处理 ---
    print("Tokenizing training data...")
    tokenized_train_dataset = train_subset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=partial_train_dataset.column_names,
        num_proc=8,
        load_from_cache_file=False 
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=None, padding=False
    )

    # --- C. 准备评估集 ---
    print(f"Loading evaluation data (Preferred: {PREFERRED_EVAL_SPLIT})...")
    partial_eval_dataset = load_mmlu_subset(PREFERRED_EVAL_SPLIT, num_subjects=NUM_SUBJECTS_TO_LOAD)
    partial_eval_dataset = partial_eval_dataset.filter(
        lambda x: x['answer'] is not None and isinstance(x['answer'], int) and 0 <= x['answer'] <= 3
    )

    # --- D. 加载模型 ---
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager", # H100 可以改回 flash_attention_2
        use_cache=False 
    )

    # --- E. 初始化 Callback ---
    eval_callback = EpochEvalCallback(
        tokenizer=tokenizer,
        eval_dataset=partial_eval_dataset,
        num_samples=200 # 每个 epoch 结束测 200 条
    )

    # --- F. Trainer 配置 ---
    training_args = TrainingArguments(
        output_dir=TEMP_OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        # num_train_epochs=NUM_EPOCHS, # 3 个 Epoch
        num_train_epochs = 10,
        bf16=True,
        logging_strategy="epoch",
        
        # 关键设置：关闭内置 eval，使用 callback
        eval_strategy="no", 
        save_strategy="no",
        
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        remove_unused_columns=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[eval_callback] # <--- 注入 Callback
    )

    print("Starting Training...")
    
    # 可选：训练前先跑一次 Baseline
    print("Running baseline evaluation...")
    run_accuracy_evaluation(model, tokenizer, partial_eval_dataset, num_samples=200)

    with open(controlFP, 'w') as file:
        file.truncate(0)
        file.write("t")

    with open(cutlassFP, 'w') as file:
        file.truncate(0)
        file.write("f")

    with open(SMChkFP, 'w') as file:
        file.truncate(0)
        file.write("f")

    # 开始训练 loop
    trainer.train()

    with open(controlFP, 'w') as file:
        file.truncate(0)
        file.write("f")

    log = trainer.state.log_history
    smchk_loss = [str(e.get("loss", "")) for e in log if "loss" in e]
    grad_norm = [str(e.get("grad_norm", "")) for e in log if "grad_norm" in e]

    torch.cuda.empty_cache()


    with open(logFP, "a") as file:
        file.write(", ".join(smchk_loss))
        file.write("\n")
        file.write(", ".join(grad_norm))
        file.write("\n")
        # file.write(str(final_F1_score))
        # file.write("\n")

    with open(falutyStepFP, "r") as file:
        lines = file.readlines()
    lines.pop(0)
    with open(falutyStepFP, "w") as file:
        file.writelines(lines)

if __name__ == "__main__":
    main()