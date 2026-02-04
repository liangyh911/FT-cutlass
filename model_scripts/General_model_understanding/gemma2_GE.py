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
    TrainerCallback
)
from tqdm import tqdm

# total traing steps 500, lanuch 1000, one epoch 187, memory usage: 61859MiB /  81559MiB
# A100 80GB: baseline loss at step 500: 0.0006, time: 35 mins / 500 iter#

# ==========================================
# 1. 配置 (Configuration)
# ==========================================

# Job id 处理 (增加默认值防止非 Slurm 环境报错)
job_id = os.getenv('SLURM_JOB_ID', 'local_debug')

# Faulty step select corresponding checkpoint
# 注意：请确保这些控制文件在你的目录下存在，或者根据需要注释掉文件读取逻辑
falutyStepFP = f"./control_{job_id}/0/faulty_step.txt"
# 为了代码能跑通，这里加个 try-except，实际使用请保留你的逻辑
try:
    with open(falutyStepFP, 'r') as file:
        faulty_step = file.readline().strip()
except FileNotFoundError:
    faulty_step = 0 # Default fallback

# # 确保输出目录存在
# os.makedirs(f"./control_{job_id}", exist_ok=True)

logFP = f"./control_{job_id}/0/output.log"
with open(logFP, "a") as file:
    file.write(f"{faulty_step}\n")
    
# cutlass control file
controlFP = f"./control_{job_id}/0/perform.txt"
cutlassFP = f"./control_{job_id}/0/cutlass.txt"
SMChkFP = f"./control_{job_id}/0/split.txt"

# --- 修改点 1: 更改模型 ID 为 Gemma 2 2B IT ---
# 如果你指的是第一代 Gemma，请改为 "google/gemma-2b-it"
MODEL_ID = "google/gemma-2-2b-it" 

TEMP_OUTPUT_DIR = "./tmp_gemma_epoch_eval_logs"

PREFERRED_TRAIN_SPLIT = "test"       
PREFERRED_EVAL_SPLIT = "validation"  

NUM_SUBJECTS_TO_LOAD = 20
NUM_TRAIN_SAMPLES = 4000 
SEED = 42

BATCH_SIZE = 8 # Gemma 2B 比 Llama 1B 稍大，如果显存紧张(如 <24G)可能需要调小 BS，H100/A100 可保持 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 1e-5 # Gemma 对学习率较敏感，2e-5 是安全值
MAX_SEQ_LENGTH = 1024 
NUM_EPOCHS = 3

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
        all_subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge"]

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
    
    if answer_idx is None or not isinstance(answer_idx, int) or not (0 <= answer_idx <= 3):
        return None, None

    answer_char = ["A", "B", "C", "D"][answer_idx]
    options_text = f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
    
    # Gemma 的 chat template 会自动处理 system role (通常转为 user 的一部分或忽略，视 template 而定)
    # 保持这种通用格式即可，apply_chat_template 会处理
    messages = [
        {"role": "user", "content": f"Analyze the following multiple-choice question and provide the correct answer. Output ONLY the corresponding letter (A, B, C, or D).\n\n{question}\n\n{options_text}\n\nAnswer:"},
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
        if messages is None: continue 
        
        # 1. Tokenize 完整对话
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized_full = tokenizer(
            full_text, truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length", add_special_tokens=False
        )
        input_ids = tokenized_full["input_ids"]
        labels = copy.deepcopy(input_ids)
        
        # 2. Tokenize Prompt 部分以计算 Mask 长度
        prompt_messages = messages[:-1]
        # 注意：add_generation_prompt=True 会加上 <start_of_turn>model，这很重要
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        tokenized_prompt = tokenizer(
            prompt_text, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False, add_special_tokens=False
        )
        prompt_len = len(tokenized_prompt["input_ids"])
        
        # Mask 掉 Prompt 部分
        if prompt_len < len(labels):
            labels[:prompt_len] = [-100] * prompt_len
        else:
            labels = [-100] * len(labels)
            
        # Mask 掉 Padding 部分
        for j in range(len(labels)):
            if tokenized_full["attention_mask"][j] == 0: labels[j] = -100

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(tokenized_full["attention_mask"])
        model_inputs["labels"].append(labels)
        
    return model_inputs

# ==========================================
# 4. 评估逻辑
# ==========================================

def run_accuracy_evaluation(model, tokenizer, eval_dataset, num_samples=200):
    with open(controlFP, 'w') as file:
        file.truncate(0)
        file.write("f")

    print(f"\n[Callback] Starting evaluation on {num_samples} samples...")
    model.eval()
    
    torch.cuda.empty_cache()
    
    original_use_cache = model.config.use_cache
    model.config.use_cache = False # Evaluation 时也可以设为 True 以加速，这里保持原逻辑 False
    
    correct_count = 0
    total_count = 0
    actual_samples = min(num_samples, len(eval_dataset))
    subset = eval_dataset.shuffle(seed=SEED).select(range(actual_samples))
    
    for item in tqdm(subset, desc="Evaluating"):
        messages, ground_truth_char = format_mmlu_prompt(item)
        if messages is None: continue 
        
        prompt_messages = messages[:-1] 
        # add_generation_prompt=True 确保生成的 Prompt 以 <start_of_turn>model 结尾
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1, 
                pad_token_id=tokenizer.pad_token_id, # 使用 tokenizer.pad_token_id
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True 
            )
        
        # 提取新生成的 token
        new_token_id = outputs[0][inputs.input_ids.shape[1]:]
        generated_char = tokenizer.decode(new_token_id, skip_special_tokens=True).strip().upper()
        
        # Gemma 有时可能会输出 "A" 或者 " A"，strip() 可以处理
        # 如果模型输出了类似 "**A**" 的 Markdown 格式，这里取第一个字符可能会有问题，但对于 1 token gen 通常没问题
        prediction = generated_char[0] if len(generated_char) > 0 else "NULL"
        
        if prediction == ground_truth_char:
            correct_count += 1
        total_count += 1
        
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\n[Callback] Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    
    with open(controlFP, 'w') as file:
        file.truncate(0)
        file.write("t")
    
    model.config.use_cache = original_use_cache
    model.train()

    with open(logFP, "a") as f:
        f.write(f"{accuracy:.4f} ")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return accuracy

# ==========================================
# 5. 自定义 Callback
# ==========================================

class EpochEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, num_samples=200, eval_steps=250):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples
        self.eval_steps = eval_steps
        self.last_eval_step = -1
    
    def _perform_eval(self, model, step):
        """
        内部辅助函数，执行评估并更新记录
        """
        print(f"\n\n*** Step {step} Evaluation Triggered... ***")
        run_accuracy_evaluation(model, self.tokenizer, self.eval_dataset, self.num_samples)
        self.last_eval_step = step

    def on_step_end(self, args, state, control, **kwargs):
        """
        Trainer 会在每个 eval_steps 结束时调用此方法
        """
        if state.global_step > 0 and \
           state.global_step % self.eval_steps == 0 and \
           state.global_step != self.last_eval_step:
            
            model = kwargs['model']
            self._perform_eval(model, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """
        当训练结束时（无论是跑满 max_steps 还是提前退出）调用
        """
        if state.global_step > 0 and state.global_step != self.last_eval_step:
            print(f"\n\n*** Training Finished at Step {state.global_step}. Running Final Evaluation... ***")
            model = kwargs['model']
            self._perform_eval(model, state.global_step)

# ==========================================
# 6. 主程序
# ==========================================

def main():
    # --- 修改点 2: Tokenizer 加载 ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Gemma 通常没有默认的 pad token，这里将其设置为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 显式设置 padding 方向，训练通常用 right，推理有时用 left，这里统一 right 没问题
    tokenizer.padding_side = "right"
    
    print(f"Loading training data (Preferred: {PREFERRED_TRAIN_SPLIT})...")
    partial_train_dataset = load_mmlu_subset(PREFERRED_TRAIN_SPLIT, num_subjects=NUM_SUBJECTS_TO_LOAD)
    
    partial_train_dataset = partial_train_dataset.filter(
        lambda x: x['answer'] is not None and isinstance(x['answer'], int) and 0 <= x['answer'] <= 3
    )

    print(f"Randomly selecting {NUM_TRAIN_SAMPLES} samples for training...")
    actual_train_samples = min(NUM_TRAIN_SAMPLES, len(partial_train_dataset))
    train_subset = partial_train_dataset.shuffle(seed=SEED).select(range(actual_train_samples))

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

    print(f"Loading evaluation data (Preferred: {PREFERRED_EVAL_SPLIT})...")
    partial_eval_dataset = load_mmlu_subset(PREFERRED_EVAL_SPLIT, num_subjects=NUM_SUBJECTS_TO_LOAD)
    partial_eval_dataset = partial_eval_dataset.filter(
        lambda x: x['answer'] is not None and isinstance(x['answer'], int) and 0 <= x['answer'] <= 3
    )

    print(f"Loading Model: {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16, # Gemma 必须用 bf16
        device_map="auto",
        attn_implementation="eager", # H100 上可以用 "flash_attention_2"
        use_cache=False 
    )

    eval_callback = EpochEvalCallback(
        tokenizer=tokenizer,
        eval_dataset=partial_eval_dataset,
        num_samples=200, 
        eval_steps=250
    )

    training_args = TrainingArguments(
        output_dir=TEMP_OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_steps=1000,
        bf16=True, # 确保开启
        
        logging_strategy="steps",
        logging_steps=1,
        
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
        callbacks=[eval_callback]
    )

    print("Starting Training...")

    with open(cutlassFP, 'w') as file:
        file.truncate(0)
        file.write("f")

    with open(SMChkFP, 'w') as file:
        file.truncate(0)
        file.write("f")
    
    # Baseline Eval
    print("Running baseline evaluation...")
    run_accuracy_evaluation(model, tokenizer, partial_eval_dataset, num_samples=200)

    # Initialize control files
    with open(controlFP, 'w') as file:
        file.truncate(0)
        file.write("t")

    trainer.train()

    with open(controlFP, 'w') as file:
        file.truncate(0)
        file.write("f")

    log = trainer.state.log_history
    smchk_loss = [str(e.get("loss", "")) for e in log if "loss" in e]
    grad_norm = [str(e.get("grad_norm", "")) for e in log if "grad_norm" in e]

    torch.cuda.empty_cache()

    with open(logFP, "a") as file:
        file.write("\n")
        file.write(" ".join(smchk_loss))
        file.write("\n")
        file.write(" ".join(grad_norm))
        file.write("\n")
    
    with open(falutyStepFP, "r") as file:
        lines = file.readlines()
    lines.pop(0)
    with open(falutyStepFP, "w") as file:
        file.writelines(lines)

if __name__ == "__main__":
    main()