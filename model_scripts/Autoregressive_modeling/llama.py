import torch
import copy
import logging
import gc
import os
import math
import collections
import string
import re
import numpy as np
import evaluate

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

# ==========================================
# 1. 配置 (Configuration)
# ==========================================

# Job id
job_id = os.getenv('SLURM_JOB_ID') or "local_dev" # Fallback for local testing

# Faulty step select corresponding checkpoint
# 保持原有的控制文件逻辑
try:
    falutyStepFP = f"./control_{job_id}/0/faulty_step.txt"
    # Ensure directory exists if testing locally
    if not os.path.exists(os.path.dirname(falutyStepFP)):
        os.makedirs(os.path.dirname(falutyStepFP), exist_ok=True)
        # Create dummy file for local test if needed, or handle error
        if not os.path.exists(falutyStepFP):
            with open(falutyStepFP, 'w') as f: f.write("0")

    with open(falutyStepFP, 'r') as file:
        faulty_step = file.readline().strip()
except Exception as e:
    faulty_step = "0"
    print(f"Warning: Could not read faulty_step, default to 0. Error: {e}")

logFP = f"./control_{job_id}/0/output.log"
# Ensure dir exists
if not os.path.exists(os.path.dirname(logFP)):
    os.makedirs(os.path.dirname(logFP), exist_ok=True)

with open(logFP, "a") as file:
    file.write(f"Start Job: {faulty_step}\n")
    
# cutlass control file
controlFP = f"./control_{job_id}/0/perform.txt"
cutlassFP = f"./control_{job_id}/0/cutlass.txt"
SMChkFP = f"./control_{job_id}/0/split.txt"

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TEMP_OUTPUT_DIR = "./tmp_general_training_logs"

# 训练配置
NUM_TRAIN_SAMPLES = 4000 
SEED = 42
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 1e-5
MAX_SEQ_LENGTH = 1024 
MAX_STEPS = 1000  # 对应您原本的设置

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 2. 通用数据集加载与处理 (Alpaca)
# ==========================================

def load_general_dataset():
    """
    使用 Alpaca 数据集作为 General Dataset
    """
    logger.info("Loading General Dataset (Alpaca)...")
    try:
        # 使用 tatsu-lab/alpaca 作为通用指令微调数据
        # ds = load_dataset("yahma/alpaca-cleaned", split="train")
        ds = load_dataset("tatsu-lab/alpaca", split="train")
    except Exception as e:
        logger.error(f"Failed to load alpaca: {e}. Fallback to wikitext.")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # 简单适配 wikitext 到类似结构以便处理
        ds = ds.map(lambda x: {"instruction": "Continue the text:", "input": "", "output": x["text"]})
    return ds

def format_general_prompt(example):
    """
    将 Alpaca 格式 (Instruction, Input, Output) 转换为 Chat 格式
    """
    instruction = example.get('instruction', '')
    inp = example.get('input', '')
    output = example.get('output', '')
    
    if inp:
        user_content = f"{instruction}\n\nInput:\n{inp}"
    else:
        user_content = instruction

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output}
    ]
    return messages

def preprocess_general_function(examples, tokenizer):
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for i in range(len(examples['instruction'])):
        single_example = {
            'instruction': examples['instruction'][i],
            'input': examples['input'][i],
            'output': examples['output'][i]
        }
        messages = format_general_prompt(single_example)
        
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized_full = tokenizer(
            full_text, truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length", add_special_tokens=False
        )
        input_ids = tokenized_full["input_ids"]
        labels = copy.deepcopy(input_ids)
        
        # Mask 掉 Prompt 部分的 Loss
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
# 3. MMLU 评估逻辑 (Inference)
# ==========================================

def load_mmlu_eval_data(num_subjects=5):
    """加载 MMLU 验证集用于最终推理"""
    try:
        all_subjects = get_dataset_config_names("cais/mmlu")
    except Exception:
        all_subjects = ["abstract_algebra", "anatomy", "astronomy"]

    all_subjects = [s for s in all_subjects if s != "all"]
    all_subjects.sort()
    # 随机或固定选几个subject做测试，避免时间过长
    selected_subjects = all_subjects[:num_subjects]
    
    dataset_list = []
    for subject in tqdm(selected_subjects, desc=f"Loading MMLU subjects"):
        try:
            # 优先用 test 或 validation
            ds = load_dataset("cais/mmlu", subject, split="test") 
            dataset_list.append(ds)
        except Exception:
            pass
            
    if not dataset_list:
        return None
    full_ds = concatenate_datasets(dataset_list)
    return full_ds

def format_mmlu_prompt(example):
    question = example['question']
    choices = example['choices']
    answer_idx = example['answer'] 
    
    if answer_idx is None or not isinstance(answer_idx, int):
        return None, None

    answer_char = ["A", "B", "C", "D"][answer_idx]
    options_text = f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Analyze the following multiple-choice question and provide the correct answer. Output ONLY the corresponding letter (A, B, C, or D)."},
        {"role": "user", "content": f"{question}\n\n{options_text}\n\nAnswer:"},
        # Inference 时不包含 assistant 的回答
    ]
    return messages, answer_char

def evaluate_mmlu(model, tokenizer, num_samples=500):
    print("\n" + "="*30)
    print(f"Starting MMLU Inference (Samples: {num_samples})")
    print("="*30)
    
    dataset = load_mmlu_eval_data(num_subjects=10) # 加载部分subject
    if dataset is None:
        print("MMLU load failed.")
        return 0.0

    model.eval()
    correct_count = 0
    total_count = 0
    
    actual_samples = min(num_samples, len(dataset))
    subset = dataset.shuffle(seed=SEED).select(range(actual_samples))
    
    for item in tqdm(subset, desc="MMLU Inference"):
        messages, ground_truth_char = format_mmlu_prompt(item)
        if messages is None: continue 
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1, 
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=False
            )
        
        # 解析输出
        new_token_id = outputs[0][inputs.input_ids.shape[1]:]
        generated_char = tokenizer.decode(new_token_id, skip_special_tokens=True).strip().upper()
        # 清洗可能产生的标点
        generated_char = generated_char[0] if len(generated_char) > 0 else "NULL"
        
        if generated_char == ground_truth_char:
            correct_count += 1
        total_count += 1
        
    acc = correct_count / total_count if total_count > 0 else 0
    print(f"MMLU Result: {acc:.2%} ({correct_count}/{total_count})")
    return acc

# ==========================================
# 4. Squad v2 评估逻辑 (修复版)
# ==========================================

def normalize_text(s):
    """移除标点、冠词、小写化，用于 Squad F1/EM 计算"""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_gold, a_pred):
    return int(normalize_text(a_gold) == normalize_text(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_text(a_gold).split()
    pred_toks = normalize_text(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_squad_v2(model, tokenizer, num_samples=200):
    print("\n" + "="*30)
    print(f"Starting Squad Inference (Samples: {num_samples})")
    print("="*30)
    
    dataset = None
    # 尝试加载 Squad v2
    try:
        dataset = load_dataset("squad_v1", split="validation")
    except Exception as e:
        logger.error(f"Failed to load Squad v2 (likely outdated datasets lib): {e}")
        # 尝试 Fallback 到 Squad v1
        try:
            logger.info("Attempting fallback to 'squad' (v1)...")
            dataset = load_dataset("squad", split="validation")
        except Exception as e2:
            logger.error(f"Failed to load Squad v1 fallback: {e2}")
            return 0.0  # <--- [关键修复] 返回 0.0 而不是 None，防止 main 函数 crash
        
    if dataset is None:
        return 0.0

    model.eval()
    
    f1_scores = []
    em_scores = []
    
    actual_samples = min(num_samples, len(dataset))
    subset = dataset.shuffle(seed=SEED).select(range(actual_samples))
    
    for item in tqdm(subset, desc="Squad Inference"):
        context = item['context']
        question = item['question']
        
        # 适配 v1 和 v2 的不同结构
        if 'answers' in item:
            answers = item['answers'] 
            gold_texts = answers['text']
        else:
            gold_texts = []

        # 处理不可回答的问题 (v2 特性)
        if len(gold_texts) == 0:
            gold_texts = [""] 

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Read the context and answer the question. Keep it concise."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=32, 
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=False
            )
            
        new_token_id = outputs[0][inputs.input_ids.shape[1]:]
        prediction = tokenizer.decode(new_token_id, skip_special_tokens=True).strip()
        
        # 计算该样本与所有参考答案的最佳匹配
        if not gold_texts: # 防御性编程
            item_em = 0
            item_f1 = 0
        else:
            item_em = max([compute_exact(a, prediction) for a in gold_texts])
            item_f1 = max([compute_f1(a, prediction) for a in gold_texts])
        
        em_scores.append(item_em)
        f1_scores.append(item_f1)
        
    avg_em = np.mean(em_scores) if em_scores else 0.0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    
    print(f"Squad Result | EM: {avg_em:.2%} | F1: {avg_f1:.2%}")
    return avg_f1


# ==========================================
# 5. WMT16 Translation Inference Logic (新增)
# ==========================================

def simple_bleu(ref_text, hyp_text):
    """
    Simplified BLEU implementation (Unigram/Bigram Precision)
    to avoid 'pip install sacrebleu' dependency issues on cluster.
    """
    def tokenize(text):
        return [t.lower() for t in re.findall(r'\w+', text)]
    
    ref_tokens = tokenize(ref_text)
    hyp_tokens = tokenize(hyp_text)
    
    if len(hyp_tokens) == 0: return 0.0
    
    # 1-gram precision
    common = collections.Counter(ref_tokens) & collections.Counter(hyp_tokens)
    p1 = sum(common.values()) / len(hyp_tokens)
    
    # Simple brevity penalty
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    else:
        bp = 1.0
        
    return p1 * bp

def evaluate_wmt16(model, tokenizer, num_samples=200):
    print("\n" + "="*30 + "\nRunning WMT16 (DE-EN) Inference...\n" + "="*30)
    
    # Try loading WMT16 German-English
    try:
        # Load 'de-en' pair
        dataset = load_dataset("wmt16", "de-en", split="test")
    except Exception as e:
        print(f"WMT16 load failed: {e}")
        return 0.0

    model.eval()
    bleu_scores = []
    
    # Random sample
    actual_samples = min(num_samples, len(dataset))
    subset = dataset.shuffle(seed=SEED).select(range(actual_samples))
    
    for item in tqdm(subset, desc="Translation"):
        # WMT structure: item['translation'] = {'de': '...', 'en': '...'}
        src_text = item['translation']['de']
        ref_text = item['translation']['en']
        
        messages = [
            {"role": "system", "content": "You are a professional translator. Translate the following German text into English."},
            {"role": "user", "content": f"German: {src_text}\n\nEnglish Translation:"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, # Translations can be longer
                do_sample=False
            )
            
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Calculate score (0.0 to 1.0)
        score = simple_bleu(ref_text, gen_text)
        bleu_scores.append(score)
        
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    print(f"WMT16 BLEU (Approx): {avg_bleu:.4f}")
    return avg_bleu

# ==========================================
# 6. GSM8K Math Inference Logic (新增)
# ==========================================

def extract_gsm8k_answer(text):
    """
    从 GSM8K 的 ground truth 或模型生成的文本中提取最后的数字。
    标准格式通常是 '#### 123'
    """
    # 尝试寻找 #### 后的数字
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    
    # 如果没找到 ####，尝试提取文本中的所有数字并取最后一个
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None

def evaluate_gsm8k(model, tokenizer, num_samples=200):
    print("\n" + "="*30 + "\nRunning GSM8K (Math) Inference...\n" + "="*30)
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        print(f"GSM8K load failed: {e}")
        return 0.0

    model.eval()
    correct = 0
    total = 0
    
    subset = dataset.shuffle(seed=SEED).select(range(min(num_samples, len(dataset))))
    
    for item in tqdm(subset, desc="GSM8K"):
        question = item['question']
        gold_answer = extract_gsm8k_answer(item['answer'])
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in math. Solve the following problem step by step, and end your response with 'The answer is #### <number>'."},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, # 数学题需要中间推理步骤
                do_sample=False
            )
            
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred_answer = extract_gsm8k_answer(gen_text)
        
        # 比较数字（转为 float 比较，防止 120.0 != 120）
        try:
            if pred_answer and gold_answer and float(pred_answer) == float(gold_answer):
                correct += 1
        except ValueError:
            if pred_answer == gold_answer:
                correct += 1
        
        total += 1
        
    acc = correct / total if total > 0 else 0
    print(f"GSM8K Accuracy: {acc:.4%}")
    return acc

# ==========================================
# 7. Llama 3.2 专用 XLSum 评估函数
# ==========================================

def evaluate_xlsum(model, tokenizer, num_samples=200):
    print("\n" + "="*30 + "\n[Llama 3.2] Starting XLSum Inference\n" + "="*30)
    
    # 1. 加载数据集
    try:
        # XLSum 英文版
        dataset = load_dataset("csebuetnlp/xlsum", "english", split="test")
    except Exception as e:
        print(f"XLSum load failed: {e}")
        return 0.0

    # 2. 加载 Rouge 指标
    try:
        rouge = evaluate.load("rouge")
    except Exception:
        print("Please install evaluate: pip install evaluate rouge_score")
        return 0.0

    model.eval()
    
    predictions = []
    references = []
    
    # 3. 随机采样
    actual_samples = min(num_samples, len(dataset))
    subset = dataset.shuffle(seed=42).select(range(actual_samples))
    
    for item in tqdm(subset, desc="Summarizing (Llama)"):
        article = item['text']
        summary = item['summary']
        
        # --- Llama 3.2 Prompt ---
        # 显式引导结构，适合 Llama
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant. Summarize the article into a single sentence."
            },
            {
                "role": "user", 
                "content": f"Article: {article}\n\nSummary:"
            }
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # XLSum 输入较长，必须截断
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64, # 摘要通常很短
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        
        # 解码
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        predictions.append(gen_text)
        references.append(summary)
        
    # 4. 计算分数
    # use_stemmer=True 是 ROUGE 的标准做法
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    rouge_l = results["rougeL"]
    
    print(f"XLSum Result | ROUGE-L: {rouge_l:.4f}")
    return rouge_l

# ==========================================
# 8. 主程序
# ==========================================

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # 保持原有的 Date Locked Template 逻辑
    LLAMA_3_TIME_LOCKED_TEMPLATE = (
        "{% set loop_messages = messages %}"
        "{{ bos_token }}"
        "{% for message in loop_messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
                "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + content + '<|eot_id|>' }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + content + '<|eot_id|>' }}"
            "{% elif message['role'] == 'system' %}"
                "{{ '<|start_header_id|>system<|end_header_id|>\n\n' }}"
                "{{ 'Cutting Knowledge Date: December 2023\n' }}"
                "{{ 'Today Date: 23 Jan 2026\n\n' }}" 
                "{{ content + '<|eot_id|>' }}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )
    tokenizer.chat_template = LLAMA_3_TIME_LOCKED_TEMPLATE
    
    # --- A. 加载 General Training Data ---
    print("Loading training data (General)...")
    raw_dataset = load_general_dataset()
    
    print(f"Randomly selecting {NUM_TRAIN_SAMPLES} samples for training...")
    actual_train_samples = min(NUM_TRAIN_SAMPLES, len(raw_dataset))
    train_subset = raw_dataset.shuffle(seed=SEED).select(range(actual_train_samples))

    # --- B. 预处理 ---
    print("Tokenizing training data...")
    tokenized_train_dataset = train_subset.map(
        lambda x: preprocess_general_function(x, tokenizer),
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=4,
        load_from_cache_file=False 
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=None, padding=False
    )

    # --- C. 加载模型 ---
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager", 
        use_cache=False 
    )

    # --- D. Trainer 配置 ---
    # 移除训练中的 MMLU Callback，改在最后统一做 Inference，
    # 或者如果需要中途监控，可以写一个简单的 LogLossCallback，
    # 但根据需求，我们将重心放在最后的 MMLU 和 Squad。
    
    training_args = TrainingArguments(
        output_dir=TEMP_OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS, # 500 steps
        bf16=True,
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
        tokenizer=tokenizer
    )

    # --- E. 训练 ---
    print("Starting Training on General Dataset...")
    
    # 信号控制
    with open(cutlassFP, 'w') as file: file.write("f")
    with open(SMChkFP, 'w') as file: file.write("f")
    with open(controlFP, 'w') as file: file.write("t") # Allow training

    trainer.train()

    with open(controlFP, 'w') as file: file.write("f") # Stop signal if needed

    # 记录 Training Log
    log = trainer.state.log_history
    loss_history = [str(e.get("loss", "")) for e in log if "loss" in e]
    
    # --- F. 清理显存以准备 Inference ---
    torch.cuda.empty_cache()
    gc.collect()

    # 开启 Cache 加速推理
    model.config.use_cache = True

    EVAL_SAMPLES = 500

    # Task 1: Inference: MMLU ---
    mmlu_acc = evaluate_mmlu(model, tokenizer, num_samples=EVAL_SAMPLES)

    # # Task 2: Inference: Squad ---
    squad_f1 = evaluate_squad_v2(model, tokenizer, num_samples=EVAL_SAMPLES)

    # # Task 3: WMT16 (Translation) ---
    wmt_score = evaluate_wmt16(model, tokenizer, num_samples=EVAL_SAMPLES)

    # # --- Task 4: GSM8K (Math) ---
    gsm_score = evaluate_gsm8k(model, tokenizer, num_samples=EVAL_SAMPLES)

    # Task 5: 
    xlsum_score = evaluate_xlsum(model, tokenizer, num_samples=EVAL_SAMPLES)

    # Write Logs
    with open(logFP, "a") as file:
        file.write("\n--- Training Losses ---\n")
        file.write(" ".join(loss_history))
        file.write("\n--- Inference Results ---\n")
        file.write(f"MMLU Acc: {mmlu_acc:.4f}\n")
        file.write(f"SQuAD F1: {squad_f1:.4f}\n")
        file.write(f"WMT16 BLEU: {wmt_score:.4f}\n")
        file.write(f"GSM8K Acc: {gsm_score:.4f}\n")
        file.write(f"XLSum_ROUGE: {xlsum_score:.4f}\n")

    # Update faulty step file
    # with open(falutyStepFP, "r") as file:
    #     lines = file.readlines()
    # if lines:
    #     lines.pop(0)
    # with open(falutyStepFP, "w") as file:
    #     file.writelines(lines)

    print("Job Finished Successfully.")

if __name__ == "__main__":
    main()