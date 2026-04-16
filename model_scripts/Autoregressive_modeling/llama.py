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
job_id = os.getenv('SLURM_JOB_ID') or "local_dev" 

# Faulty step select corresponding checkpoint
try:
    falutyStepFP = f"./control_{job_id}/0/faulty_step.txt"
    if not os.path.exists(os.path.dirname(falutyStepFP)):
        os.makedirs(os.path.dirname(falutyStepFP), exist_ok=True)
        if not os.path.exists(falutyStepFP):
            with open(falutyStepFP, 'w') as f: f.write("0")

    with open(falutyStepFP, 'r') as file:
        faulty_step = file.readline().strip()
except Exception as e:
    faulty_step = "0"
    # print(f"Warning: Could not read faulty_step, default to 0. Error: {e}")

logFP = f"./control_{job_id}/0/output.log"
if not os.path.exists(os.path.dirname(logFP)):
    os.makedirs(os.path.dirname(logFP), exist_ok=True)

with open(logFP, "a") as file:
    # file.write(f"Start Job: {faulty_step}\n")
    file.write(f"{faulty_step}\n")
    
# cutlass control file
controlFP = f"./control_{job_id}/0/perform.txt"
cutlassFP = f"./control_{job_id}/0/cutlass.txt"
Enable_Core_Checker = f"./control_{job_id}/0/enable_core_checker.txt"
DEBUG = f"./control_{job_id}/0/DEBUG.txt"

Adaptive_Mod = f"./control_{job_id}/0/adaptive_mod.txt"
Adaptive_Freq = f"./control_{job_id}/0/check_freq.txt"

Faulty_injection = f"./control_{job_id}/0/FI.txt"

Injection_Plan = f"./control_{job_id}/0/plan.txt"

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TEMP_OUTPUT_DIR = "./tmp_llama_logs"

# 训练配置
NUM_TRAIN_SAMPLES = 4000 
SEED = 42
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 1e-5
MAX_SEQ_LENGTH = 1024 
MAX_STEPS = 1000

# 推理配置
INFERENCE_BATCH_SIZE = 16  # 显存允许的话可以开到 16 或 32
EVAL_SAMPLES = 480

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 2. 通用数据集加载与处理 (Alpaca)
# ==========================================

def load_general_dataset():
    logger.info("Loading General Dataset (Alpaca)...")
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
    except Exception as e:
        logger.error(f"Failed to load alpaca: {e}. Fallback to wikitext.")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        ds = ds.map(lambda x: {"instruction": "Continue the text:", "input": "", "output": x["text"]})
    return ds

def format_general_prompt(example):
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
# 3. Batch Inference: MMLU
# ==========================================

def evaluate_mmlu_batch(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting MMLU Inference (BATCH)\n" + "="*30)
    
    try:
        all_subjects = get_dataset_config_names("cais/mmlu")
        all_subjects.sort()
        # 选前10个subject
        selected = [s for s in all_subjects if s != "all"][:10]
        ds_list = []
        for s in selected:
            try: ds_list.append(load_dataset("cais/mmlu", s, split="test"))
            except: pass
        if not ds_list: return 0.0
        dataset = concatenate_datasets(ds_list)
    except: return 0.0

    model.eval()
    subset = dataset.shuffle(seed=SEED).select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    batch_size = INFERENCE_BATCH_SIZE
    
    for i in tqdm(range(0, len(subset), batch_size), desc="MMLU Batch"):
        batch_items = subset[i : i + batch_size]
        batch_prompts = []
        batch_truths = []
        
        for j in range(len(batch_items['question'])):
            q = batch_items['question'][j]
            c = batch_items['choices'][j]
            a_idx = batch_items['answer'][j]
            truth = ["A", "B", "C", "D"][a_idx]
            batch_truths.append(truth)
            
            opts = f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}"
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Analyze the following multiple-choice question and provide the correct answer. Output ONLY the corresponding letter (A, B, C, or D)."},
                {"role": "user", "content": f"{q}\n\n{opts}\n\nAnswer:"}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id, do_sample=False
            )
            
        generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for pred_text, truth in zip(decoded_preds, batch_truths):
            match = re.search(r'\b([A-D])\b', pred_text.upper())
            if match:
                pred_char = match.group(1)
            else:
                if len(pred_text) > 0 and pred_text[0].upper() in ['A', 'B', 'C', 'D']:
                    pred_char = pred_text[0].upper()
                else:
                    pred_char = "Z"
            
            if pred_char == truth:
                correct += 1
            total += 1
            
    acc = correct / total if total > 0 else 0
    print(f"MMLU Result: {acc:.2%}")
    return acc

# ==========================================
# 4. Batch Inference: SQuAD v2
# ==========================================

def normalize_text(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_text(a_gold).split()
    pred_toks = normalize_text(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0: return int(gold_toks == pred_toks)
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

def evaluate_squad_batch(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting SQuAD Inference (BATCH)\n" + "="*30)
    try: ds = load_dataset("squad_v1", split="validation")
    except: 
        try: ds = load_dataset("squad", split="validation")
        except: return 0.0

    subset = ds.shuffle(seed=SEED).select(range(min(num_samples, len(ds))))
    f1_scores = []
    
    batch_size = INFERENCE_BATCH_SIZE
    
    for i in tqdm(range(0, len(subset), batch_size), desc="SQuAD Batch"):
        batch_items = subset[i : i + batch_size]
        batch_prompts = []
        batch_golds = []
        
        for j in range(len(batch_items['context'])):
            ctx = batch_items['context'][j]
            qn = batch_items['question'][j]
            
            if 'answers' in batch_items:
                ans_list = batch_items['answers'][j]['text']
            else:
                ans_list = []
            if not ans_list: ans_list = [""]
            batch_golds.append(ans_list)
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Read the context and answer the question. Keep it concise."},
                {"role": "user", "content": f"Context: {ctx}\n\nQuestion: {qn}\n\nAnswer:"}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for pred, golds in zip(preds, batch_golds):
            pred_clean = pred.strip()
            score = max([compute_f1(g, pred_clean) for g in golds])
            f1_scores.append(score)
            
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    print(f"SQuAD F1: {avg_f1:.2%}")
    return avg_f1

# ==========================================
# 5. Batch Inference: WMT16
# ==========================================

def simple_bleu(ref, hyp):
    def tokenize(text): return [t.lower() for t in re.findall(r'\w+', text)]
    ref_t, hyp_t = tokenize(ref), tokenize(hyp)
    if not hyp_t: return 0.0
    common = collections.Counter(ref_t) & collections.Counter(hyp_t)
    p1 = sum(common.values()) / len(hyp_t)
    bp = math.exp(1 - len(ref_t) / len(hyp_t)) if len(hyp_t) < len(ref_t) else 1.0
    return p1 * bp

def evaluate_wmt16_batch(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting WMT16 Inference (BATCH)\n" + "="*30)
    try: ds = load_dataset("wmt16", "de-en", split="test")
    except: return 0.0
    
    subset = ds.shuffle(seed=SEED).select(range(min(num_samples, len(ds))))
    scores = []
    
    batch_size = INFERENCE_BATCH_SIZE
    for i in tqdm(range(0, len(subset), batch_size), desc="WMT Batch"):
        batch_items = subset[i : i + batch_size]
        batch_prompts = []
        batch_refs = []
        
        for trans in batch_items['translation']:
            src, ref = trans['de'], trans['en']
            batch_refs.append(ref)
            messages = [
                {"role": "system", "content": "You are a professional translator. Translate the following German text into English."},
                {"role": "user", "content": f"German: {src}\n\nEnglish Translation:"}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for pred, ref in zip(preds, batch_refs):
            scores.append(simple_bleu(ref, pred.strip()))
            
    avg_bleu = np.mean(scores) if scores else 0.0
    print(f"WMT16 BLEU: {avg_bleu:.4f}")
    return avg_bleu

# ==========================================
# 6. Batch Inference: GSM8K
# ==========================================

def extract_gsm_num(text):
    if "####" in text: return text.split("####")[-1].strip().replace(",", "")
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return nums[-1].replace(",", "") if nums else None

GSM8K_FEW_SHOT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Let's think step by step.
There are 15 trees originally.
Then workers plant some trees.
Now there are 21 trees.
So the workers planted 21 - 15 = 6 trees.
The answer is #### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Let's think step by step.
There are originally 3 cars.
2 more cars arrive.
3 + 2 = 5.
The answer is #### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Let's think step by step.
Leah had 32 chocolates.
Her sister had 42.
Total number of chocolates initially is 32 + 42 = 74.
They ate 35.
Remaining chocolates = 74 - 35 = 39.
The answer is #### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Let's think step by step.
Jason started with 20 lollipops.
He has 12 left.
So he gave Denny 20 - 12 = 8 lollipops.
The answer is #### 8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Let's think step by step.
Shawn started with 5 toys.
He got 2 toys from his mom.
He got 2 toys from his dad.
Total new toys = 2 + 2 = 4.
Total toys now = 5 + 4 = 9.
The answer is #### 9

Question: There were 9 computers in the server room. Five more computers were installed each day for 4 days. How many computers are now in the server room?
Let's think step by step.
Originally there were 9 computers.
5 computers were installed each day for 4 days.
Total computers installed = 5 * 4 = 20.
Total computers now = 9 + 20 = 29.
The answer is #### 29

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Let's think step by step.
Michael started with 58 balls.
On Tuesday he lost 23, so he had 58 - 23 = 35.
On Wednesday he lost 2 more, so he had 35 - 2 = 33.
The answer is #### 33

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Let's think step by step.
Olivia started with $23.
She bought 5 bagels.
Each bagel cost $3.
Total cost = 5 * 3 = $15.
Money left = 23 - 15 = 8.
The answer is #### 8
"""

def evaluate_gsm8k_batch(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting GSM8K Inference (8-Shot CoT)\n" + "="*30)
    try: ds = load_dataset("openai/gsm8k", "main", split="test")
    except: return 0.0
    
    subset = ds.shuffle(seed=SEED).select(range(min(num_samples, len(ds))))
    correct = 0
    total = 0
    
    batch_size = INFERENCE_BATCH_SIZE
    
    # 进度条
    for i in tqdm(range(0, len(subset), batch_size), desc="GSM8K Batch"):
        batch_items = subset[i : i + batch_size]
        batch_prompts = []
        batch_golds = []
        
        for j in range(len(batch_items['question'])):
            qn = batch_items['question'][j]
            ans = batch_items['answer'][j]
            batch_golds.append(extract_gsm_num(ans))
            
            # --- 关键修改：拼接 Few-Shot 示例 ---
            # 注意：对于 Llama 3.2，我们将示例放在 user content 里效果通常更好
            prompt_content = f"{GSM8K_FEW_SHOT}\n\nQuestion: {qn}\nLet's think step by step."
            
            messages = [
                {"role": "system", "content": "You are a math expert. Solve the problem step by step following the examples."},
                {"role": "user", "content": prompt_content}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, # 数学题需要写步骤，稍微给长一点
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=False,
                temperature=None,
                top_p=None
            )
            
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for pred, gold in zip(preds, batch_golds):
            pred_num = extract_gsm_num(pred)
            try:
                # 尝试转 float 比较，避免 120.0 != 120
                if pred_num and gold and abs(float(pred_num) - float(gold)) < 1e-6:
                    correct += 1
            except: 
                if str(pred_num) == str(gold): correct += 1
            total += 1
            
    acc = correct / total if total > 0 else 0
    print(f"GSM8K Acc: {acc:.2%}")
    return acc

# ==========================================
# 7. Batch Inference: XLSum (Llama 3.2 Optimized)
# ==========================================

def evaluate_xlsum_batch(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting XLSum Inference (BATCH)\n" + "="*30)
    try: 
        ds = load_dataset("csebuetnlp/xlsum", "english", split="test")
        rouge = evaluate.load("rouge")
    except: return 0.0
    
    subset = ds.shuffle(seed=SEED).select(range(min(num_samples, len(ds))))
    preds_all = []
    refs_all = []
    
    batch_size = INFERENCE_BATCH_SIZE
    
    for i in tqdm(range(0, len(subset), batch_size), desc="XLSum Batch"):
        batch_items = subset[i : i + batch_size]
        batch_prompts = []
        
        # Add references
        refs_all.extend(batch_items['summary'])
        
        for article in batch_items['text']:
            # Llama 3.2 Optimized Prompt: 禁止 "Here is a summary"
            messages = [
                {"role": "system", "content": "You are a professional editor. Summarize the news article below into a single, short headline. Output ONLY the summary text. Do not start with 'Here is a summary'."},
                {"role": "user", "content": f"Article: {article}\n\nHeadline:"}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
        # 截断以支持 Batch 处理 (Llama 上下文限制)
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            # 限制输出长度为 64，防止废话
            outputs = model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        # Post-process batch
        cleaned_preds = []
        for text in decoded:
            t = text.strip()
            # 简单清洗，切除第一行之后的内容
            if "\n" in t: t = t.split("\n")[0]
            # 去除可能的废话前缀
            if t.lower().startswith("here is"):
                 if ":" in t: t = t.split(":", 1)[1].strip()
            cleaned_preds.append(t)
        
        preds_all.extend(cleaned_preds)
        
    res = rouge.compute(predictions=preds_all, references=refs_all, use_stemmer=True)
    score = res['rougeL']
    print(f"XLSum ROUGE-L: {score:.4f}")
    return score

# ==========================================
# 8. 主程序
# ==========================================

def main():
    
    eval_mode = int(sys.argv[1])
    core_checker_mode = sys.argv[2]

    print(f"Loading Tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # --- [关键] 设置 Padding Side 为 Left ---
    # Batch 推理必须用 Left Padding，否则生成结果会错位
    tokenizer.padding_side = "left"
    
    # 保持原有的 Date Locked Template (仅 Llama 需要)
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

    # --- A. 加载数据 ---
    print("Loading training data (General)...")
    raw_dataset = load_general_dataset()
    actual_train_samples = min(NUM_TRAIN_SAMPLES, len(raw_dataset))
    train_subset = raw_dataset.shuffle(seed=SEED).select(range(actual_train_samples))

    # --- B. 预处理 (Switch to Right Padding for Training) ---
    print("Tokenizing training data...")
    # 训练时通常用 Right Padding，为了兼容 Trainer 和 DataCollator
    tokenizer.padding_side = "right" 
    
    tokenized_train_dataset = train_subset.map(
        lambda x: preprocess_general_function(x, tokenizer),
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=4
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
    training_args = TrainingArguments(
        output_dir=TEMP_OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
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

    # clean log files
    with open(f"./control_{job_id}/0/time/attn.txt", "w") as file: file.truncate(0)
    with open(f"./control_{job_id}/0/time/mlp.txt", "w") as file: file.truncate(0)
    with open(f"./control_{job_id}/0/time/preparation.txt", "w") as file: file.truncate(0)
    with open(f"./control_{job_id}/0/time/bgemm.txt", "w") as file: file.truncate(0)
    with open(f"./control_{job_id}/0/time/gemm.txt", "w") as file: file.truncate(0)
    with open(f"./control_{job_id}/0/time/update.txt", "w") as file: file.truncate(0)
    with open(f"./control_{job_id}/0/time/gemm_python.txt", "w") as file: file.truncate(0)
    with open(f"./control_{job_id}/0/time/bgemm_python.txt", "w") as file: file.truncate(0)
    with open(f"./control_{job_id}/0/time/training.txt", "w") as file: file.truncate(0)
   
    with open(f"./control_{job_id}/0/banned_smid.txt", "w") as file: file.truncate(0)
    with open(f"./control_{job_id}/0/SM_checking_results.txt", "w") as file: file.truncate(0)

    # get ground truth faulty smid
    faulty_smid = -1
    with open(Injection_Plan, 'r') as file:
        first_line = file.readline().strip()
        faulty_smid = int(first_line.split()[0])

    # --- E. 训练 ---
    print("Starting Training on General Dataset...")
    
    with open(cutlassFP, 'w') as file: file.write("f")
    with open(controlFP, 'w') as file: file.write("t")

    if eval_mode == 0:
        with open(DEBUG, 'w') as file: file.write("t")
        if core_checker_mode == "Baseline":
            with open(Enable_Core_Checker, 'w') as file: file.write("f")
            with open(Adaptive_Mod, 'w') as file: file.write("f")
            with open(Adaptive_Freq, 'w') as file: file.write(f"{1}")
            with open(Faulty_injection, 'w') as file: file.write("f")
        elif core_checker_mode == "Basic":
            with open(Enable_Core_Checker, 'w') as file: file.write("t")
            with open(Adaptive_Mod, 'w') as file: file.write("f")
            with open(Adaptive_Freq, 'w') as file: file.write(f"{1}")
            with open(Faulty_injection, 'w') as file: file.write("f")
        elif core_checker_mode == "1":
            with open(Enable_Core_Checker, 'w') as file: file.write("t")
            with open(Adaptive_Mod, 'w') as file: file.write("t")
            with open(Adaptive_Freq, 'w') as file: file.write(f"{1}")
            with open(Faulty_injection, 'w') as file: file.write("f")
        elif core_checker_mode == "2":
            with open(Enable_Core_Checker, 'w') as file: file.write("t")
            with open(Adaptive_Mod, 'w') as file: file.write("t")
            with open(Adaptive_Freq, 'w') as file: file.write(f"{10}")
            with open(Faulty_injection, 'w') as file: file.write("f")
    else:
        with open(DEBUG, 'w') as file: file.write("f")
        with open(Enable_Core_Checker, 'w') as file: file.write("t")
        with open(Adaptive_Mod, 'w') as file: file.write("t")
        with open(Adaptive_Freq, 'w') as file: file.write(f"{10}")
        with open(Faulty_injection, 'w') as file: file.write("t")

    trainer.train()

    with open(controlFP, 'w') as file: file.write("f") 

    log = trainer.state.log_history
    loss_history = [str(e.get("loss", "")) for e in log if "loss" in e]
    grad_norm = [str(e.get("grad_norm", "")) for e in log if "grad_norm" in e]
    training_time = [str(e.get("train_runtime", "")) for e in log if "train_runtime" in e]

    with open(f"./control_{job_id}/0/time/training.txt", 'a') as file: file.write(" ".join(training_time))

    if eval_mode == 0: return

    # --- F. Batch Inference Phase ---
    torch.cuda.empty_cache()
    gc.collect()
    model.config.use_cache = True
    
    # [关键] 切换回 Left Padding 进行 Batch 推理
    tokenizer.padding_side = "left"

    mmlu_acc = evaluate_mmlu_batch(model, tokenizer, EVAL_SAMPLES)
    squad_f1 = evaluate_squad_batch(model, tokenizer, EVAL_SAMPLES)
    wmt_score = evaluate_wmt16_batch(model, tokenizer, EVAL_SAMPLES)
    gsm_score = evaluate_gsm8k_batch(model, tokenizer, EVAL_SAMPLES)
    xlsum_score = evaluate_xlsum_batch(model, tokenizer, EVAL_SAMPLES)

    # Write Logs
    with open(logFP, "a") as file:
        file.write(f"{faulty_smid}\n")
        # file.write("\n--- Training Losses ---\n")
        file.write(" ".join(loss_history))
        file.write("\n")
        file.write(" ".join(grad_norm))
        # file.write("\n--- Inference Results ---\n")
        file.write(f"\n{mmlu_acc:.4f} ")
        file.write(f"{squad_f1:.4f} ")
        file.write(f"{wmt_score:.4f} ")
        file.write(f"{gsm_score:.4f} ")
        file.write(f"{xlsum_score:.4f}\n")
    
    with open(falutyStepFP, "r") as file:
        lines = file.readlines()
    lines.pop(0)
    with open(falutyStepFP, "w") as file:
        file.writelines(lines)

    print("Job Finished Successfully.")

if __name__ == "__main__":
    main()