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
    print(f"Warning: Could not read faulty_step, default to 0. Error: {e}")

logFP = f"./control_{job_id}/0/output.log"
if not os.path.exists(os.path.dirname(logFP)):
    os.makedirs(os.path.dirname(logFP), exist_ok=True)

with open(logFP, "a") as file:
    file.write(f"Start Job: {faulty_step}\n")
    
# cutlass control file
controlFP = f"./control_{job_id}/0/perform.txt"
cutlassFP = f"./control_{job_id}/0/cutlass.txt"
SMChkFP = f"./control_{job_id}/0/split.txt"

# --- [修改点 1] 更改模型为 Qwen2.5-1.5B-Instruct ---
MODEL_ID = "Qwen/Qwen3-1.7B"
TEMP_OUTPUT_DIR = "./tmp_qwen_training_logs"

# 训练配置
NUM_TRAIN_SAMPLES = 4000 
SEED = 42
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 1e-5
MAX_SEQ_LENGTH = 1024 
MAX_STEPS = 1000 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 2. 通用数据集加载与处理 (Alpaca)
# ==========================================

def load_general_dataset():
    logger.info("Loading General Dataset (Alpaca)...")
    try:
        # 建议使用 cleaned 版本，效果更好，这里保持您的 tatsu-lab 
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
        
        # apply_chat_template 会自动处理 Qwen 的 ChatML 格式 (<|im_start|>...)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized_full = tokenizer(
            full_text, truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length", add_special_tokens=False
        )
        input_ids = tokenized_full["input_ids"]
        labels = copy.deepcopy(input_ids)
        
        # Mask Prompt
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
# 3. MMLU 评估逻辑
# ==========================================

def load_mmlu_eval_data(num_subjects=5):
    try:
        all_subjects = get_dataset_config_names("cais/mmlu")
    except Exception:
        all_subjects = ["abstract_algebra", "anatomy", "astronomy"]

    all_subjects = [s for s in all_subjects if s != "all"]
    all_subjects.sort()
    selected_subjects = all_subjects[:num_subjects]
    
    dataset_list = []
    for subject in tqdm(selected_subjects, desc=f"Loading MMLU subjects"):
        try:
            ds = load_dataset("cais/mmlu", subject, split="test") 
            dataset_list.append(ds)
        except Exception:
            pass
            
    if not dataset_list: return None
    full_ds = concatenate_datasets(dataset_list)
    return full_ds

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

def evaluate_mmlu(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting MMLU Inference\n" + "="*30)
    dataset = load_mmlu_eval_data(num_subjects=10)
    if dataset is None: return 0.0

    model.eval()
    correct_count = 0
    total_count = 0
    
    actual_samples = min(num_samples, len(dataset))
    subset = dataset.shuffle(seed=SEED).select(range(actual_samples))
    
    # --- DEBUG: Print first 3 outputs to see what's wrong ---
    debug_limit = 3 
    
    for i, item in enumerate(tqdm(subset, desc="MMLU")):
        messages, ground_truth_char = format_mmlu_prompt(item)
        if messages is None: continue 
        
        # Apply template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=10, # Give it space to chat if it wants
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=False
            )
        
        # Decode ONLY the new tokens
        new_token_id = outputs[0][inputs.input_ids.shape[1]:]
        raw_output = tokenizer.decode(new_token_id, skip_special_tokens=True).strip()
        
        # --- Logic Fix: Robust Parsing ---
        # 1. Try to find the first single letter A-D
        match = re.search(r'\b([A-D])\b', raw_output.upper())
        if match:
            prediction = match.group(1)
        else:
            # 2. If no single letter, check if string STARTS with A/B/C/D (e.g. "A is correct")
            if len(raw_output) > 0 and raw_output[0].upper() in ['A', 'B', 'C', 'D']:
                prediction = raw_output[0].upper()
            else:
                prediction = "NULL"

        if prediction == ground_truth_char:
            correct_count += 1
        total_count += 1
        
        # --- DEBUG PRINTING ---
        if i < debug_limit:
            print(f"\n[DEBUG Sample {i}]")
            print(f"GT: {ground_truth_char}")
            print(f"Raw Output: '{raw_output}'")
            print(f"Parsed Pred: {prediction}")
            print("-" * 20)
            
    acc = correct_count / total_count if total_count > 0 else 0
    print(f"MMLU Result: {acc:.2%} ({correct_count}/{total_count})")
    return acc
    
# ==========================================
# 4. Squad v2 评估逻辑
# ==========================================

def normalize_text(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_text(a_gold).split()
    pred_toks = normalize_text(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_squad_v2(model, tokenizer, num_samples=200):
    print("\n" + "="*30 + "\nStarting Squad Inference (Optimized)\n" + "="*30)
    
    dataset = None
    try:
        dataset = load_dataset("squad_v1", split="validation")
    except Exception:
        try:
            dataset = load_dataset("squad", split="validation")
        except Exception:
            return 0.0
    if dataset is None: return 0.0

    model.eval()
    f1_scores = []
    
    actual_samples = min(num_samples, len(dataset))
    subset = dataset.shuffle(seed=SEED).select(range(actual_samples))
    
    for item in tqdm(subset, desc="Squad"):
        context = item['context']
        question = item['question']
        
        if 'answers' in item:
            gold_texts = item['answers']['text']
        else:
            gold_texts = []
        if len(gold_texts) == 0: gold_texts = [""] 

        # --- Prompt Optimization ---
        # 针对 SQuAD F1，我们强制模型进行抽取式回答，禁止废话。
        # 加入 /no_think 适配 Qwen3
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant. /no_think Read the context and answer the question. Output ONLY the answer text. Do not write complete sentences."
            },
            {
                "role": "user", 
                "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            }
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=32, # SQuAD 答案通常很短
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=False
            )
            
        new_token_id = outputs[0][inputs.input_ids.shape[1]:]
        prediction = tokenizer.decode(new_token_id, skip_special_tokens=True).strip()
        
        if not gold_texts:
            item_f1 = 0
        else:
            item_f1 = max([compute_f1(a, prediction) for a in gold_texts])
        
        f1_scores.append(item_f1)
        
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    print(f"Squad Result | F1: {avg_f1:.2%}")
    return avg_f1

# ==========================================
# 5. WMT16 Translation Inference Logic
# ==========================================

def simple_bleu(ref_text, hyp_text):
    def tokenize(text):
        return [t.lower() for t in re.findall(r'\w+', text)]
    ref_tokens = tokenize(ref_text)
    hyp_tokens = tokenize(hyp_text)
    if len(hyp_tokens) == 0: return 0.0
    common = collections.Counter(ref_tokens) & collections.Counter(hyp_tokens)
    p1 = sum(common.values()) / len(hyp_tokens)
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    else: bp = 1.0
    return p1 * bp

def evaluate_wmt16(model, tokenizer, num_samples=200):
    print("\n" + "="*30 + "\nRunning WMT16 (DE-EN) Inference\n" + "="*30)
    try:
        dataset = load_dataset("wmt16", "de-en", split="test")
    except Exception: return 0.0

    model.eval()
    bleu_scores = []
    
    actual_samples = min(num_samples, len(dataset))
    subset = dataset.shuffle(seed=SEED).select(range(actual_samples))
    
    for item in tqdm(subset, desc="Translation"):
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
                max_new_tokens=128, 
                do_sample=False
            )
            
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        score = simple_bleu(ref_text, gen_text)
        bleu_scores.append(score)
        
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    print(f"WMT16 BLEU (Approx): {avg_bleu:.4f}")
    return avg_bleu

# ==========================================
# 6. GSM8K Math Inference Logic
# ==========================================

def extract_gsm8k_answer(text):
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers: return numbers[-1].replace(",", "")
    return None

def evaluate_gsm8k(model, tokenizer, num_samples=200):
    print("\n" + "="*30 + "\nRunning GSM8K (Math) Inference\n" + "="*30)
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="test")
    except Exception: return 0.0

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
                max_new_tokens=256, 
                do_sample=False
            )
            
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred_answer = extract_gsm8k_answer(gen_text)
        
        try:
            if pred_answer and gold_answer and float(pred_answer) == float(gold_answer):
                correct += 1
        except ValueError:
            if pred_answer == gold_answer: correct += 1
        total += 1
        
    acc = correct / total if total > 0 else 0
    print(f"GSM8K Accuracy: {acc:.4%}")
    return acc

# ==========================================
# 7. XLSum 评估函数
# ==========================================
def evaluate_xlsum(model, tokenizer, num_samples=200):
    print("\n" + "="*30 + "\n[Qwen 3] Starting XLSum Inference\n" + "="*30)
    
    # 1. 加载数据集
    try:
        dataset = load_dataset("csebuetnlp/xlsum", "english", split="test")
    except Exception as e:
        print(f"XLSum load failed: {e}")
        return 0.0

    # 2. 加载 Rouge
    try:
        rouge = evaluate.load("rouge")
    except Exception:
        return 0.0

    model.eval()
    
    predictions = []
    references = []
    
    actual_samples = min(num_samples, len(dataset))
    subset = dataset.shuffle(seed=42).select(range(actual_samples))
    
    for item in tqdm(subset, desc="Summarizing (Qwen)"):
        article = item['text']
        summary = item['summary']
        
        # --- Qwen 3 Prompt ---
        # 1. 角色设定为 News Editor
        # 2. 加入 /no_think 防止输出思考过程 (针对 Qwen3)
        messages = [
            {
                "role": "system", 
                "content": "You are a professional news editor. /no_think Summarize the news below into a one-line headline."
            },
            {
                "role": "user", 
                "content": article
            }
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Qwen 支持较长上下文，稍微放宽到 2048
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        predictions.append(gen_text)
        references.append(summary)
        
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    rouge_l = results["rougeL"]
    
    print(f"XLSum Result | ROUGE-L: {rouge_l:.4f}")
    return rouge_l

# ==========================================
# 7. 主程序
# ==========================================

def main():
    # --- [修改点 2] Tokenizer 初始化与配置 ---
    print(f"Loading Tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Qwen 的 Pad Token 处理 (Qwen 的 EOS 是 <|im_end|>，通常没有默认 pad_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # --- [重要] 移除了原有的 LLAMA_3_TIME_LOCKED_TEMPLATE ---
    # Qwen 模型自带了 ChatML 格式的模板，不要强制覆盖为 Llama 格式
    
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
        attn_implementation="eager", # Qwen 也可以用 flash_attention_2
        use_cache=False 
    )

    # --- D. Trainer 配置 ---
    training_args = TrainingArguments(
        output_dir=TEMP_OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS, # 1000 steps
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
    
    with open(cutlassFP, 'w') as file: file.write("f")
    with open(SMChkFP, 'w') as file: file.write("f")
    with open(controlFP, 'w') as file: file.write("t") 

    trainer.train()

    with open(controlFP, 'w') as file: file.write("f") 

    log = trainer.state.log_history
    loss_history = [str(e.get("loss", "")) for e in log if "loss" in e]
    
    # --- F. 推理 ---
    torch.cuda.empty_cache()
    gc.collect()
    model.config.use_cache = True

    # 统一推理样本数
    EVAL_SAMPLES = 500

    # Task 1: Inference: MMLU
    # mmlu_acc = evaluate_mmlu(model, tokenizer, num_samples=EVAL_SAMPLES)

    # Task 2: Inference: Squad
    squad_f1 = evaluate_squad_v2(model, tokenizer, num_samples=EVAL_SAMPLES)

    # Task 3: WMT16
    wmt_score = evaluate_wmt16(model, tokenizer, num_samples=EVAL_SAMPLES)

    # Task 4: GSM8K
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

    print("Job Finished Successfully.")

if __name__ == "__main__":
    main()