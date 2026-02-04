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
# 1. Configuration
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
    file.write(f"Start Job: {faulty_step}\n")
    
# cutlass control file
controlFP = f"./control_{job_id}/0/perform.txt"
cutlassFP = f"./control_{job_id}/0/cutlass.txt"
SMChkFP = f"./control_{job_id}/0/split.txt"

# Model Config
MODEL_ID = "Qwen/Qwen3-1.7B"
TEMP_OUTPUT_DIR = "./tmp_qwen_batch_logs"

# Training Config
NUM_TRAIN_SAMPLES = 4000 
SEED = 42
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 1e-5
MAX_SEQ_LENGTH = 1024 
MAX_STEPS = 1000

# Inference Config
INFERENCE_BATCH_SIZE = 16  # [Key Change] Batch size for inference
EVAL_SAMPLES = 480

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 2. General Dataset Loading (Alpaca)
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
# 3. Batch Inference Logic: MMLU
# ==========================================

def evaluate_mmlu_batch(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting MMLU Inference (BATCH)\n" + "="*30)
    
    # Load Data
    try:
        all_subjects = get_dataset_config_names("cais/mmlu")
        all_subjects.sort()
        # Select first 10 subjects to save loading time
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
    
    # Batch Loop
    for i in tqdm(range(0, len(subset), batch_size), desc="MMLU Batch"):
        batch_items = subset[i : i + batch_size]
        batch_prompts = []
        batch_truths = []
        
        # Prepare Batch Data
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
            # Apply template but do not tokenize yet
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        # Tokenize Batch (Padding is handled here)
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=10, 
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=False
            )
            
        # Decode only the generated part
        generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for pred_text, truth in zip(decoded_preds, batch_truths):
            # Parsing Logic
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
# 4. Batch Inference Logic: SQuAD v2
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
        batch_golds = [] # List of list of strings
        
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
                {"role": "system", "content": "You are a helpful assistant. /no_think Read the context and answer the question. Output ONLY the answer text. Do not write complete sentences."},
                {"role": "user", "content": f"Context: {ctx}\n\nQuestion: {qn}\n\nAnswer:"}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id, do_sample=False
            )
            
        generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for pred, golds in zip(preds, batch_golds):
            pred_clean = pred.strip()
            # Calculate F1 vs all valid answers
            score = max([compute_f1(g, pred_clean) for g in golds])
            f1_scores.append(score)
            
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    print(f"SQuAD F1: {avg_f1:.2%}")
    return avg_f1

# ==========================================
# 5. Batch Inference Logic: WMT16
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
# 6. Batch Inference Logic: GSM8K
# ==========================================

def extract_gsm_num(text):
    if "####" in text: return text.split("####")[-1].strip().replace(",", "")
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return nums[-1].replace(",", "") if nums else None

def evaluate_gsm8k_batch(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting GSM8K Inference (BATCH)\n" + "="*30)
    try: ds = load_dataset("openai/gsm8k", "main", split="test")
    except: return 0.0
    
    subset = ds.shuffle(seed=SEED).select(range(min(num_samples, len(ds))))
    correct = 0
    total = 0
    
    batch_size = INFERENCE_BATCH_SIZE
    for i in tqdm(range(0, len(subset), batch_size), desc="GSM8K Batch"):
        batch_items = subset[i : i + batch_size]
        batch_prompts = []
        batch_golds = []
        
        for j in range(len(batch_items['question'])):
            qn = batch_items['question'][j]
            ans = batch_items['answer'][j]
            batch_golds.append(extract_gsm_num(ans))
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant specialized in math. Solve the following problem step by step, and end your response with 'The answer is #### <number>'."},
                {"role": "user", "content": f"Question: {qn}"}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for pred, gold in zip(preds, batch_golds):
            pred_num = extract_gsm_num(pred)
            try:
                if pred_num and gold and float(pred_num) == float(gold): correct += 1
            except: 
                if pred_num == gold: correct += 1
            total += 1
            
    acc = correct / total if total > 0 else 0
    print(f"GSM8K Acc: {acc:.2%}")
    return acc

# ==========================================
# 7. Batch Inference Logic: XLSum
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
        
        # Add references immediately
        refs_all.extend(batch_items['summary'])
        
        for article in batch_items['text']:
            messages = [
                {"role": "system", "content": "You are a professional news editor. /no_think Summarize the news below into a one-line headline."},
                {"role": "user", "content": article}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
        # XLSum inputs can be very long, truncate them for safe batching
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for p in preds:
            preds_all.append(p.strip())
            
    res = rouge.compute(predictions=preds_all, references=refs_all, use_stemmer=True)
    score = res['rougeL']
    print(f"XLSum ROUGE-L: {score:.4f}")
    return score

# ==========================================
# 8. Main Function
# ==========================================

def main():
    print(f"Loading Tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # --- [CRITICAL] Set Padding Side for Batch Inference ---
    # For decoder-only models, we need left-padding during generation.
    tokenizer.padding_side = "left" 
    
    # --- A. Load General Training Data ---
    print("Loading training data (General)...")
    raw_dataset = load_general_dataset()
    actual_train_samples = min(NUM_TRAIN_SAMPLES, len(raw_dataset))
    train_subset = raw_dataset.shuffle(seed=SEED).select(range(actual_train_samples))

    # --- B. Preprocess (Switch to Right Padding for Training) ---
    print("Tokenizing training data...")
    tokenizer.padding_side = "right" # Training usually uses right padding (handled by collator typically, but explicit is safer)
    
    tokenized_train_dataset = train_subset.map(
        lambda x: preprocess_general_function(x, tokenizer),
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=4
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=None, padding=False
    )

    # --- C. Load Model ---
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager", 
        use_cache=False 
    )

    # --- D. Trainer Config ---
    training_args = TrainingArguments(
        output_dir=TEMP_OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        warmup_ratio=0.1,
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

    # --- E. Training ---
    print("Starting Training...")
    with open(cutlassFP, 'w') as file: file.write("f")
    with open(SMChkFP, 'w') as file: file.write("f")
    with open(controlFP, 'w') as file: file.write("t") 

    trainer.train()

    with open(controlFP, 'w') as file: file.write("f") 

    log = trainer.state.log_history
    loss_history = [str(e.get("loss", "")) for e in log if "loss" in e]
    
    # --- F. Inference Phase ---
    torch.cuda.empty_cache()
    gc.collect()
    model.config.use_cache = True
    
    # [CRITICAL] Switch back to Left Padding for Inference
    tokenizer.padding_side = "left"

    # Run Batch Inference for all tasks
    mmlu_acc = evaluate_mmlu_batch(model, tokenizer, EVAL_SAMPLES)
    squad_f1 = evaluate_squad_batch(model, tokenizer, EVAL_SAMPLES)
    wmt_score = evaluate_wmt16_batch(model, tokenizer, EVAL_SAMPLES)
    gsm_score = evaluate_gsm8k_batch(model, tokenizer, EVAL_SAMPLES)
    xlsum_score = evaluate_xlsum_batch(model, tokenizer, EVAL_SAMPLES)

    # Write Logs
    with open(logFP, "a") as file:
        file.write("\n--- Training Losses ---\n")
        file.write(" ".join(loss_history))
        file.write("\n--- Inference Results ---\n")
        file.write(f"MMLU Acc: {mmlu_acc:.4f}\n")
        file.write(f"SQuAD F1: {squad_f1:.4f}\n")
        file.write(f"WMT16 BLEU: {wmt_score:.4f}\n")
        file.write(f"GSM8K Acc: {gsm_score:.4f}\n")
        file.write(f"XLSum ROUGE: {xlsum_score:.4f}\n")

    print("Job Finished Successfully.")

if __name__ == "__main__":
    main()