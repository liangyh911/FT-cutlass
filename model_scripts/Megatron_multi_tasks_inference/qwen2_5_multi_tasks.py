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
# 1. Config
# ==========================================

MODEL_ID = "/projects/qwen-ckpts/Qwen2.5-7B-mcore-to-hf-cutlass"

job_id = os.getenv('SLURM_JOB_ID') or "local_dev" 
logFP = f"./control_{job_id}/eval_results.txt"

INFERENCE_BATCH_SIZE = 16  
EVAL_SAMPLES = 480
SEED = 42

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 2. Batch Inference Logic: MMLU
# ==========================================

def evaluate_mmlu_batch(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting MMLU Inference (BATCH)\n" + "="*30)
    
    # Load Data (Standard Method since fsspec is fixed)
    try:
        all_subjects = get_dataset_config_names("cais/mmlu")
        all_subjects.sort()
        # Select first 10 subjects to save loading time
        selected = [s for s in all_subjects if s != "all"][:10]
        ds_list = []
        for s in selected:
            try: 
                # Qwen2.5 建议使用 test split
                ds_list.append(load_dataset("cais/mmlu", s, split="test"))
            except Exception as e: 
                print(f"Skipping {s}: {e}")
                pass
        if not ds_list: return 0.0
        dataset = concatenate_datasets(ds_list)
    except Exception as e:
        print(f"MMLU Load Error: {e}")
        return 0.0

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
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        # Tokenize Batch (Left Padding for generation)
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5, 
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
                # Fallback: check first char
                clean_pred = pred_text.strip().upper()
                if len(clean_pred) > 0 and clean_pred[0] in ['A', 'B', 'C', 'D']:
                    pred_char = clean_pred[0]
                else:
                    pred_char = "Z"
            
            if pred_char == truth:
                correct += 1
            total += 1
            
    acc = correct / total if total > 0 else 0
    print(f"MMLU Result: {acc:.2%}")
    return acc

# ==========================================
# 3. Batch Inference Logic: SQuAD v2
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
    print("\n" + "="*30 + "\nStarting SQuAD Inference (Strict Mode)\n" + "="*30)
    # ... (加载数据代码不变) ...
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
            ans_list = batch_items['answers'][j]['text'] if 'answers' in batch_items else [""]
            if not ans_list: ans_list = [""]
            batch_golds.append(ans_list)
            
            # --- 修改：System Prompt 极度简化，要求简短 ---
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Output ONLY the exact answer string. Do not use full sentences."},
                {"role": "user", "content": f"Context: {ctx}\n\nQuestion: {qn}\n\nAnswer:"}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            # max_new_tokens 调小，防止它废话
            outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for pred, golds in zip(preds, batch_golds):
            # --- 修改：强力清洗 ---
            pred_clean = pred.strip().split('\n')[0] # 只取第一行
            # 去除常见废话
            for prefix in ["The answer is", "It is"]:
                if pred_clean.lower().startswith(prefix.lower()):
                    pred_clean = pred_clean[len(prefix):].strip()
            
            score = max([compute_f1(g, pred_clean) for g in golds])
            f1_scores.append(score)
            
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    print(f"SQuAD F1: {avg_f1:.2%}")
    return avg_f1
    
# ==========================================
# 4. Batch Inference Logic: WMT16
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
    print("\n" + "="*30 + "\nStarting WMT16 Inference (Fast Clean BLEU)\n" + "="*30)
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
                {"role": "system", "content": "Translate the German text into English. Output ONLY the English text."},
                {"role": "user", "content": src}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for pred, ref in zip(preds, batch_refs):
            # --- 暴力正则清洗 ---
            clean_pred = pred.strip()
            # 删掉所有 "Translation:" 或 "English:" 前缀
            clean_pred = re.sub(r'^(Translation|English|Answer|Output):\s*', '', clean_pred, flags=re.IGNORECASE)
            # 如果包含换行，只取第一行
            clean_pred = clean_pred.split('\n')[0].strip()
            
            scores.append(simple_bleu(ref, clean_pred))
            
    avg_bleu = np.mean(scores) if scores else 0.0
    print(f"WMT16 BLEU: {avg_bleu:.4f}")
    return avg_bleu

# ==========================================
# 5. Batch Inference Logic: GSM8K
# ==========================================

def extract_gsm_num_robust(text):
    # 1. 优先找 #### (官方标准)
    if "####" in text: 
        return text.split("####")[-1].strip().replace(",", "")
    
    # 2. 其次找 boxed (Qwen Math 常用格式)
    if "\\boxed{" in text:
        try:
            return text.split("\\boxed{")[1].split("}")[0].strip()
        except: pass

    # 3. 再次找 "The answer is"
    patterns = [
        r"(?:The|Therefore|Thus),? the answer is\s*([-+]?[\d,]+(?:\.\d+)?)",
        r"=\s*([-+]?[\d,]+(?:\.\d+)?)$" 
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "").rstrip(".")
            
    # 4. 保底：抓取最后一个数字
    # 注意：先去掉可能的无关后缀（比如 assistant 这种）
    for stop_word in ["<|im_end|>", "\n\nQuestion:", "\nassistant", "User:"]:
        if stop_word in text:
            text = text.split(stop_word)[0]
            
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(",", ""))
    if nums:
        return nums[-1]
    
    return None

# 标准 4-Shot CoT 模板 (精简版，防止占太多 Context)
GSM_FEW_SHOT_PREFIX = """Answer the following math problems.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: Let's think step by step. There are 15 trees originally. Then workers plant some trees. Now there are 21 trees. So the workers planted 21 - 15 = 6 trees. The answer is #### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is #### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Let's think step by step. Total number of chocolates initially is 32 + 42 = 74. They ate 35. Remaining chocolates = 74 - 35 = 39. The answer is #### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Let's think step by step. Jason started with 20 lollipops. He has 12 left. So he gave Denny 20 - 12 = 8 lollipops. The answer is #### 8
"""

def evaluate_gsm8k_batch(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting GSM8K Inference (4-Shot In-Context)\n" + "="*30)
    try: ds = load_dataset("openai/gsm8k", "main", split="test")
    except: return 0.0
    
    subset = ds.shuffle(seed=SEED).select(range(min(num_samples, len(ds))))
    correct = 0
    total = 0
    
    # 准备 Stop Tokens
    stop_token_ids = [tokenizer.eos_token_id]
    for t in ["<|im_end|>", "<|endoftext|>"]:
        try: stop_token_ids.append(tokenizer.convert_tokens_to_ids(t))
        except: pass

    batch_size = INFERENCE_BATCH_SIZE
    
    for i in tqdm(range(0, len(subset), batch_size), desc="GSM8K Batch"):
        batch_items = subset[i : i + batch_size]
        batch_prompts = []
        batch_golds = []
        
        for j in range(len(batch_items['question'])):
            qn = batch_items['question'][j]
            ans = batch_items['answer'][j]
            
            # 提取标准答案
            if "####" in ans:
                gold_num = ans.split("####")[-1].strip().replace(",", "")
            else:
                gold_num = re.findall(r"[-+]?\d*\.\d+|\d+", ans)[-1]
            batch_golds.append(gold_num)
            
            # --- Prompt 构建: 将例题直接拼接到用户输入中 ---
            # 这种方式对 Qwen 来说比 Chat History 更直观
            full_content = f"{GSM_FEW_SHOT_PREFIX}\nQuestion: {qn}\nAnswer:"
            
            messages = [
                {"role": "system", "content": "You are a math expert."},
                {"role": "user", "content": full_content}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                pad_token_id=tokenizer.eos_token_id, 
                eos_token_id=stop_token_ids,
                do_sample=False, 
                repetition_penalty=1.0, # 保持数学严谨性，不惩罚重复
            )
            
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for pred, gold in zip(preds, batch_golds):
            # 提取
            pred_num = extract_gsm_num_robust(pred)
            
            # --- 深度 Debug: 打印前 3 个错误的完整推理 ---
            # if total < 3 and str(pred_num) != str(gold):
            #     print(f"\n[DEBUG FAILURE] Gold: {gold} | Pred: {pred_num}")
            #     # 打印模型生成的完整推理过程（前 200 字符）
            #     reasoning_snippet = pred.replace("\n", " ")[:200]
            #     print(f"Reasoning: {reasoning_snippet}...")

            try:
                if pred_num and gold and abs(float(pred_num) - float(gold)) < 1e-6: correct += 1
            except: 
                if str(pred_num) == str(gold): correct += 1
            total += 1
            
    acc = correct / total if total > 0 else 0
    print(f"GSM8K Acc: {acc:.2%}")
    return acc

# ==========================================
# 6. Batch Inference Logic: XLSum
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
        refs_all.extend(batch_items['summary'])
        
        for article in batch_items['text']:
            # --- 修改：明确要求 One-sentence summary ---
            messages = [
                {"role": "system", "content": "You are a professional editor. Summarize the news article into a single, concise sentence of under 30 words."},
                {"role": "user", "content": f"Article: {article}\n\nSummary:"}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        
        with torch.no_grad():
            # 限制 max_new_tokens 为 40-50，强迫模型精简
            outputs = model.generate(**inputs, max_new_tokens=45, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for t in decoded:
            clean_t = t.strip().split("\n")[0] # 只要第一行
            preds_all.append(clean_t)
            
    res = rouge.compute(predictions=preds_all, references=refs_all, use_stemmer=True)
    print(f"XLSum ROUGE-L: {res['rougeL']:.4f}")
    return res['rougeL']

# ==========================================
# 7. Main Function
# ==========================================

def main():
    print(f"Loading Tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # [CRITICAL] Qwen2.5/Qwen3 often uses eos_token as pad_token if pad is None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # [CRITICAL] Set Padding Side for Batch Inference
    # For decoder-only models, we need left-padding during generation.
    tokenizer.padding_side = "left" 
    
    # --- Load Model ---
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )

    # --- Inference Phase ---
    model.config.use_cache = True
    
    # Run Batch Inference for all tasks
    mmlu_acc = evaluate_mmlu_batch(model, tokenizer, EVAL_SAMPLES)
    squad_f1 = evaluate_squad_batch(model, tokenizer, EVAL_SAMPLES)
    wmt_score = evaluate_wmt16_batch(model, tokenizer, EVAL_SAMPLES)
    gsm_score = evaluate_gsm8k_batch(model, tokenizer, EVAL_SAMPLES)
    xlsum_score = evaluate_xlsum_batch(model, tokenizer, EVAL_SAMPLES)
    
    # Write Logs
    with open(logFP, "a") as file:
        # file.write("\n--- Training Losses ---\n")
        # file.write(" ".join(loss_history))
        # file.write("\n")
        # file.write(" ".join(grad_norm))
        # file.write("\n--- Inference Results ---\n")
        file.write(f"{mmlu_acc:.4f} ")
        file.write(f"{squad_f1:.4f} ")
        file.write(f"{wmt_score:.4f} ")
        file.write(f"{gsm_score:.4f} ")
        file.write(f"{xlsum_score:.4f}\n")
    
    print("Job Finished Successfully.")

if __name__ == "__main__":
    main()