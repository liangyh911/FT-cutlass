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
    AutoModelForCausalLM
)
from tqdm import tqdm

# ==========================================
# 1. Config
# ==========================================

# Job id
job_id = os.getenv('SLURM_JOB_ID') or "local_dev" 
logFP = f"./control_{job_id}/eval_results.txt"

MODEL_ID = "/projects/llama3-ckpts/Llama-3.1-8B-mcore-to-hf-cutlass"
# MODEL_ID = "/projects/llama3-ckpts/Meta-Llama-3.1-8B"
# MODEL_ID = "/projects/llama3-ckpts/Meta-Llama-3.1-8B-hf-from-mg"

INFERENCE_BATCH_SIZE = 16  
EVAL_SAMPLES = 480
SEED = 42

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# [Critical] Llama 3 Stop Tokens
# 128001: <|end_of_text|> (Text completion end)
# 128009: <|eot_id|> (Chat turn end)
STOP_TOKEN_IDS = [128001, 128009]

# ==========================================
# 2. Batch Inference: MMLU (PPL - Text Only)
# ==========================================
# 保持 54% 的 PPL 方法，不做改动

def evaluate_mmlu_batch(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting MMLU Inference (PPL - Text)\n" + "="*30)
    try:
        try:
            all_subjects = get_dataset_config_names("cais/mmlu")
            all_subjects.sort()
            selected = [s for s in all_subjects if s != "all"][:10]
        except:
            selected = ["abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", 
                        "college_biology", "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine"]
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
    
    # Llama 3 Tokenizer " A" vs "A" check
    # 这里的 ID 需要根据实际 Tokenizer 确认，通常 direct encoding 即可
    cand_ids = [
        tokenizer.encode("A", add_special_tokens=False)[0],
        tokenizer.encode("B", add_special_tokens=False)[0],
        tokenizer.encode("C", add_special_tokens=False)[0],
        tokenizer.encode("D", add_special_tokens=False)[0]
    ]
    
    batch_size = INFERENCE_BATCH_SIZE
    
    for i in tqdm(range(0, len(subset), batch_size), desc="MMLU Batch"):
        batch_items = subset[i : i + batch_size]
        batch_prompts = []
        batch_truths = []
        
        for j in range(len(batch_items['question'])):
            q = batch_items['question'][j]
            c = batch_items['choices'][j]
            a_idx = batch_items['answer'][j]
            batch_truths.append(a_idx)
            opts = f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}"
            prompt = f"Question: {q}\n{opts}\nAnswer:"
            batch_prompts.append(prompt)
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            next_token_logits = outputs.logits[:, -1, :] 
            cand_logits = next_token_logits[:, cand_ids]
            predictions = torch.argmax(cand_logits, dim=-1)
        
        for pred_idx, truth_idx in zip(predictions.cpu().numpy(), batch_truths):
            if pred_idx == truth_idx: correct += 1
            total += 1
            
    acc = correct / total if total > 0 else 0
    print(f"MMLU Result: {acc:.2%}")
    return acc

# ==========================================
# 3. Batch Inference: SQuAD (Base Model Optimized)
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
    print("\n" + "="*30 + "\nStarting SQuAD Inference (Base Model Style)\n" + "="*30)
    try: ds = load_dataset("squad_v1", split="validation")
    except: 
        try: ds = load_dataset("squad", split="validation")
        except: return 0.0

    subset = ds.shuffle(seed=SEED).select(range(min(num_samples, len(ds))))
    f1_scores = []
    batch_size = INFERENCE_BATCH_SIZE
    
    # [关键] 准备一个简短的 One-Shot 示例
    # 这告诉 Base 模型：“不要废话，直接给我答案短语”
    # 如果 Context 长度允许，拼在前面；如果不够长，至少 Prompt 格式要对。
    
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
            
            # [核心修改] 纯文本补全 Prompt
            # 这种格式对 Base 模型最友好，它会自然地补全 Answer 后面的内容
            prompt = f"Passage: {ctx}\nQuestion: {qn}\nAnswer:"
            
            # 手动 Tokenize，不加 Special Tokens (如 BOS) 避免重复
            batch_prompts.append(prompt)
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, # 答案通常很短，限制长度防止废话
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=False
            )
        
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for pred, golds in zip(preds, batch_golds):
            # 后处理：Base 模型可能会换行继续生成 Question，必须截断
            pred_clean = pred.strip().split('\n')[0]
            
            # 进一步清洗：去掉可能的 "The answer is" 前缀
            # 虽然 Base 模型不太会生成这个，但防一手
            if "answer is" in pred_clean.lower():
                pred_clean = pred_clean.split("answer is")[-1].strip()
                
            f1_scores.append(max([compute_f1(g, pred_clean) for g in golds]))
            
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    print(f"SQuAD F1: {avg_f1:.2%}")
    return avg_f1

# ==========================================
# 4. Batch Inference: WMT16 (Text Only)
# ==========================================
# 保持 0.66 的纯文本方法

def simple_bleu(ref, hyp):
    def tokenize(text): return [t.lower() for t in re.findall(r'\w+', text)]
    ref_t, hyp_t = tokenize(ref), tokenize(hyp)
    if not hyp_t: return 0.0
    common = collections.Counter(ref_t) & collections.Counter(hyp_t)
    p1 = sum(common.values()) / len(hyp_t)
    bp = math.exp(1 - len(ref_t) / len(hyp_t)) if len(hyp_t) < len(ref_t) else 1.0
    return p1 * bp

def evaluate_wmt16_batch(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting WMT16 Inference (Text)\n" + "="*30)
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
            prompt = f"Translate the following German sentence to English.\nGerman: {src}\nEnglish:"
            batch_prompts.append(prompt)
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=False
            )
            
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for pred, ref in zip(preds, batch_refs):
            clean = pred.strip().split('\n')[0]
            scores.append(simple_bleu(ref, clean))
            
    avg_bleu = np.mean(scores) if scores else 0.0
    print(f"WMT16 BLEU: {avg_bleu:.4f}")
    return avg_bleu

# ==========================================
# 5. Batch Inference: GSM8K (Chat Zero-Shot)
# ==========================================

GSM8K_8_SHOT_PROMPT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then workers plant some trees. Now there are 21 trees. So the workers planted 21 - 15 = 6 trees. The answer is #### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is #### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Total number of chocolates initially is 32 + 42 = 74. They ate 35. Remaining chocolates = 74 - 35 = 39. The answer is #### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. He has 12 left. So he gave Denny 20 - 12 = 8 lollipops. The answer is #### 8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. He got 2 toys from his mom. He got 2 toys from his dad. Total new toys = 2 + 2 = 4. Total toys now = 5 + 4 = 9. The answer is #### 9

Question: There were 9 computers in the server room. Five more computers were installed each day for 4 days. How many computers are now in the server room?
Answer: Originally there were 9 computers. 5 computers were installed each day for 4 days. Total computers installed = 5 * 4 = 20. Total computers now = 9 + 20 = 29. The answer is #### 29

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Michael started with 58 balls. On Tuesday he lost 23, so he had 58 - 23 = 35. On Wednesday he lost 2 more, so he had 35 - 2 = 33. The answer is #### 33

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia started with $23. She bought 5 bagels. Each bagel cost $3. Total cost = 5 * 3 = $15. Money left = 23 - 15 = 8. The answer is #### 8

"""

def extract_gsm_num_robust(text):
    if "####" in text: return text.split("####")[-1].strip().replace(",", "")
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(",", ""))
    return nums[-1] if nums else None

def evaluate_gsm8k_batch(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting GSM8K Inference (8-Shot Base)\n" + "="*30)
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
            
            if "####" in ans:
                gold_num = ans.split("####")[-1].strip().replace(",", "")
            else:
                gold_num = re.findall(r"[-+]?\d*\.\d+|\d+", ans)[-1]
            batch_golds.append(gold_num)
            
            # [关键] 构造 Prompt: 8个例题 + 当前问题
            # 不加 "Let's think step by step"，让模型自然模仿上面的例题风格
            prompt = f"{GSM8K_8_SHOT_PROMPT}Question: {qn}\nAnswer:"
            batch_prompts.append(prompt)
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, # 足够写完步骤
                pad_token_id=tokenizer.eos_token_id,
                # [关键] 不设置 eos_token_id 列表，因为 Base 模型可能不会生成 EOT
                # 我们完全依赖下面的字符串截断
                do_sample=False
            )
            
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for pred, gold in zip(preds, batch_golds):
            # [关键] 后处理截断：Base 模型会一直生成下去，必须手动切断
            # 我们找第一个 "\nQuestion:"，如果没有就找 "\n\n"
            clean_pred = pred
            for stop_str in ["\nQuestion:", "\n\nQuestion", "Question:"]:
                if stop_str in clean_pred:
                    clean_pred = clean_pred.split(stop_str)[0]
                    break
            
            pred_num = extract_gsm_num_robust(clean_pred)
            
            # Debug
            if total < 2 and str(pred_num) != str(gold):
                print(f"[Diff] Gold: {gold} | Pred: {pred_num}")
                # 打印前 100 个字符看是否格式对了
                print(f"       Output: {clean_pred.replace(chr(10), ' ')[:100]}...")

            try:
                if pred_num and gold and abs(float(pred_num) - float(gold)) < 1e-6: correct += 1
            except: 
                if str(pred_num) == str(gold): correct += 1
            total += 1
            
    acc = correct / total if total > 0 else 0
    print(f"GSM8K Acc: {acc:.2%}")
    return acc

# ==========================================
# 6. Batch Inference: XLSum (Text)
# ==========================================
# 保持纯文本方法

def evaluate_xlsum_batch(model, tokenizer, num_samples=500):
    print("\n" + "="*30 + "\nStarting XLSum Inference (Text)\n" + "="*30)
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
            prompt = f"Article: {article}\n\nSummary:"
            batch_prompts.append(prompt)
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for t in decoded:
            clean = t.strip().split("\n")[0]
            preds_all.append(clean)
            
    res = rouge.compute(predictions=preds_all, references=refs_all, use_stemmer=True)
    score = res['rougeL']
    print(f"XLSum ROUGE-L: {score:.4f}")
    return score

# ==========================================
# 7. Main Function
# ==========================================

def main():
    print(f"Loading Tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    # [关键] 恢复您自定义的修复版模板
    # 这能解决日期报错，同时配合 tokenizer.apply_chat_template 使用
    LLAMA_3_FIXED_TEMPLATE = (
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
    tokenizer.chat_template = LLAMA_3_FIXED_TEMPLATE
    
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = True
    
    # Run Inference
    mmlu_acc = evaluate_mmlu_batch(model, tokenizer, EVAL_SAMPLES)
    squad_f1 = evaluate_squad_batch(model, tokenizer, EVAL_SAMPLES)
    wmt_score = evaluate_wmt16_batch(model, tokenizer, EVAL_SAMPLES)
    gsm_score = evaluate_gsm8k_batch(model, tokenizer, EVAL_SAMPLES)
    xlsum_score = evaluate_xlsum_batch(model, tokenizer, EVAL_SAMPLES)

    # # Write Logs
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