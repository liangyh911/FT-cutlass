import os
import math
import gc
 
import torch
import string
import collections
import copy
from datasets import load_dataset
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
# 1. config
# ==========================================

# Job id
job_id = os.getenv('SLURM_JOB_ID')

# Faulty step select corresponding checkpoint
falutyStepFP = f"/home/yuhangl/control_{job_id}/faulty_step.txt"
with open(falutyStepFP, 'r') as file:
    faulty_step = int(file.readline())
faulty_epoch = math.floor(faulty_step / 63)
local_faulty_step = faulty_step % 63

logFP = f"/home/yuhangl/control_{job_id}/output.log"
with open(logFP, "a") as file:
    file.write(f"{faulty_step}, ({faulty_epoch}, {local_faulty_step})\n")

# cutlass control file
controlFP = f"/home/yuhangl/control_{job_id}/perform.txt"
cutlassFP = f"/home/yuhangl/control_{job_id}/cutlass.txt"
SMChkFP = f"/home/yuhangl/control_{job_id}/split.txt"

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "./llama3-1b-squad-trainer-2k"

NUM_TRAIN_SAMPLES = 4000
SEED = 42

BATCH_SIZE = 32
GRAD_ACCUMULATION = 2
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 256

# ==========================================
# 2. Data Processing
# ==========================================

def create_conversation_format(example):
    context = example['context']
    question = example['question']
    if len(example['answers']['text']) > 0:
        answer = example['answers']['text'][0]
    else:
        answer = "unanswerable"
        
    # complete conversion format
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Read the context and answer the question. If the question cannot be answered from the context, strictly reply with 'unanswerable'."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        {"role": "assistant", "content": answer}
    ]
    return messages

def preprocess_function(examples, tokenizer):
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for i in range(len(examples['context'])):
        # 1. construct conversation (User + Assistant)
        single_example = {
            'context': examples['context'][i], 
            'question': examples['question'][i], 
            'answers': examples['answers'][i]
        }
        messages = create_conversation_format(single_example)
        
        # 2. Tokenize
        # apply_chat_template will add tokens like <|begin_of_text|>, <|eot_id|> 
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized_full = tokenizer(
            full_text, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH, 
            # padding=False,
            padding="max_length",
            add_special_tokens=False 
        )
        
        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        labels = copy.deepcopy(input_ids)
        
        # 3. compute Prompt's length (for Masking)
        prompt_messages = messages[:-1] # 去掉最后一个 assistant 的回复
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        tokenized_prompt = tokenizer(
            prompt_text, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH, 
            padding=False,
            # padding="max_length", 
            add_special_tokens=False
        )
        prompt_len = len(tokenized_prompt["input_ids"])
        
        # 4. Masking: set label of Prompt to -100 (ignore Loss)
        # safty check for token differences
        if prompt_len < len(labels):
            for j in range(prompt_len):
                labels[j] = -100
        else:
            # only prompt，ignore it
            labels = [-100] * len(labels)

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
        
    return model_inputs

# ==========================================
# 3. evaluation function
# ==========================================
def normalize_text(s):
    def remove_articles(text):
        return " ".join([t for t in text.split() if t not in {"a", "an", "the"}])
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def run_evaluation(model, tokenizer, eval_dataset, num_samples=200):
    with open(controlFP, 'w') as file:
        file.truncate(0)
        file.write("f")

    print(f"\n[Evaluation] Starting evaluation on {num_samples} random samples...")
    model.eval()
    
    # clean cache
    torch.cuda.empty_cache()
    
    f1_scores = []
    subset = eval_dataset.shuffle(seed=SEED).select(range(min(num_samples, len(eval_dataset))))
    
    original_use_cache = model.config.use_cache
    model.config.use_cache = False
    
    for item in tqdm(subset, desc="Evaluating"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Read the context and answer the question. If the question cannot be answered from the context, strictly reply with 'unanswerable'."},
            {"role": "user", "content": f"Context: {item['context']}\n\nQuestion: {item['question']}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=50, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=False
            )
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        ground_truths = item['answers']['text']
        
        score = 0.0
        gen_norm = normalize_text(generated_text)
        if not ground_truths:
            score = 1.0 if "unanswerable" in gen_norm else 0.0
        else:
            if "unanswerable" in gen_norm:
                score = 0.0
            else:
                score = max([compute_f1(generated_text, gt) for gt in ground_truths])
        f1_scores.append(score)
        
    avg_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n[Result] Epoch F1 Score: {avg_f1:.4f}")

    with open(logFP, "a") as f:
        f.write(f"{avg_f1:.4f} ")

    with open(controlFP, 'w') as file:
        file.truncate(0)
        file.write("t")
    
    # back to training mode
    model.config.use_cache = original_use_cache
    model.train() 
    
    return avg_f1

# ==========================================
# 4. Custom Callback
# ==========================================

class SquadEpochEvalCallback(TrainerCallback):
    """
    在每个 Epoch 结束时触发 SQuAD 评估
    """
    def __init__(self, tokenizer, eval_dataset, num_samples=200):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Trainer 会在每个 epoch 结束时自动调用此方法
        """
        print(f"\n\n*** Epoch {state.epoch:.1f} Finished. Running SQuAD Evaluation... ***")
        
        model = kwargs['model']
        
        # evaluate
        f1 = run_evaluation(
            model=model,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            num_samples=self.num_samples
        )
        
        # clean cache
        torch.cuda.empty_cache()
        gc.collect()

# ==========================================
# 5. main function
# ==========================================

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 1. load sample data
    dataset = load_dataset("squad_v2")
    print(f"Randomly selecting {NUM_TRAIN_SAMPLES} samples from training set...")
    train_subset = dataset["train"].shuffle(seed=SEED).select(range(NUM_TRAIN_SAMPLES))

    # 2. data preporcessing (Map)
    # "context/question/answers" to "input_ids/labels"
    print("Tokenizing and Masking labels...")
    tokenized_train_dataset = train_subset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names, # remove original text columns, only keep tensor
        num_proc=8
    )

    # 3. Data Collator
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer=tokenizer,
    #     model=None,
    #     padding=True,
    #     pad_to_multiple_of=8
    # )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=False,
    )

    # 4. load model
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        use_cache=False 
    )

    # 5. Initialize Callback
    # Callback for evaluating after each epoch
    epoch_eval_callback = SquadEpochEvalCallback(
        tokenizer=tokenizer,
        eval_dataset=dataset["validation"],
        num_samples=200
    )

    # 6. Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=20,
        # max_steps = 1,
        bf16=True,
        # logging_steps=10,
        logging_strategy="epoch",
        eval_strategy="no", 
        save_strategy="no",
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        # warmup_ratio=0.1,
        # lr_scheduler_type="cosine",
        # report_to="none",
        remove_unused_columns=True # only use input_ids, labels, ...
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator, # collator processes batch padding
        tokenizer=tokenizer,
        callbacks=[epoch_eval_callback] 
    )

    # evaluate Baseline 
    # print("Running baseline evaluation...")
    # run_evaluation(model, tokenizer, dataset["validation"], num_samples=200)
    
    # # cutlass control file
    # controlFP = f"/home/yuhangl/control_{job_id}/perform.txt"
    # cutlassFP = f"/home/yuhangl/control_{job_id}/cutlass.txt"
    # SMChkFP = f"/home/yuhangl/control_{job_id}/split.txt"

    with open(controlFP, 'w') as file:
        file.truncate(0)
        file.write("t")

    with open(cutlassFP, 'w') as file:
        file.truncate(0)
        file.write("f")

    with open(SMChkFP, 'w') as file:
        file.truncate(0)
        file.write("f")

    print("Starting Training...")
    trainer.train()

    with open(controlFP, 'w') as file:
        file.truncate(0)
        file.write("f")

    log = trainer.state.log_history
    smchk_loss = [str(e.get("loss", "")) for e in log if "loss" in e]
    grad_norm = [str(e.get("grad_norm", "")) for e in log if "grad_norm" in e]
    
    # print(f"Saving model to {OUTPUT_DIR}...")
    # trainer.save_model(OUTPUT_DIR)
    # tokenizer.save_pretrained(OUTPUT_DIR)

    # # 8. Final evaluation (Using the trained model)
    # print("Running Final Evaluation...")
    # model.config.use_cache = True
    # final_F1_score = run_evaluation(trainer.model, tokenizer, dataset["validation"], num_samples=200)
    
    torch.cuda.empty_cache()

    with open(logFP, "a") as file:
        file.write("\n")
        file.write(" ".join(smchk_loss))
        file.write("\n")
        file.write(" ".join(grad_norm))
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