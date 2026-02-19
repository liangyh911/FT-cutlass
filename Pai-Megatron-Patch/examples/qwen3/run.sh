sh run_mcore_qwen3.sh  \
dsw  \
4B   \
8    \
8 \
1e-5   \
1e-6   \
1024  \
1024  \
bf16  \
2   \
1  \
1 \
1 \
1 \
false \
true   \
false \
true \
sel   \
false \
100000  \
/mnt/qwen-datasets/mmap_qwen3_sft_datasets_text_document   \
/mnt/qwen-datasets/mmap_qwen3_sft_datasets_text_document   \
/mnt/qwen-ckpts/Qwen3-4B-to-mcore  \
1  \
0   \
/mnt/logs/output_mcore_qwen3_finetune

# # export MP_DATASET_TYPE="raw"
# sh run_mcore_qwen3.sh  \
# dsw  \
# 4B   \
# 8    \
# 8 \
# 1e-5   \
# 1e-6   \
# 1024  \
# 1024  \
# bf16  \
# 2   \
# 1  \
# 1 \
# 1 \
# 1 \
# false \
# true   \
# false \
# true \
# sel   \
# false \
# 100000  \
# /mnt/qwen-datasets/alpaca_zh-train-general.json    \
# /mnt/qwen-datasets/alpaca_zh-valid-general.json   \
# /mnt/qwen-ckpts/Qwen3-4B-to-mcore  \
# 100  \
# 0   \
# /mnt/logs/output_mcore_qwen3_finetune_2

# bash scripts/qwen3/run_8xH20.sh \
# 4B \
# /mnt/qwen-ckpts/Qwen3-4B \
# /mnt/qwen-ckpts/Qwen3-4B-to-mcore  \
# false \
# true \
# bf16

# bash scripts/qwen3/run_8xH20.sh \
# 4B \
# /mnt/qwen-ckpts/Qwen3-4B-to-mcore \
# /mnt/qwen-ckpts/Qwen3-4B-mcore-to-hf  \
# true \
# true \
# bf16 \
# /mnt/qwen-ckpts/Qwen3-4B

# bash scripts/qwen3/run_8xH20.sh \
# 4B \
# /mnt/logs/output_mcore_qwen3_finetune/checkpoint/finetune-mcore-qwen3-moe-megatron-4B  \
# /mnt/qwen-ckpts/Qwen3-4B-mcore-to-hf-cutlass  \
# true \
# true \
# bf16 \
# /mnt/qwen-ckpts/Qwen3-4B

# accelerate launch --main_process_port 29051 -m lm_eval \
# --model hf \
# --model_args pretrained=./test/Qwen3-4B-mcore-to-hf_iter_0004000,trust_remote_code=True \
# --tasks cmmlu,ceval-valid  \
# --batch_size 16

# 4.57.3


# bash run_build_idxmap_sft_dataset.sh \
# /mnt/qwen-datasets/qwen_sft.json \
# Qwen3Tokenizer \
# 1024 \
# /mnt/qwen-datasets/mmap_qwen3_sft_datasets \
# /mnt/qwen-ckpts/Qwen3-4B

# QKV
# input_parallel: torch.Size([1024, 8, 2560]), weight: torch.Size([3072, 2560]), output_parallel: torch.Size([1024, 8, 3072])
# hidden_states: torch.Size([1024, 8, 2560]), query: torch.Size([1024, 8, 16, 128]), key: torch.Size([1024, 8, 4, 128]), value: torch.Size([1024, 8, 4, 128])
# QK
# query: (torch.Size([1024, 128, 128])->torch.Size([128, 1024, 128])), key: (torch.Size([1024, 128, 128])->torch.Size([128, 128, 1024])), AS: torch.Size([128, 1024, 1024])
# AV
# attention_probs: torch.Size([128, 1024, 1024]), value: (torch.Size([1024, 128, 128])->torch.Size([128, 1024, 128])), context: torch.Size([128, 1024, 128])
# WO
# core_attn_out: torch.Size([1024, 8, 2048]), output: torch.Size([1024, 8, 2560])
# UP
# input_parallel: torch.Size([1024, 8, 2560]), weight: torch.Size([9728, 2560]), output_parallel: torch.Size([1024, 8, 9728])
# hidden_states: torch.Size([1024, 8, 2560]), intermediate_parallel: torch.Size([1024, 8, 9728])
# DO
# intermediate_parallel: torch.Size([1024, 8, 4864]), output: torch.Size([1024, 8, 2560])