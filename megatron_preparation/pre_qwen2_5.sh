#!/bin/bash

# Download models
cd /mnt
modelscope download --model Qwen/Qwen2.5-7B  --local_dir ./qwen-ckpts/Qwen2.5-7B

# convert HF model to Megatron-Core model
cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen2_5/run_8xH20.sh \
7B \
/mnt/qwen-ckpts/Qwen2.5-7B \
/mnt/qwen-ckpts/Qwen2.5-7B-to-mcore  \
false \
true \
bf16

# make qwen 2.5 sft datasets
cd /workspace/Pai-Megatron-Patch/toolkits/sft_data_preprocessing
bash run_build_idxmap_sft_dataset.sh \
/mnt/qwen-datasets/alpaca_data.json \
Qwen2Tokenizer \
1024 \
/mnt/qwen-datasets/mmap_qwen2_sft_datasets_en \
/mnt/qwen-ckpts/Qwen2.5-7B

