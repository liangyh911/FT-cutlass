#!/bin/bash

# make model and dataset dir
cd /mnt
mkdir qwen-ckpts qwen-datasets

# download the model checkpoint
modelscope download --model Qwen/Qwen3-8B --local_dir ./qwen-ckpts/Qwen3-8B

# convert HF model to Megatron-Core model
cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen3/run_8xH20.sh \
8B \
/mnt/qwen-ckpts/Qwen3-8B \
/mnt/qwen-ckpts/Qwen3-8B-to-mcore  \
false \
true \
bf16

# download the dataset
cd /mnt/qwen-datasets
wget --no-check-certificate https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/qwen_sft.json

# make the datasets
cd /workspace/Pai-Megatron-Patch/toolkits/sft_data_preprocessing
bash run_build_idxmap_sft_dataset.sh \
/mnt/qwen-datasets/alpaca_data.json \
Qwen3Tokenizer \
1024 \
/mnt/qwen-datasets/mmap_qwen3_sft_datasets_en \
/mnt/qwen-ckpts/Qwen3-8B