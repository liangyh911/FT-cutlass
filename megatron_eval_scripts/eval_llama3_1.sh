#!/bin/bash

# Get Job id
JOB_ID=$SLURM_JOB_ID

mv ./control ./control_$JOB_ID

# Run Baseline
bash ./megatron_run_model_scripts/run_llama3_1.sh 0 "Baseline"
python get_runtime_Megatron.py
runtime_baseline=$(cat ./control_$JOB_ID/training.txt)
attn_baseline=$(cat ./control_$JOB_ID/attn.txt)
mlp_baseline=$(cat ./control_$JOB_ID/attn.txt)

# Run CoreChecker-basic
bash ./megatron_run_model_scripts/run_llama3_1.sh 0 "Basic"
python get_runtime_Megatron.py
runtime_basic=$(cat ./control_$JOB_ID/training.txt)
attn_basic=$(cat ./control_$JOB_ID/attn.txt)
mlp_basic=$(cat ./control_$JOB_ID/attn.txt)

# Run CoreChecker-1-level-adapt
bash ./megatron_run_model_scripts/run_llama3_1.sh 0 "1"
python get_runtime_Megatron.py
runtime_1=$(cat ./control_$JOB_ID/training.txt)
attn_1=$(cat ./control_$JOB_ID/attn.txt)
mlp_1=$(cat ./control_$JOB_ID/attn.txt)

# Run CoreChecker-2-level-adapt
bash ./megatron_run_model_scripts/run_llama3_1.sh 0 "2"
python get_runtime_Megatron.py
runtime_2=$(cat ./control_$JOB_ID/training.txt)
attn_2=$(cat ./control_$JOB_ID/attn.txt)
mlp_2=$(cat ./control_$JOB_ID/attn.txt)

# calculate overhead
echo "Baseline runtime: $runtime_baseline"

overhead=$((runtime_basic-runtime_baseline) / runtime_baseline)
echo "CoreChecker basic runtime: $runtime_basic, overhead: $overhead"
overhead=$((attn_basic-attn_baseline) / attn_baseline)
echo "CoreChecker basic ATTN runtime: $attn_basic, overhead: $overhead"
overhead=$((mlp_basic-mlp_baseline) / mlp_baseline)
echo "CoreChecker basic MLP runtime: $mlp_basic, overhead: $overhead"

overhead=$((runtime_1-runtime_baseline) / runtime_baseline)
echo "CoreChecker-1-level-adapt Training runtime: $runtime_1, overhead: $overhead"
overhead=$((attn_1-attn_baseline) / attn_baseline)
echo "CoreChecker-1-level-adapt ATTN runtime: $attn_1, overhead: $overhead"
overhead=$((mlp_1-mlp_baseline) / mlp_baseline)
echo "CoreChecker-1-level-adapt MLP runtime: $mlp_1, overhead: $overhead"

overhead=$((runtime_2-runtime_baseline) / runtime_baseline)
echo "CoreChecker-2-level-adapt Training runtime: $runtime_2, overhead: $overhead"
overhead=$((attn_2-attn_baseline) / attn_baseline)
echo "CoreChecker-2-level-adapt ATTN runtime: $attn_2, overhead: $overhead"
overhead=$((mlp_2-mlp_baseline) / mlp_baseline)
echo "CoreChecker-2-level-adapt MLP runtime: $mlp_2, overhead: $overhead"

# Detection Evaluation
bash ./megatron_run_model_scripts/run_llama3_1.sh 1 "2"

mv ./control_$JOB_ID ./control