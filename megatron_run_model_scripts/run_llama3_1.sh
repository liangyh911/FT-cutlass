#!/bin/bash

# [Rank 0] (after 1 iterations) memory (MB) | allocated: 34663.55029296875 | max allocated: 44819.12841796875 | reserved: 45930.0 | max reserved: 45930.0
# [Rank 2] (after 1 iterations) memory (MB) | allocated: 34663.55029296875 | max allocated: 44819.12841796875 | reserved: 45930.0 | max reserved: 45930.0
# [Rank 1] (after 1 iterations) memory (MB) | allocated: 34663.55029296875 | max allocated: 44819.12841796875 | reserved: 45930.0 | max reserved: 45930.0
# [Rank 3] (after 1 iterations) memory (MB) | allocated: 34663.55029296875 | max allocated: 44819.12841796875 | reserved: 45930.0 | max reserved: 45930.0

# 30 mins/experiment

# clean logs, "cannot import name 'triton_key"
export TORCH_LOGS="-dynamo"

# Eval_mode
eval_mode=$1

# CoreChecker mode
core_checker_mode=$2

# Get Job id
JOB_ID=$SLURM_JOB_ID

# rename control file
mv ./control_llama3 ./control_$JOB_ID

# faulty starting step and faulty duration
FAULTY_STEP=490
FAULTY_DURATION=20

# directory
MODEL_DIR="/projects"
DS_DIR="/mnt"

# set TP and PP
TP=2
PP=2

# Model size
MODEL_Size="8B"

# set the faulty GPU
FAULTY_GPU="-1"
if [[ "$eval_mode" == "1" ]]; then
  FAULTY_GPU="0"
fi
printf "%d" "$FAULTY_GPU" > "./control_$JOB_ID/faulty_GPU.txt"

# set the faulty bit
if [[ "$FAULTY_GPU" != "-1" ]]; then
  FAULTY_BIT=12
  printf "%d" "$FAULTY_BIT" > "./control_$JOB_ID/$FAULTY_GPU/bit.txt"
  FAULTY_UNIT=5
  printf "%d" "$FAULTY_UNIT" > "./control_$JOB_ID/$FAULTY_GPU/total_faulty_steps.txt"
fi

# checkpointing interval
checkpointing_interval=1000
total_steps=1000
# num_checkpoints=$((total_steps / checkpointing_interval))
# if (( num_checkpoints == 0 )); then
#     num_checkpoints=1
# fi

for i in {1..1}; do 
  echo "--- Iteration $i ---" >> "./control_$JOB_ID/eval_results.txt"

  # read ground turth faulty gpu
  if [[ "$FAULTY_GPU" != "-1" ]]; then
    head -n 1 "./control_$JOB_ID/$FAULTY_GPU/plan.txt" | awk '{print $1}' >> "./control_$JOB_ID/eval_results.txt"
  fi
  
  # head -n 1 "./control_$JOB_ID/$FAULTY_GPU/plan.txt" | awk '{print $1}' >> "./control_$JOB_ID/eval_results.txt"
  
  # Get time stamp
  # TIME_STAMP=$EPOCHSECONDS
  # TIME_STAMP=$(date +%s)
  # echo "$TIME_STAMP" >> "./control_$JOB_ID/eval_results.txt"

  # init control files for trainning
  for d in ./control_$JOB_ID/*; do
    [ -f "$d/perform.txt" ] && printf "t" > "$d/perform.txt"
    [ -f "$d/cutlass.txt" ] && printf "f" > "$d/cutlass.txt"

    if [[ "$core_checker_mode" != "baseline" ]]; then
      [ -f "$d/enable_core_checker.txt" ] && printf "f" > "$d/enable_core_checker.txt"
      [ -f "$d/adaptive_mod.txt" ] && printf "f" > "$d/adaptive_mod.txt"
    elif [[ "$core_checker_mode" != "basic" ]]; then
      [ -f "$d/enable_core_checker.txt" ] && printf "t" > "$d/enable_core_checker.txt"
      [ -f "$d/adaptive_mod.txt" ] && printf "f" > "$d/adaptive_mod.txt"
      Check_Freq=1
      [ -f "$d/check_freq.txt" ] && printf "%d" "$Check_Freq" > "$d/check_freq.txt"
    elif [[ "$core_checker_mode" != "1" ]]; then
      [ -f "$d/enable_core_checker.txt" ] && printf "t" > "$d/enable_core_checker.txt"
      [ -f "$d/adaptive_mod.txt" ] && printf "f" > "$d/adaptive_mod.txt"
      Check_Freq=1
      [ -f "$d/check_freq.txt" ] && printf "%d" "$Check_Freq" > "$d/check_freq.txt"
    elif [[ "$core_checker_mode" != "2" ]]; then
      [ -f "$d/enable_core_checker.txt" ] && printf "t" > "$d/enable_core_checker.txt"
      [ -f "$d/adaptive_mod.txt" ] && printf "t" > "$d/adaptive_mod.txt"
      Check_Freq=10
      [ -f "$d/check_freq.txt" ] && printf "%d" "$Check_Freq" > "$d/check_freq.txt"
    fi

    # set DEBUG 
    if [[ "$eval_mode" == "0" ]]; then
      [ -f "$d/DEBUG.txt" ] && printf "t" > "$d/DEBUG.txt"
    else
      [ -f "$d/DEBUG.txt" ] && printf "f" > "$d/DEBUG.txt"
    fi


    # # set DEBUG 
    # [ -f "$d/DEBUG.txt" ] && printf "f" > "$d/DEBUG.txt"
    # # enable core_checker
    # # [ -f "$d/split.txt" ] && printf "t" > "$d/split.txt"
    # [ -f "$d/enable_core_checker.txt" ] && printf "t" > "$d/enable_core_checker.txt"
    # # enable core_checker adaptive mode
    # [ -f "$d/adaptive_mod.txt" ] && printf "t" > "$d/adaptive_mod.txt"
    # Check_Freq=10
    # [ -f "$d/check_freq.txt" ] && printf "%d" "$Check_Freq" > "$d/check_freq.txt"

    # clear context of files
    [ -f "$d/faults_precentage.bin" ] && printf "" > "$d/faults_precentage.bin"
    [ -f "$d/fi_info.bin" ] && printf "" > "$d/fi_info.bin"
    [ -f "$d/fi_info.bin" ] && printf "" > "$d/value_range.bin"

    [ -f "$d/banned_smid.txt" ] && printf "" > "$d/banned_smid.txt"
    [ -f "$d/SM_checking_results.txt" ] && printf "" > "$d/SM_checking_results.txt"

    [ -f "$d/time/attn.txt" ] && printf "" > "$d/time/attn.txt"
    [ -f "$d/time/mlp.txt" ] && printf "" > "$d/time/mlp.txt"
    [ -f "$d/time/gemm.txt" ] && printf "" > "$d/time/gemm.txt"
    [ -f "$d/time/bgemm.txt" ] && printf "" > "$d/time/bgemm.txt"
    [ -f "$d/time/update.txt" ] && printf "" > "$d/time/update.txt"
    [ -f "$d/time/preparation.txt" ] && printf "" > "$d/time/preparation.txt"
    
    # # get directory basename (assuming directories are numbered by GPU)
    # gpu_id=$(basename "$d")

    # # sync fault injection plan, skip the faulty GPU
    # if [[ "$gpu_id" != "$FAULTY_GPU" ]]; then
    #   cp "./control_$JOB_ID/$FAULTY_GPU/faulty_step.txt" "$d/faulty_step.txt"
    #   cp "./control_$JOB_ID/$FAULTY_GPU/plan.txt" "$d/plan.txt"
    # fi
  done

  # Trainning
  sh ./Pai-Megatron-Patch/examples/llama3_1/run_mcore_llama3_1.sh  \
  dsw  \
  "$MODEL_Size"   \
  8    \
  8 \
  1e-6   \
  1e-7   \
  1024  \
  1024  \
  bf16  \
  "$TP"   \
  "$PP"  \
  1 \
  false \
  true   \
  false \
  true \
  sel   \
  false \
  "$checkpointing_interval"  \
  "$DS_DIR/llama3-datasets/mmap_llama3_sft_datasets_en_text_document"   \
  "$DS_DIR/llama3-datasets/mmap_llama3_sft_datasets_en_text_document"   \
  "$MODEL_DIR/llama3-ckpts/Meta-Llama-3.1-8B-mcore-tp2-pp2"  \
  "$total_steps"   \
  0   \
  "$MODEL_DIR/logs/output_mcore_llama3_1"
  # "$MODEL_DIR/logs/output_mcore_llama3_1"

  # Disable cutlass for evaluation
  for d in ./control_$JOB_ID/*; do
    [ -f "$d/perform.txt" ] && printf "f" > "$d/perform.txt"
    [ -f "$d/cutlass.txt" ] && printf "f" > "$d/cutlass.txt"
    [ -f "$d/split.txt" ] && printf "f" > "$d/split.txt"
  done

  # Read faulty step
  if [[ "$FAULTY_GPU" != "-1" ]]; then
    src="./control_$JOB_ID/$FAULTY_GPU/faulty_step.txt"
    dst="./control_$JOB_ID/eval_results.txt"
    head -n 1 "$src" >> "$dst"
    sed -i '1d' "$src"
  fi

  # Evaluation
  # ckpt_abs_path=$(realpath /mnt/logs/output_mcore_llama3_1/checkpoint/*)
  # ckpt_abs_path=$(realpath /projects/logs/output_mcore_llama3_1/checkpoint/*)
  # num_checkpoints=$(find $ckpt_abs_path -maxdepth 1 -type d -name 'iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9]' | wc -l)

  ckpt_abs_path=$(realpath $MODEL_DIR/logs/output_mcore_llama3_1/checkpoint/*)
  # collect all iters (ascending, strip leading zeros)
  mapfile -t all_iters < <(
    find "$ckpt_abs_path" -maxdepth 1 -type d -name 'iter_[0-9]*' \
      -printf '%f\n' | sed 's/^iter_//' | sort -n
  )

  num_iters=${#all_iters[@]}
  echo "Found $num_iters checkpoints: ${all_iters[*]}"

  if [[ $num_iters -eq 0 ]]; then
    echo "ERROR: No iter_xxxxxxx found in $ckpt_abs_path"
    exit 1
    # continue
  fi

  if [[ "$eval_mode" != "1" ]]; then
    num_iters=0
  fi

  for ((i=0; i<num_iters; i++)); do
  # for ((i=1; i<=1; i++)); do
    current_iter=${all_iters[$i]}
    echo "--- Processing iter $current_iter ---"
    
    # select one checkpoint
    printf "$current_iter" > "$ckpt_abs_path/latest_checkpointed_iteration.txt"

    # convert the checkpoint (8B needs more than 64GB Memory)
    cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
    bash hf2mcore_convertor_llama3_1.sh \
    "$MODEL_Size" \
    "$ckpt_abs_path"    \
    "$MODEL_DIR/llama3-ckpts/Llama-3.1-8B-mcore-to-hf-cutlass"  \
    "$TP"  \
    "$PP"  \
    true \
    true \
    false \
    bf16 \
    "$MODEL_DIR/llama3-ckpts/Meta-Llama-3.1-8B"

    # evaluation
    cd /workspace
    # CUDA_VISIBLE_DEVICES=0 python ./model_evaluation_scripts/llama3_1_GE_eval.py
    CUDA_VISIBLE_DEVICES=0 python ./model_evaluation_scripts/llama3_1_multi_tasks.py

    # delete current checkpoint
    rm -r $MODEL_DIR/llama3-ckpts/Llama-3.1-8B-mcore-to-hf-cutlass
  done

  printf "" > "$d/training.txt"

  # Read Loss and Grad norm from tensorboard
  SRC="$MODEL_DIR/logs/output_mcore_llama3_1/tensorboard"
  python read_tensorboard.py $SRC

  # add new line to validation file
  printf "\n" >> "/workspace/control_$JOB_ID/eval_results.txt"

  # # collect the tensorboard log
  # SRC="$MODEL_DIR/logs/output_mcore_qwen3_finetune/tensorboard"
  # DEST_BASE="/workspace/tensorboards_log"
  # NEW_NAME="finetune-mcore-qwen3-moe-megatron-8B-$TIME_STAMP"

  # mkdir -p "$DEST_BASE"

  # RUN_DIR=$(ls -d "$SRC"/*/ | head -n 1)

  # mv "$RUN_DIR" "$DEST_BASE/$NEW_NAME"

  # delete all checkpoints for this experiment
  rm -r $MODEL_DIR/logs/output_mcore_llama3_1
done

mv ./control_$JOB_ID ./control_llama3