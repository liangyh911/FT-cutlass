#!/bin/bash

# clean logs, "cannot import name 'triton_key"
export TORCH_LOGS="-dynamo"

# Get Job id
JOB_ID=$SLURM_JOB_ID

# rename control file
mv ./control_qwen2 ./control_$JOB_ID
  
# faulty starting step and faulty duration
FAULTY_STEP=150
FAULTY_DURATION=20

# directory
MODEL_DIR="/projects"
DS_DIR="/mnt"

# set TP and PP
TP=2
PP=2

# Model size
MODEL_Size="7B"

# set the faulty GPU
FAULTY_GPU="0"
printf "%d" "$FAULTY_GPU" > "./control_$JOB_ID/faulty_GPU.txt"

# set the faulty bit
if [[ "$FAULTY_GPU" != "-1" ]]; then
  FAULTY_BIT=12
  printf "%d" "$FAULTY_BIT" > "./control_$JOB_ID/$FAULTY_GPU/bit.txt"
  FAULTY_UNIT=5
  printf "%d" "$FAULTY_UNIT" > "./control_$JOB_ID/$FAULTY_GPU/total_faulty_steps.txt"
fi

for i in {1..33}; do 
  echo "--- Iteration $i ---" >> "./control_$JOB_ID/eval_results.txt"

  # Get time stamp
  # TIME_STAMP=$EPOCHSECONDS
  # TIME_STAMP=$(date +%s)
  # echo "$TIME_STAMP" >> "./control_$JOB_ID/eval_results.txt"

  # init control files for trainning
  for d in ./control_$JOB_ID/*; do
    [ -d "$d" ] || continue

    [ -f "$d/perform.txt" ] && printf "t" > "$d/perform.txt"
    [ -f "$d/perform.txt" ] && printf "f" > "$d/cutlass.txt"
    [ -f "$d/split.txt" ] && printf "f" > "$d/split.txt"

    # clear context of files
    [ -f "$d/faults_precentage.bin" ] && printf "" > "$d/faults_precentage.bin"
    [ -f "$d/fi_info.bin" ] && printf "" > "$d/fi_info.bin"
    [ -f "$d/fi_info.bin" ] && printf "" > "$d/value_range.bin"

    # # get directory basename (assuming directories are numbered by GPU)
    # gpu_id=$(basename "$d")

    # # sync fault injection plan, skip the faulty GPU
    # if [[ "$FAULTY_GPU" =~ ^[0-3]$ && "$gpu_id" != "$FAULTY_GPU" ]]; then
    #   cp "./control_$JOB_ID/$FAULTY_GPU/faulty_step.txt" "$d/faulty_step.txt"
    #   cp "./control_$JOB_ID/$FAULTY_GPU/plan.txt" "$d/plan.txt"
    # fi
  done

  # Fine-tuning
  checkpointing_interval=1000
  total_steps=1000

  sh ./Pai-Megatron-Patch/examples/qwen2_5/run_mcore_qwen.sh  \
  dsw  \
  "$MODEL_Size"   \
  8    \
  8 \
  1e-5   \
  1e-6   \
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
  "$DS_DIR/qwen-datasets/mmap_qwen2_sft_datasets_en_text_document"   \
  "$DS_DIR/qwen-datasets/mmap_qwen2_sft_datasets_en_text_document "  \
  "$MODEL_DIR/qwen-ckpts/Qwen2.5-7B-to-mcore-tp2-pp2"  \
  "$total_steps"  \
  50   \
  "$MODEL_DIR/logs/output_mcore_qwen2.5_finetune"

  # Disable cutlass for evaluation
  for d in ./control_$JOB_ID/*; do
    [ -f "$d/perform.txt" ] && printf "f" > "$d/perform.txt"
    [ -f "$d/perform.txt" ] && printf "f" > "$d/cutlass.txt"
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
  ckpt_abs_path=$(realpath $MODEL_DIR/logs/output_mcore_qwen2.5_finetune/checkpoint/*)
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
  fi

  for ((i=0; i<num_iters; i++)); do
    current_iter=${all_iters[$i]}
    echo "--- Processing iter $current_iter ---"

    # write current iter (no zero padding)
    printf "$current_iter" > "$ckpt_abs_path/latest_checkpointed_iteration.txt"

    # convert the checkpoint
    cd ./Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
    bash scripts/qwen2_5/run_8xH20.sh \
      "$MODEL_Size" \
      "$ckpt_abs_path" \
      "$MODEL_DIR/qwen-ckpts/Qwen2.5-7B-mcore-to-hf-cutlass" \
      true \
      true \
      bf16 \
      "$MODEL_DIR/qwen-ckpts/Qwen2.5-7B" \
      "$TP" \
      "$PP"

    # evaluation
    cd /workspace
    # CUDA_VISIBLE_DEVICES=0 python ./model_evaluation_scripts/qwen2_5_GE_eval.py
    CUDA_VISIBLE_DEVICES=0 python ./model_evaluation_scripts/qwen2_5_multi_tasks.py

    # cleanup
    rm -rf $MODEL_DIR/qwen-ckpts/Qwen2.5-7B-mcore-to-hf-cutlass
  done

  # Read Loss and Grad norm from tensorboard
  SRC="$MODEL_DIR/logs/output_mcore_qwen2.5_finetune/tensorboard"
  python read_tensorboard.py $SRC

  # add new line to validation file
  printf "\n" >> "/workspace/control_$JOB_ID/eval_results.txt"

  # # collect the tensorboard log
  # SRC="$MODEL_DIR/logs/output_mcore_qwen2.5_finetune/tensorboard"
  # DEST_BASE="/workspace/tensorboards_log"
  # NEW_NAME="finetune-mcore-qwen2.5-moe-megatron-8B-$TIME_STAMP"

  # mkdir -p "$DEST_BASE"

  # RUN_DIR=$(ls -d "$SRC"/*/ | head -n 1)

  # mv "$RUN_DIR" "$DEST_BASE/$NEW_NAME"

  # delete all checkpoints for this experiment
  rm -r $MODEL_DIR/logs/output_mcore_qwen2.5_finetune

done

mv ./control_$JOB_ID ./control_qwen2