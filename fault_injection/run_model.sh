#!/bin/bash

# Get Job id
JOB_ID=$SLURM_JOB_ID

# rename control file
mv /home/yuhangl/control_1 /home/yuhangl/control_$JOB_ID

OUTPUT_FILE="/home/yuhangl/control_$JOB_ID/llama_output.log"

# Running a Python script a fixed number of times
for i in {1..1}; do
  echo "--- Iteration $i ---" >> "$OUTPUT_FILE"
  python llama.py
  echo "" >> "$OUTPUT_FILE"
done

mv /home/yuhangl/control_$JOB_ID /home/yuhangl/control_1