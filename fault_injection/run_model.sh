#!/bin/bash

OUTPUT_FILE="llama_output.log"

# Example 1: Running a Python script a fixed number of times
for i in {1..3}; do
  echo "--- Iteration $i ---" >> "$OUTPUT_FILE"
  python llama.py >> "$OUTPUT_FILE" 2>&1
  echo "" >> "$OUTPUT_FILE"
done

# # Example 2: Running a Python script with different arguments from a list
# arguments=("arg1" "arg2" "arg3")
# for arg in "${arguments[@]}"; do
#   echo "Running Python script with argument: $arg"
#   python3 your_script.py "$arg"
# done
