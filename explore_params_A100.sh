#!/bin/bash

usage() {
  echo "Usage: $0 <duration> <model_name> <dataset> <parameters_file>"
  echo "  <duration>        short, default, or long training"
  echo "  <model_name>      Name of the model"
  echo "  <dataset>         Name of the dataset"
  echo "  <parameters_file> Path to the parameters file"
  exit 1
}

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
  usage
fi

# Read the command-line arguments
DURATION="$1"
MODEL_NAME="$2"
DATASET="$3"
PARAMETERS_FILE="$4"

# Check if the parameters file exists
if [[ ! -f "$PARAMETERS_FILE" ]]; then
  echo "Parameters file not found: $PARAMETERS_FILE"
  exit 1
fi

# Define the static part of the srun command
base_srun_command="srun --input none -C A100 --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --time=360 python train_lightning.py --nodes 1 --workers 20 --dataset $DATASET --model $MODEL_NAME --duration $DURATION"

# Read the parameters and execute srun commands
while IFS=' ' read -r groups s0 s1 s2; do
  # Construct the full srun command with dynamic parameters
  full_srun_command="$base_srun_command --groups $groups --s0 $s0 --s1 $s1 --s2 $s2"

  # Run the srun command
  echo "Running: $full_srun_command"
  $full_srun_command
done < "$PARAMETERS_FILE"

echo "All srun commands executed."

