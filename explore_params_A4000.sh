#!/bin/bash

usage() {
  echo "Usage: $0 <duration> <model_name> <dataset> <parameters_file> <repetitions> <seed>"
  echo "  <duration>        short, default, or long training"
  echo "  <model_name>      Name of the model"
  echo "  <dataset>         Name of the dataset"
  echo "  <parameters_file> Path to the parameters file"
  echo "  <repetitions>     Number of times to repeat the execution"
  echo "  <seed>            Random seed to be passed to the training script"
  exit 1
}

# Check if the correct number of arguments is provided
if [ "$#" -ne 6 ]; then
  usage
fi

# Read the command-line arguments
DURATION="$1"
MODEL_NAME="$2"
DATASET="$3"
PARAMETERS_FILE="$4"
REPETITIONS="$5"
SEED="$6"

# Check if the parameters file exists
if [[ ! -f "$PARAMETERS_FILE" ]]; then
  echo "Parameters file not found: $PARAMETERS_FILE"
  exit 1
fi

# Define the static part of the srun command
base_srun_command="srun --input none -C A4000 --nodes=1 --ntasks-per-node=4 --gres=gpu:4 --time=300 python train_lightning.py --nodes 1 --workers 16 --dataset $DATASET --model $MODEL_NAME --duration $DURATION --seed $SEED --repetitions $REPETITIONS"

while IFS=' ' read -r groups s0 s1 s2; do
  # Construct the full srun command with dynamic parameters
  full_srun_command="$base_srun_command --groups $groups --s0 $s0 --s1 $s1 --s2 $s2"

  # Run the srun command
   echo "Running: $full_srun_command (Repetition $((i+1))/$REPETITIONS)"
  $full_srun_command
done < "$PARAMETERS_FILE"

echo "All srun commands executed."
