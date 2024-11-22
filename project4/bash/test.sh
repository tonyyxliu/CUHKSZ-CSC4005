#!/bin/bash
#SBATCH -o ./Project4-Results.txt
#SBATCH -p Project
#SBATCH -J Project4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Get the current directory
CURRENT_DIR=$(pwd)
echo "Current directory: ${CURRENT_DIR}"

# Path to the dataset
TRAIN_X=${CURRENT_DIR}/MINST/train-images-idx3-ubyte
TRAIN_Y=${CURRENT_DIR}/MINST/train-labels-idx1-ubyte
TEST_X=${CURRENT_DIR}/MINST/t10k-images-idx3-ubyte
TEST_Y=${CURRENT_DIR}/MINST/t10k-labels-idx1-ubyte

# Hyperparameters
HIDDEN_DIM=400
EPOCHS=10
LEARNING_RATE=0.001
BATCH=32

echo "Running"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/mlp $TRAIN_X $TRAIN_Y $TEST_X $TEST_Y $HIDDEN_DIM $EPOCHS $LEARNING_RATE $BATCH
echo ""