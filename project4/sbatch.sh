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

TRAIN_X=./dataset/training/train-images.idx3-ubyte
TRAIN_Y=./dataset/training/train-labels.idx1-ubyte
TEST_X=./dataset/testing/t10k-images.idx3-ubyte
TEST_Y=./dataset/testing/t10k-labels.idx1-ubyte

# Softmax
echo "Softmax Sequential"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/softmax $TRAIN_X $TRAIN_Y $TEST_X $TEST_Y
echo ""

echo "Softmax OpenACC"
srun -n 1 --gpus 1 ${CURRENT_DIR}/build/softmax_openacc $TRAIN_X $TRAIN_Y $TEST_X $TEST_Y
echo ""

# NN
echo "NN Sequential"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/nn $TRAIN_X $TRAIN_Y $TEST_X $TEST_Y
echo ""

echo "NN OpenACC"
srun -n 1 --gpus 1 ${CURRENT_DIR}/build/nn_openacc $TRAIN_X $TRAIN_Y $TEST_X $TEST_Y
echo ""