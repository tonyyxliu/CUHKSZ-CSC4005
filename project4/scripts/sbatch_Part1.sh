#!/bin/bash
#SBATCH -o ./Project4-Results-Part1.txt
#SBATCH -p Release
#SBATCH -J Project4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Necessary Environment Variables for Triton

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "------------------------------------------------------------------"

# Part 1: Triton Softmax
echo ">>> Running Part 1: Triton Softmax"

srun -n 1 --gpus 1 python3 ./part1_softmax_and_vector_add/triton_softmax.py

echo "------------------------------------------------------------------"


# Part 2: CUDA Softmax
echo ">>> Running Part 2: CUDA Softmax"

echo "Compiling CUDA code"
#nvcc ./part1_softmax_and_vector_add/cuda_softmax.cu -o ./part1_softmax_and_vector_add/cuda_softmax_exec
nvcc ./part1_softmax_and_vector_add/cuda_softmax.cu -o ./part1_softmax_and_vector_add/cuda_softmax_exec
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi
echo "Compilation successful."
echo ""

N=8192
C=8192

srun -n 1 --gpus 1 ./part1_softmax_and_vector_add/cuda_softmax_exec $N $C

echo ""
echo "Job finished at: $(date)"
