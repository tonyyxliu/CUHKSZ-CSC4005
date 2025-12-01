#!/bin/bash
#SBATCH -o ./Project4-results-Part3.txt
#SBATCH -p Release
#SBATCH -J Project4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Necessary Environment Variables for Triton

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "------------------------------------------------------------------"

# Part 3
echo ">>> Running Part 3: Triton Flash Attention Sparse"

srun -n 1 --gpus 1 python3 ./part3_sparse_flash_attention/triton_part.py

echo "------------------------------------------------------------------"

echo ""
echo "Job finished at: $(date)"