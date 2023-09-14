#!/bin/bash
#SBATCH -o ./result.out
#SBATCH -p Debug
#SBATCH -J Tutorial1-Sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

# Compile the programs before execution
srun hostname
# Four different processes for MPI (Multi-Process Program)
srun -n 4 --mpi=pmi2 ./mpi_vector_addition
# One task, four threads (Multi-Thread Program)
srun -n 1 --cpus-per-task 4 ./openmp_vector_addition
# One task, four threads (Multi-Thread Program)
srun -n 1 --cpus-per-task 4 ./pthread_vector_addition
# One task, with one GPU card
srun -n 1 --gpus 1 ./cuda_vector_addition
