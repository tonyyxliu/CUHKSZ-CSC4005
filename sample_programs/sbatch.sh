#!/bin/bash
#SBATCH -o ./sample-programs.txt
#SBATCH -p Debug
#SBATCH -J Sample-Programs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

# Get the current directory
CURRENT_DIR=$(pwd)
echo "Current directory: ${CURRENT_DIR}"

## Allocate Computing Resources
# Check the computing resource allocated to you
echo "hostname"
srun -n 4 hostname

## AVX512 Vectorization
echo "AVX512 Vectorization"
srun -n 4 ./avx512_vectorized_addition

## MPI
echo "MPI"
srun -n 4 --mpi=pmi2 ./mpi_hello
# or simply ignore the '--mpi=pmi2' with 'srun -n 4 ./mpi_hello'

## Pthread
echo "pthread"
srun -n 4 ./pthread_hello

## OpenMP
echo "OpenMP"
srun -n 4 ./openmp_hello

## CUDA
echo "CUDA"
srun -n 4 ./cuda_hello

## OpenACC
# OpenACC has not been configured for all nodes in the cluster yet.
# Before further announcement, use `./openacc_parallel` instead to execute locally
echo "OpenACC"
srun -n 4 ./openacc_parallel

# ## Triton
# echo "Triton"
# srun -n 1 python3 ./triton-tutorial/01-vector-add.py
