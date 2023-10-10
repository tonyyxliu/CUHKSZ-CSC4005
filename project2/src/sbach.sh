#!/bin/bash
#SBATCH -o /nfsmnt/117010349/CSC4005-2023Fall/project2/build/Project2.txt
#SBATCH -p Project
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Naive
echo "Naive Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 /nfsmnt/117010349/CSC4005-2023Fall/project2/build/src/naive /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix5.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix6.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/build/result.txt
echo ""

# Memory Locality
echo "Memory Locality Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 /nfsmnt/117010349/CSC4005-2023Fall/project2/build/src/locality /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix5.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix6.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/build/result.txt
echo ""

# SIMD + Reordering
echo "SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 /nfsmnt/117010349/CSC4005-2023Fall/project2/build/src/simd /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix5.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix6.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/build/result.txt
echo ""

# OpenMP + SIMD + Reordering
echo "OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores /nfsmnt/117010349/CSC4005-2023Fall/project2/build/src/openmp $num_cores /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix5.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix6.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/build/result.txt
  echo ""
done

# MPI + OpenMP + SIMD + Reordering
echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
echo "Number of Processes: 1, Number of Threads: 32"
srun -n 1 --cpus-per-task 32 --mpi=pmi2 /nfsmnt/117010349/CSC4005-2023Fall/project2/build/src/mpi 32 /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix5.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix6.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/build/result.txt
echo ""

echo "Number of Processes: 2, Number of Threads: 16"
srun -n 2 --cpus-per-task 16 --mpi=pmi2 /nfsmnt/117010349/CSC4005-2023Fall/project2/build/src/mpi 16 /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix5.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix6.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/build/result.txt
echo ""

echo "Number of Processes: 4, Number of Threads: 8"
srun -n 4 --cpus-per-task 8 --mpi=pmi2 /nfsmnt/117010349/CSC4005-2023Fall/project2/build/src/mpi 8 /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix5.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix6.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/build/result.txt
echo ""

echo "Number of Processes: 8, Number of Threads: 4"
srun -n 8 --cpus-per-task 4 --mpi=pmi2 /nfsmnt/117010349/CSC4005-2023Fall/project2/build/src/mpi 4 /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix5.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix6.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/build/result.txt
echo ""

echo "Number of Processes: 16, Number of Threads: 2"
srun -n 16 --cpus-per-task 2 --mpi=pmi2 /nfsmnt/117010349/CSC4005-2023Fall/project2/build/src/mpi 2 /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix5.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix6.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/build/result.txt
echo ""

echo "Number of Processes: 32, Number of Threads: 1"
srun -n 32 --cpus-per-task 1 --mpi=pmi2 /nfsmnt/117010349/CSC4005-2023Fall/project2/build/src/mpi 1 /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix5.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/matrices/matrix6.txt /nfsmnt/117010349/CSC4005-2023Fall/project2/build/result.txt
echo ""