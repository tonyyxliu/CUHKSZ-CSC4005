#!/bin/bash
#SBATCH -o ./Project2-Results.txt
#SBATCH -p Project
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

CURRENT_DIR=$(pwd)/src

# Naive
echo "Naive Sparse Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src_sparse/naive_sparse ${CURRENT_DIR}/../matrices_sparse/matrix5.txt ${CURRENT_DIR}/../matrices_sparse/matrix6.txt ${CURRENT_DIR}/../build/result.txt
echo ""

# Parallelized
echo "Parallelized Sparse Matrix Multiplication (Optimized with -O2)"
echo "Number of Processes: 1, Number of Threads: 32"
srun -n 1 --cpus-per-task 32 --mpi=pmi2 ${CURRENT_DIR}/../build/src_sparse/parallelized_sparse 32 ${CURRENT_DIR}/../matrices_sparse/matrix5.txt ${CURRENT_DIR}/../matrices_sparse/matrix6.txt ${CURRENT_DIR}/../build/result.txt
echo ""

echo "Number of Processes: 2, Number of Threads: 16"
srun -n 2 --cpus-per-task 16 --mpi=pmi2 ${CURRENT_DIR}/../build/build/src_sparse/parallelized_sparse 16 ${CURRENT_DIR}/../matrices_sparse/matrix5.txt ${CURRENT_DIR}/../matrices_sparse/matrix6.txt ${CURRENT_DIR}/../build/result.txt
echo ""

echo "Number of Processes: 4, Number of Threads: 8"
srun -n 4 --cpus-per-task 8 --mpi=pmi2 ${CURRENT_DIR}/../build/src_sparse/parallelized_sparse 8 ${CURRENT_DIR}/../matrices_sparse/matrix5.txt ${CURRENT_DIR}/../matrices_sparse/matrix6.txt ${CURRENT_DIR}/../build/result.txt
echo ""

echo "Number of Processes: 8, Number of Threads: 4"
srun -n 8 --cpus-per-task 4 --mpi=pmi2 ${CURRENT_DIR}/../build/src_sparse/parallelized_sparse 4 ${CURRENT_DIR}/../matrices_sparse/matrix5.txt ${CURRENT_DIR}/../matrices_sparse/matrix6.txt ${CURRENT_DIR}/../build/result.txt
echo ""

echo "Number of Processes: 16, Number of Threads: 2"
srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src_sparse/parallelized_sparse 2 ${CURRENT_DIR}/../matrices_sparse/matrix5.txt ${CURRENT_DIR}/../matrices_sparse/matrix6.txt ${CURRENT_DIR}/../build/result.txt
echo ""

echo "Number of Processes: 32, Number of Threads: 1"
srun -n 32 --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/../build/src_sparse/parallelized_sparse 1 ${CURRENT_DIR}/../matrices_sparse/matrix5.txt ${CURRENT_DIR}/../matrices_sparse/matrix6.txt ${CURRENT_DIR}/../build/result.txt
echo ""