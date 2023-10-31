#!/bin/bash
#SBATCH -o /nfsmnt/your/path/to/project3/build/Project3.txt
#SBATCH -p Project
#SBATCH -J Project3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Quick Sort
# Sequential
echo "Quick Sort Sequential (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 /nfsmnt/your/path/to/project3/build/src/quicksort/quicksort_sequential 100000000
echo ""
# MPI
echo "Quick Sort MPI (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n $num_cores --cpus-per-task 1 --mpi=pmi2 /nfsmnt/your/path/to/project3/build/src/quicksort/quicksort_mpi 100000000
done
echo ""

# Bucket Sort
# Sequential
echo "Bucket Sort Sequential (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 /nfsmnt/your/path/to/project3/build/src/bucketsort/bucketsort_sequential 100000000 1000000
echo ""
# MPI
echo "Bucket Sort MPI (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n $num_cores --cpus-per-task 1 --mpi=pmi2 /nfsmnt/your/path/to/project3/build/src/bucketsort/bucketsort_mpi 100000000 1000000
done
echo ""

# Odd-Even Sort
# Sequential
echo "Odd-Even Sort Sequential (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 /nfsmnt/your/path/to/project3/build/src/odd-even-sort/odd-even-sort_sequential 200000
echo ""
# MPI
echo "Odd-Even Sort MPI (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n $num_cores --cpus-per-task 1 --mpi=pmi2 /nfsmnt/your/path/to/project3/build/src/odd-even-sort/odd-even-sort_mpi 200000
done
echo ""
