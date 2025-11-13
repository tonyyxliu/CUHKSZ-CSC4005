#!/bin/bash
#SBATCH -o ./Project3-results.txt
#SBATCH -p Release
#SBATCH -J Project3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

DATA_SIZE=100000000
SEARCH_SIZE=200000000  # Larger size for searching tasks

# std::sort Sequential
## Uniformly distributed dataset
echo "std::sort (Optimized with -O2)"
for num_cores in 1 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ../build/src/std_sort $DATA_SIZE
done
echo ""

# Task 1: Parallel Merge Sort with Parallel Merging on CPU
echo "Parallel Merge Sort with Parallel Merging on CPU (Optimized with -O2)"
for num_cores in 1 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ../build/src/mergesort/mergesort $num_cores $DATA_SIZE
done
echo ""

# Task 2: Parallel Quick Sort with Parallel Partitioning on CPU
echo "Parallel Quick Sort with Parallel Partitioning on CPU (Optimized with -O2)"
for num_cores in 1 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ../build/src/quicksort/quicksort $num_cores $DATA_SIZE
done
echo ""

# Task 3: Parallel Radix Sort on GPU
echo "Parallel Radix Sort on GPU (Optimized with -O2)"
srun -n 1 --cpus-per-task $num_cores --gres=gpu:1 ../build/src/radixsort-gpu/radixsort $DATA_SIZE
echo ""

# Task 4: Parallel Searching for Data Array on CPU
echo "Parallel Searching for Data Array on CPU (Optimized with -O2)"
for num_cores in 1 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ../build/src/searching-cpu/searching_array_parallel $num_cores $SEARCH_SIZE
done

echo "Parallel Searching for Data Array on GPU (Optimized with -O2)"
srun -n 1 --cpus-per-task $num_cores --gres=gpu:1 ../build/src/searching-gpu/searching_array_gpu $SEARCH_SIZE
