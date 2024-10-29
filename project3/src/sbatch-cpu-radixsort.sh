#!/bin/bash
#SBATCH -o ./Project3-cpu-radixsort.txt
#SBATCH -p Project
#SBATCH -J Project
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32


# Radix Sort
# Sequential
echo "Radix Sort Sequential (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ./build/src/radixsort-cpu/radixsort_sequential 100000000
echo ""
# MPI
echo "Radix Sort MPI (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n $num_cores --cpus-per-task 1 --mpi=pmi2 ./build/src/radixsort-cpu/radixsort_mpi 100000000
done
echo ""
# OpenMP
echo "Radix Sort OpenMP (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ./build/src/radixsort-cpu/radixsort_omp $num_cores 100000000
done
echo ""