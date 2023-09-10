# How to compile and run these programs

## Allocate Computing Resources
```bash
# Allocate 4 processors from 1 node in Debug partition, which lasts for 10 minutes
salloc -N1 -n4 -pDebug -t10
# Check the computing resource allocated to you
srun hostname
```

## AVX512 Vectorization
```bash
# Compilation
g++ -mavx512f avx512_vectorized_addition.cpp -o avx512_vectorized_addition
# Execution
srun -n 4 ./avx512_vectorized_addition
```

## MPI
```bash
# Check the MPI plugins for slurm
[root@node21 ~]srun --mpi=list
MPI plugin types are...
        cray_shasta
        none
        pmi2
# Compilation
mpic++ ./mpi_hello.cpp -o mpi_hello
# Execution
srun -n 4 --mpi=pmi2 ./mpi_hello
# or simply ignore the '--mpi=pmi2' with 'srun -n 4 ./mpi_hello'
```

## Pthread
```bash
# Compilation
g++ -lpthread pthread_hello.cpp -o pthread_hello
# Execution
srun -n 4 ./pthread_hello
```

## OpenMP
```bash
# Compilation
g++ -fopenmp openmp_hello.cpp -o openmp_hello
# Execution
srun -n 4 ./openmp_hello
```

## CUDA
```bash
# Compilation
nvcc cuda_hello.cu -o cuda_hello
# Execution
srun -n 4 ./cuda_hello
```

## OpenACC
```bash
# Compilation
pgc++ -acc -mp openacc_parallel.cpp -o openacc_parallel
# Execution
# OpenACC has not been configured for all nodes in the cluster yet.
# Before further announcement, use `./openacc_parallel` instead to execute locally
srun -n 4 ./openacc_parallel
```
