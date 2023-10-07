# Tutorial 1: Introduction & Environment Setup

Arthor: Liu Yuxuan
\
Email: [118010200@link.cuhk.edu.cn](mailto:118010200@link.cuhk.edu.cn)

## Sample Program: Parallel Vector Addition

```bash
# MPI
mpic++ ./mpi_vector_addition.cpp -o ./mpi_vector_addition

# OpenMP
g++ -fopenmp ./openmp_vector_addition.cpp -o ./openmp_vector_addition

# Pthread
g++ -lpthread ./pthread_vector_addition.cpp -o ./pthread_vector_addition

# CUDA
nvcc ./cuda_vector_addition.cu -o ./cuda_vector_addition

# Execution
# sbatch approach
sbatch ./sbatch.sh
# Interactive approach
## Follow the content in sbatch.sh
```
