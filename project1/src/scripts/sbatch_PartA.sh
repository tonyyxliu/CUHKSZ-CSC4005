#!/bin/bash
#SBATCH -o ./Project1-PartA-Results.txt
#SBATCH -p Project
#SBATCH -J Project1-PartA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Necessary Environment Variables for Triton
export TRITON_PTXAS_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/ptxas                                                                      
export TRITON_CUOBJDUMP_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/cuobjdump                                                              
export TRITON_NVDISASM_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/nvdisasm  
export PATH=/opt/rh/rh-python38/root/usr/bin:$PATH

# Get the current directory
CURRENT_DIR=$(pwd)/src/scripts
echo "Current directory: ${CURRENT_DIR}"

# Sequential PartA
echo "Sequential PartA (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../../build/src/cpu/sequential_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
echo ""

# SIMD PartA
echo "SIMD(AVX2) PartA (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../../build/src/cpu/simd_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
echo ""

# MPI PartA
echo "MPI PartA (Optimized with -O2)"
for num_processes in 1 2 4 8 16 32
do
  echo "Number of processes: $num_processes"
  srun -n $num_processes --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/../../build/src/cpu/mpi_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
  echo ""
done

# Pthread PartA
echo "Pthread PartA (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../../build/src/cpu/pthread_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg ${num_cores}
  echo ""
done

# OpenMP PartA
echo "OpenMP PartA (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../../build/src/cpu/openmp_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
  echo ""
done

# CUDA PartA
echo "CUDA PartA"
srun -n 1 --gpus 1 ${CURRENT_DIR}/../../build/src/gpu/cuda_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
echo ""

# OpenACC PartA
echo "OpenACC PartA"
srun -n 1 --gpus 1 ${CURRENT_DIR}/../../build/src/gpu/openacc_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
echo ""

# Triton PartA
echo "Triton PartA"
srun -n 1 --gpus 1 python3 ${CURRENT_DIR}/../gpu/triton_PartA.py ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
echo ""
