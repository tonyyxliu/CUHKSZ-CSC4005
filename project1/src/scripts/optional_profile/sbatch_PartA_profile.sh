#!/bin/bash
#SBATCH -o ./Project1-PartA-Results-Profile.txt
#SBATCH -p Release
#SBATCH -J Project1-PartA-Profile
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

# Necessary Environment Variables for Triton
export TRITON_PTXAS_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/12.2/bin/ptxas                                                                      
export TRITON_CUOBJDUMP_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/12.2/bin/cuobjdump                                                              
export TRITON_NVDISASM_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/12.2/bin/nvdisasm

# Get the current directory
CURRENT_DIR=$(pwd)/src/scripts
echo "Current directory: ${CURRENT_DIR}"

# Sequential PartA
echo "Sequential PartA (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 perf stat ${CURRENT_DIR}/../../build/src/cpu/sequential_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
echo ""

# SIMD PartA
echo "SIMD(AVX2) PartA (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 perf stat ${CURRENT_DIR}/../../build/src/cpu/simd_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
echo ""

# MPI PartA
echo "MPI PartA (Optimized with -O2)"
for num_processes in 1 2 4
do
  echo "Number of processes: $num_processes"
  srun -n $num_processes --cpus-per-task 1 --mpi=pmi2 perf stat ${CURRENT_DIR}/../../build/src/cpu/mpi_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
  echo ""
done

# Pthread PartA
echo "Pthread PartA (Optimized with -O2)"
for num_cores in 1 2 4
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores perf stat ${CURRENT_DIR}/../../build/src/cpu/pthread_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg ${num_cores}
  echo ""
done

# OpenMP PartA
echo "OpenMP PartA (Optimized with -O2)"
for num_cores in 1 2 4
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores perf stat ${CURRENT_DIR}/../../build/src/cpu/openmp_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
  echo ""
done

# CUDA PartA
echo "CUDA PartA"
srun -n 1 --gpus 1 nsys profile -t cuda,nvtx,osrt -o ./profile/cuda_PartA.qdrep ${CURRENT_DIR}/../../build/src/gpu/cuda_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
srun -n 1 perf stat ./profile/cuda_PartA.qdrep
echo ""

# OpenACC PartA
echo "OpenACC PartA"
srun -n 1 --gpus 1 nsys profile -t cuda,nvtx,osrt -o ./profile/openacc_PartA.qdrep ${CURRENT_DIR}/../../build/src/gpu/openacc_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
srun -n 1 perf stat ./profile/openacc_PartA.qdrep
echo ""

# Triton PartA
echo "Triton PartA"
srun -n 1 --gpus 1 python3 ${CURRENT_DIR}/../gpu/triton_PartA.py ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Gray.jpg
echo ""
