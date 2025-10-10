#!/bin/bash
#SBATCH -o ./Project1-PartB-Results-Profile.txt
#SBATCH -p Release
#SBATCH -J Project1-PartB-Profile
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

# Sequential PartB (Array-of-Structure)
echo "Sequential PartB (Array-of-Structure) (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 perf stat ${CURRENT_DIR}/../../build/src/cpu/sequential_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
echo ""
echo "Energy Profiling"
srun -n 1 --cpus-per-task 1 likwid-powermeter ${CURRENT_DIR}/../../build/src/cpu/sequential_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
echo ""

# Sequential PartB (Structure-of-Array)
echo "Sequential PartB (Structure-of-Array) (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 perf stat ${CURRENT_DIR}/../../build/src/cpu/sequential_PartB_soa ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
echo ""
echo "Energy Profiling"
srun -n 1 --cpus-per-task 1 likwid-powermeter ${CURRENT_DIR}/../../build/src/cpu/sequential_PartB_soa ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
echo ""

# SIMD PartB
echo "SIMD(AVX2) PartB (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 perf stat ${CURRENT_DIR}/../../build/src/cpu/simd_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
echo ""
echo "Energy Profiling"
srun -n 1 --cpus-per-task 1 likwid-powermeter ${CURRENT_DIR}/../../build/src/cpu/simd_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
echo ""

# MPI PartB
echo "MPI PartB (Optimized with -O2)"
for num_processes in 1 2 4
do
  echo "Number of processes: $num_processes"
  srun -n $num_processes --cpus-per-task 1 --mpi=pmi2 perf stat ${CURRENT_DIR}/../../build/src/cpu/mpi_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
  echo ""
  echo "Energy Profiling"
  srun -n $num_processes --cpus-per-task 1 --mpi=pmi2 likwid-powermeter ${CURRENT_DIR}/../../build/src/cpu/mpi_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
  echo ""
done

# Pthread PartB
echo "Pthread PartB (Optimized with -O2)"
for num_cores in 1 2 4
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores perf stat ${CURRENT_DIR}/../../build/src/cpu/pthread_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg ${num_cores}
  echo ""
  echo "Energy Profiling"
  srun -n 1 --cpus-per-task $num_cores likwid-powermeter ${CURRENT_DIR}/../../build/src/cpu/pthread_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg ${num_cores}
  echo ""
done

# OpenMP PartB
echo "OpenMP PartB (Optimized with -O2)"
for num_cores in 1 2 4
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores perf stat ${CURRENT_DIR}/../../build/src/cpu/openmp_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg ${num_cores}
  echo ""
  echo "Energy Profiling"
  srun -n 1 --cpus-per-task $num_cores likwid-powermeter ${CURRENT_DIR}/../../build/src/cpu/openmp_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg ${num_cores}
  echo ""
done

# CUDA PartB
echo "CUDA PartB"
srun -n 1 --gpus 1 nsys profile -t cuda,nvtx,osrt -o ./profile/cuda_PartB.qdrep ${CURRENT_DIR}/../../build/src/gpu/cuda_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
srun -n 1 perf stat ./profile/cuda_PartB.qdrep
echo ""

# OpenACC PartB
echo "OpenACC PartB"
srun -n 1 --gpus 1 nsys profile -t cuda,nvtx,osrt -o ./profile/openacc_PartB.qdrep ${CURRENT_DIR}/../../build/src/gpu/openacc_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
srun -n 1 perf stat ./profile/openacc_PartB.qdrep
echo ""

# Triton PartB
echo "Triton PartB"
srun -n 1 --gpus 1 python3 ${CURRENT_DIR}/../gpu/triton_PartB.py ${CURRENT_DIR}/../../images/kodim08_grayscale.png ${CURRENT_DIR}/../../images/kodim08_grayscale_blur.png ${CURRENT_DIR}/../../images/time_PartB.png
echo ""