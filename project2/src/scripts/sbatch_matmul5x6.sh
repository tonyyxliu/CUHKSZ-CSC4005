#!/bin/bash
#SBATCH -o ./Project2-Matmul5x6.txt
#SBATCH -p Debug
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

CURRENT_DIR=$(pwd)
REPEAT=3

###########
## Naive ##
###########
echo "Naive MatMul (-O2)"
for i in $(seq 1 $REPEAT); do
    echo "Iteration $i..."
    srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/naive ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt
done
echo ""

#################
## Task 1: FMA ##
#################
echo "Naive MatMul with FMA (__restrict__ ptr)"
for i in $(seq 1 $REPEAT); do
    echo "Iteration $i..."
    srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/fma_restrict ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt
done
echo ""

echo "Naive MatMul with FMA (standalone var)"
for i in $(seq 1 $REPEAT); do
    echo "Iteration $i..."
    srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/fma_standalone_var ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt
done
echo ""

###########################
## Task 2: Constant Decl ##
###########################
echo "Naive MatMul with const ptr decl"
for i in $(seq 1 $REPEAT); do
    echo "Iteration $i..."
    srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/const_ptr_decl ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt
done
echo ""

#############################
## Task 3: Memory Locality ##
#############################

# Transposition
echo "MatMul with Transposition"
for i in $(seq 1 $REPEAT); do
    echo "Iteration $i..."
    srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/transpose ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt
done

echo ""

# Loop Re-ordering
echo "MatMul with Loop Re-ordering"
for i in $(seq 1 $REPEAT); do
    echo "Iteration $i..."
    srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/loop_interchange ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt
done
echo ""

####################
## Task 4: Tiling ##
####################

# Tiling + Transposition
echo "Tiling + Transposition"
for block_size in 8 16 32 64 128
do
  echo "BLOCK_SIZE: $block_size"
  for i in $(seq 1 $REPEAT); do
    echo "Iteration $i..."
    srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/tiling_transpose $block_size ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt
  done
  echo ""
done

# Tiling + Loop-interchange
echo "Tiling + Loop interchange"
for block_size in 8 16 32 64 128
do
  echo "BLOCK_SIZE: $block_size"
  for i in $(seq 1 $REPEAT); do
    echo "Iteration $i..."
    srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/tiling_loop_interchange $block_size ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt
  done
  echo ""
done

################################
## Task 5: Auto Vectorization ##
################################
echo "Auto Vectorization with BLOCK_SIZE = 32"
block_size=32
for i in $(seq 1 $REPEAT); do
  echo "Iteration $i..."
  srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/src/autovec $block_size ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt
done
echo ""

####################
## Task 6: OpenMP ##
####################
echo "OpenMP + Everything (BLOCK_SIZE = 32)"
block_size=32
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  for i in $(seq 1 $REPEAT); do
    echo "Iteration $i..."
    srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/build/src/openmp $num_cores $block_size ${CURRENT_DIR}/matrices/matrix5.txt ${CURRENT_DIR}/matrices/matrix6.txt
  done
  echo ""
done
