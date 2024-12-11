export TRITON_PTXAS_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/cuobjdump
export TRITON_NVDISASM_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/nvdisasm
export PATH=/opt/rh/rh-python38/root/usr/bin:$PATH

srun -n 1 --gpus 1 python3 ./triton/mlp_triton.py ./MINST