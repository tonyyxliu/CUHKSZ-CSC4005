# Get the current directory
CURRENT_DIR=$(pwd)
echo "Current directory: ${CURRENT_DIR}"

# Compile the mlp code
g++ ./code/mlp_main.cpp ./code/utlis.cpp ./code/mlp_network.cpp ./code/ops.cpp -O2 -o ./build/mlp
echo "Compiled the mlp code"