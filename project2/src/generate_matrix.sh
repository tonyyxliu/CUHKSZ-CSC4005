#!/bin/bash
## Author: Liu Yuxuan
## Email: yuxuanliu1@link.cuhk.edu.cn
## Last modified on 2025.10.12

mkdir -p ../matrices

python3 ./generate_matrix.py 4 ../matrices/matrix1.txt
python3 ./generate_matrix.py 4 ../matrices/matrix2.txt
python3 ./generate_matrix.py 128 ../matrices/matrix3.txt
python3 ./generate_matrix.py 128 ../matrices/matrix4.txt
python3 ./generate_matrix.py 1024 ../matrices/matrix5.txt
python3 ./generate_matrix.py 1024 ../matrices/matrix6.txt
python3 ./generate_matrix.py 2048 ../matrices/matrix7.txt
python3 ./generate_matrix.py 2048 ../matrices/matrix8.txt
