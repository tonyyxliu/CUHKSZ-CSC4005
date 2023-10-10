import random

row_num = 100
col_num = 100

maximum_limit = 100.0

with open("matrix.txt", "w") as matrix_file:
    matrix_file.write(f"{row_num} {col_num}\n")
    for i in range(row_num):
        for j in range(col_num):
            random_number = int(random.uniform(0.0, maximum_limit))
            if j < col_num - 1:
                matrix_file.write(f"{random_number} ")
            else:
                matrix_file.write(f"{random_number}\n")
            
