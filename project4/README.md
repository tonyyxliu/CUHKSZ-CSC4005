# Project 4: Parallel Programming with Machine Learning

### This project weights 12.5% for your final grade (4 Projects for 50%)

### Release Date:

November 24th, 2024 (Beijing Time, UTC+08:00)

### Deadline:

11:59 P.M., December 13th, 2024 (Beijing Time, UTC+08:00)

### TA/USTF In Charge of This Assignment

**Mr. Zhang Qihang** for Implementation (qihangzhang@link.cuhk.edu.cn)

**Mr. Hou Tianci** for Implementation (tiancihou@link.cuhk.edu.cn)

## Prologue

In this project, you will have the opportunity to gain insight and practice in using OpenAcc & Triton to accelerate machine learning algorithms. Specifically, you will be accelerating MINST Handwritten Digit Recognition with neural networks.

First, you will need to understand the basic principles and algorithms of neural networks. Then, you will work with OpenAcc, a programming model for parallel computing that makes it easier for you to optimize your code to run on GPUs, thereby greatly increasing the speed of computation.

This assignment will help you understand the importance of parallel computing in machine learning, especially when working with large-scale data and complex models. You will learn how to effectively utilize hardware resources to improve the performance and efficiency of machine learning algorithms.

**REMIND: Please start ASAP to avoid the peak period of cluster job submission.**

## Task0: Setup

Download the dataset from BB or Internet. Unzip `dataset.zip` to folder `project4`. 

The structure of working directory should look like below:

```tree
.
├── CMakeLists.txt
├── MINST
│   ├── readme.txt
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
│   ├── train-images-idx3-ubyte
│   └── train-labels-idx1-ubyte
├── README.md
├── build
├── reference
│   └── mlp_pytorch.py
├── src
│   ├── mlp_main.cpp
│   ├── mlp_network.hpp
│   ├── mlp_openacc_fusion.cpp
│   ├── mlp_openacc_kernel.cpp
│   ├── mlp_sequential.cpp
│   ├── ops.hpp
│   ├── ops_openacc_fusion.cpp
│   ├── ops_openacc_kernel.cpp
│   ├── ops_sequential.cpp
│   ├── utils.cpp
│   ├── utils.hpp
├── test.sh
└── triton
    ├── mlp_triton.py
    └── ops
        ├── op_addbias.py
        ├── op_matmul.py
        ├── op_relu.py
        ├── op_relu_backward.py
        └── op_sum.py

```

## Task1: Train MNIST with MLP


The inference and training process of a neural network can be described by the following formulas:

1. **Forward Propagation (Inference)**
    The forward propagation process of a neural network can be described by the following formula, where $a^{(l)}$ is the activation value of the $l$ th layer $W^{(l)}$ is the weight of the $l$ th layer, $b^{(l)}$ is the bias of the $l$ th layer, and $f$ is the activation function:

    $$a^{(l)}=f(W^{(l)} a^{(l−1)}+b^{(l)})$$

    This process starts from the input layer, through the calculation of each layer’s weights and biases, as well as the activation function, and finally obtains the predicted value of the output layer.

2. **Backward Propagation (Training)**
   The training process of a neural network mainly updates the weights and biases through the backpropagation algorithm. First, we need to define a loss function $L$ to measure the gap between the predicted value and the true value. Then, we update the weights and biases by calculating the gradient of the loss function for the weights and biases:

$$ \frac{\partial{L}}{\partial{W^{(l)}}} = \frac{\partial{L}}{\partial{a^{(l)}}} \frac{\partial{a^{(l)}}}{\partial{W^{(l)}}} $$

$$\frac{\partial{L}}{\partial{b^{(l)}}} = \frac{\partial{L}}{\partial{a^{(l)}}} \frac{\partial{a^{(l)}}}{\partial{b^{(l)}}} $$

   Here, $\frac{\partial{L}}{\partial{a^{(l)}}}$ can be propagated from the next layer to the previous layer through the chain rule. Finally, we use the gradient descent method to update the weights and biases:

$$ W^{(l)} = W^{(l)} - \alpha \frac{\partial{L}}{\partial{W^{(l)}}} $$

$$ b^{(l)} = b^{(l)} - \alpha \frac{\partial{L}}{\partial{b^{(l)}}} $$

   Here, $\alpha$ is the learning rate, which controls the step size of the update.

In this project, we are going to implement a 2-layer NN with bias by using the SGD method. The target function is below:

$$ Z = W_2^T ReLU(W_1^T x+b_1)+b_2 $$

where $W_1 \in \mathbb{R}^{n \times d}$ and $W_2 \in \mathbb{R}^{d \times k}$ represent the weights of the network (which has a $d$-dimensional hidden unit), $b_1 \in \mathbb{R}^{d}$ and $b_2 \in \mathbb{R}^{k}$ represent the bias of the network, and where $z \in \mathbb{R}^k$ represents the logits output by the network. The formula of ReLU activation function is $f(x) = \max(0,x)$. We again use the softmax / cross-entropy loss, meaning that we want to solve the optimization problem.

Using the chain rule, you can derive the backpropagation updates for this network.

Here is the sample forward code in Python with `Pytorch`:

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        fc1_out = self.fc1(x)
        relu_out = self.relu(fc1_out)
        out = self.fc2(relu_out)
        return out
```

We already define **all functions** you need in the MLP in `ops_sequential.cpp` file (Also for OpenAcc Version). Also, the brief introduction and parameters for function are in `ops.hpp`.

Complete the code and use these function to do MLP training in `mlp_sequential.cpp`.

## Task2: Accelerate MLP with OpenAcc

First, copy your code in `ops_sequential.cpp` and `mlp_sequential.cpp` to OpenAcc version code! We have hint in code files for what you need to change for different sub-task.

### 2.1 Accelerate with Kernel

Implement a more flexible and easily extensible OpenAcc acceleration approach. This means using `#pragma acc` for each kernel call in the MLP training code structure, which may result in frequent communication between the CPU and GPU for kernel computation results.

### 2.2 Accelerate with Fusion

Implement an OpenAcc acceleration approach that minimizes GPU communication. This involves transferring all necessary data to the GPU at the start of the MLP training process and using a single `#pragma acc`. 

Please implement different methods and measure their execution times, then **analyze** the time differences.

## Task3: Train MLP with Triton

Task 3 is **optional for undergraduates but compulsotry for postgraduates**.

You can use Triton to overload functions for backpropagation, thereby leveraging PyTorch's autograd functionality to train neural networks.

**NOTICE**: All opreation should not directly use the pytorch function!

## Extra Credit: Extend Neural Network to Convolutional Neural Network with&without OpenAcc

You need to implement and accelerate the  `conv` function in `ops_sequential.cpp` with&without OpenAcc. Copy and rename the `mlp_main.cpp` and `mlp_XXX.cpp` to run the CNN training.

You can use any hyperparameters and filters as you like.

## How to Execute the Program

### Compilation

```bash
cd /path/to/project4
mkdir build && cd build
cmake ..
make
```

Compilation with `cmake` may fail in docker container, if so, please compile with `gcc` and `pgc++` in the terminal with the correct optimization options.

### Job Submission

Execute the bash script.

Make sure you are in the `project4` dir.

```bash
sbatch ./test.sh
```

## Baseline

For **1 epoch** with **400** hidden layer:

| MLP Sequential | MLP OpenAcc(kernel) | MLP OpenAcc(fusion) |
| -------------- | ------------------- | ------------------- |
| ~50000 ms      | ~9000 ms            | ~6500 ms            |

## Appendix

### About acc rate

If you use the same random seed for init the network, the acc rate will be:

(The output for `hidden layer: 400`, `learning rate: 0.001`, `batch number=32`)

```
Sequential (Optimized with -O2)
Training two layer neural network 400 hidden units
| Epoch |  Acc Rate  |  Training Time
|     1 |   92.330%  |   47477 ms
|     2 |   93.950%  |   47482 ms
|     3 |   95.000%  |   47474 ms
|     4 |   95.730%  |   47448 ms
|     5 |   96.310%  |   47392 ms
|     6 |   96.570%  |   47449 ms
|     7 |   96.870%  |   47403 ms
|     8 |   97.090%  |   47437 ms
|     9 |   97.300%  |   47469 ms
|    10 |   97.400%  |   47368 ms
Execution Time: 519906 milliseconds

OpenACC kernel
Training two layer neural network 400 hidden units
| Epoch |  Acc Rate  |  Training Time 233
|     1 |   92.330%  |   9312 ms
|     2 |   93.950%  |   8220 ms
|     3 |   94.970%  |   8243 ms
|     4 |   95.720%  |   8236 ms
|     5 |   96.300%  |   8279 ms
|     6 |   96.560%  |   8288 ms
|     7 |   96.870%  |   8336 ms
|     8 |   97.100%  |   8303 ms
|     9 |   97.260%  |   8303 ms
|    10 |   97.400%  |   8358 ms
Execution Time: 87501 milliseconds

OpenACC fusion
Training two layer neural network 400 hidden units
| Epoch |  Acc Rate  |  Training Time
|     1 |   92.330%  |   7379 ms
|     2 |   93.950%  |   6205 ms
|     3 |   94.970%  |   6213 ms
|     4 |   95.720%  |   6251 ms
|     5 |   96.300%  |   6251 ms
|     6 |   96.560%  |   6298 ms
|     7 |   96.870%  |   6326 ms
|     8 |   97.100%  |   6312 ms
|     9 |   97.260%  |   6340 ms
|    10 |   97.400%  |   6354 ms
Execution Time: 67134 milliseconds
```

Some student may get the result 
```
| Epoch |  Acc Rate  |  Training Time
|     1 |   79.470%  |
|     2 |   84.470%  |
|     3 |   86.530%  |
|     4 |   87.950%  |
|     5 |   88.850%  |
|     6 |   89.360%  |
|     7 |   89.650%  |
|     8 |   89.950%  |
|     9 |   90.310%  |
|    10 |   90.550%  |
```

This is because, during the implementation, I accounted for the effect of batch size on accuracy by default. If you have any questions, please refer to [this Chinese blog](https://kexue.fm/archives/10542). For students who obtained the above results, please multiply the learning rate by the batch size, and you should achieve a result close to 92%(like first table). If you followed our initialization method and obtained such a result, **you don’t need to worry about losing points**!

## Requirements & Grading Policy

- **Machine Learning (50%)**
  
  - Task1: Train MLP with CPP (20%)
  - Task2: Accelerate MLP with OpenAcc (20%)
    - 2.1: Accelerate by kernel (10%)
    - 2.2: Accelerate by fusion (10%)
  - Task3: Train MLP with Triton (10% optional for undergraduates but compulsotry for postgraduates)

  Your programs should be able to compile & execute to get the expected computation result to get the full grade in this part.
  
- **Performance of Your Program (30%)**

  - 10% for Task 1
  - 15% for Task 2
    - 7.5% for 2.1
    - 7.5% for 2.2
  - 5% for Task 3 (optional for undergraduates but compulsotry for postgraduates)

  Try your best to do optimization on your parallel programs for higher speedup. If your programs show similar performance to the baseline performance, then you can get the full mark for this part. Points will be deducted if your parallel programs perform poorly while no justification can be found in the report.

- **One Report in PDF (20%, No Page Limit)**

  - **Regular Report (10%)**
    The report does not have to be very long and beautiful to help you get a good grade, but you need to include what you have done and what you have learned in this project. The following components should be included in the report:
    - How to compile and execute your program to get the expected output on the cluster.
    - Explain clearly how you designed and implemented each algorithm
    - Show the experiment results you get, and do some numerical analysis, such as calculating the speedup and efficiency, demonstrated with tables and figures.
    - What kinds of optimizations have you tried to speed up your parallel program, and how do they work?
    - Any interesting discoveries you found during the experiment?
  
  - **Profiling OpenAcc with nsys (10%)**
    You are required to practice profiling OpenAcc programs with nsys as we explained in the [Instruction of profiling tools with perf and nsys](https://github.com/tonyyxliu/CSC4005-2023Fall/blob/main/docs/Instruction%20on%20Profiling%20with%20perf%20and%20nsys.md#nsys-for-gpu-profiling). The command line profiling of nsys is mandatory while the GUI Nsight System is optional.

- **Extra Credits (Max 10%)**
  - Train MLP by Triton (5%, only for undergraduates)
  - Train CNN with&without OpenAcc(10%)

  Extra optimizations or interesting discoveries in the first three tasks may also earn you some extra credits.

  Please write what you do for Extra Credits **in your report**!

### The Extra Credit Policy

According to the professor, the extra credits in this project cannot be added to other projects to make them full marks. The credits are the honor you received from the professor and the teaching staff, and the professor may help raise you to a higher grade level if you are at the boundary of two grade levels and he thinks you deserve a better grade with your extra credits. For example, if you are among the top students with B+ grade, and get enough extra credits, the professor may raise you to A- grade. Furthermore, the professor will invite a few students with high extra credits to have dinner with him.

### Grading Policy for Late Submission

1. late submission for less than 10 minutes after the DDL is tolerated for possible issues during submission.
2. 10 Points deduction for each day after the DDL (11 minutes late will be considered as one day, so be careful)
3. Zero points if you submitted your project late for more than two days
If you have some special reasons for late submission, please send an email to the professor and cc TA Liu Yuxuan.

### File Structure to Submit on BlackBoard

Do not upload build file, bash file and dataset.

Only Code and Report Needed!

```bash
<Your StudentID>.pdf  # Report
<Your StudentID>.zip  # Codes
├── src
│   ├── mlp_main.cpp
│   ├── mlp_network.hpp
│   ├── mlp_openacc_fusion.cpp
│   ├── mlp_openacc_kernel.cpp
│   ├── mlp_sequential.cpp
│   ├── ops.hpp
│   ├── ops_openacc_fusion.cpp
│   ├── ops_openacc_kernel.cpp
│   ├── ops_sequential.cpp
│   ├── utils.cpp
│   └── utils.hpp
└── triton(If needed)
    ├── mlp_triton.py
    └── ops
        ├── op_addbias.py
        ├── op_matmul.py
        ├── op_relu.py
        ├── op_relu_backward.py
        └── op_sum.py
```
