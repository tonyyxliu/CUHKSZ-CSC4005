# Project 3: Parallel Sorting/Searching Algorithms

## This project weights 12.5% for your final grade (4 Projects for 50%)

### Release Date:

November 3rd, 2025 (Beijing Time, UTC+08:00)

### Deadline:

11:59 P.M., November 28th, 2025 (Beijing Time, UTC+08:00)

### Teaching Stuff In Charge of This Assignment

TA Mr. Liu Yuxuan (刘宇轩先生) (yuxuanliu1@link.cuhk.edu.cn)

USTF Mr. Fang Zihao (方梓豪先生) (zihaofang1@link.cuhk.edu.cn)

## Introduction

Sorting and searching algorithms are of great importance in computer science. In data structure and algorithm courses, we have learned about various sorting and searching algorithms. However, have you ever considered making these algorithms run in parallel on multi-core CPUs or even running them on GPUs? In fact, parallel sorting and searching algorithms are more challenging than the projects we have completed for two main reasons. Firstly, it is not that easy to divide the entire task into different threads. Secondly, the work of different threads is no longer independent, which means that they must synchronize their states and ensure that the algorithms run correctly.

In this project, we have prepared three classical sorting algorithms and the binary search algorithm for you. You need to implement them with a step-by-step instruction from easy to hard, and finally achieve both CPU and GPU sorting/searching. Get started and do your best!

1. **Parallel Merge Sort with Parallel Merging on CPU**
2. **Parallel Quick Sort with Parallel Partitioning on CPU**
3. **Parallel Radix Sort on GPU**
4. **Parallel Multi-Data Binary Searching on CPU**
5. **Parallel Multi-Data Binary Searching on GPU**

You need to use **OpenMP** for all CPU tasks and **OpenACC** for all GPU tasks. You can use C++ features up to C++17. Make sure that you don't modify the CMake files.

## Task #1: Parallel Merge Sort with Parallel Merging on CPU

To achieve parallel divide and conquer, the number of threads should be dynamically determined during the run-time, and the OpenMP tasking directive can help.

In this task, merge sort is chosen as a classic sorting algorithm applying divide and conquer philosophy. It is a highly efficient and widely used sorting algorithm in computer science. It is known for its ability to sort large datasets with excellent time complexity and stability. Merge Sort employs a divide-and-conquer strategy, which involves breaking down the unsorted list into smaller sub-lists, sorting each sub-list, and then merging the sorted sub-lists to obtain the final sorted result.

Here's a brief overview of how Merge Sort works:

1. **Divide**: The unsorted list is divided into two equal-sized sub-lists until each sub-list contains only one element. This process continues recursively.
2. **Conquer**: The one-element sub-lists are considered sorted by default. Then, the adjacent sub-lists are merged together. During the merge process, elements are compared and rearranged in a way that ensures they are in the correct order.
3. **Combine**: The merging and sorting process continues until all sub-lists are merged into a single, fully sorted list. This final list contains all the elements from the original list, sorted in ascending order.

![Merge Sort](../docs/images/merge-sort.png)

```c++
// Merge two subarrays of vector vec[]
// First subarray is vec[l..m]
// Second subarray is vec[m+1..r]
void merge(std::vector<int>& vec, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary vectors
    std::vector<int> L(n1);
    std::vector<int> R(n2);

    // Copy data to temporary vectors L[] and R[]
    for (int i = 0; i < n1; i++) {
        L[i] = vec[l + i];
    }
    for (int i = 0; i < n2; i++) {
        R[i] = vec[m + 1 + i];
    }

    // Merge the temporary vectors back into v[l..r]
    int i = 0; // Initial index of the first subarray
    int j = 0; // Initial index of the second subarray
    int k = l; // Initial index of the merged subarray

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            vec[k] = L[i];
            i++;
        } else {
            vec[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        vec[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        vec[k] = R[j];
        j++;
        k++;
    }
}

// Main function to perform merge sort on a vector v[]
void mergeSort(std::vector<int>& vec, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        // Sort first and second halves
        mergeSort(vec, l, m);
        mergeSort(vec, m + 1, r);

        // Merge the sorted halves
        merge(vec, l, m, r);
    }
}
```

Above is the sequential version of merge sort, and your task is to implement a **thread-level parallel merge sort with parallel merging** in `src/mergesort/mergesort.cpp`. You need to use OpenMP to complete the template.

### OpenMP Tasking

The `task` pragma can be used in OpenMP to explicitly define a task. According to [IBM Documentation](https://www.ibm.com/docs/en/zos/2.4.0?topic=processing-pragma-omp-task), you can use the task pragma when you want to identify a block of code to be executed in parallel with the code outside the task region. The task pragma can be useful for parallelizing irregular algorithms such as pointer chasing or recursive algorithms. The task directive takes effect only if you specify the SMP compiler option.

#### Hints:

It may feel very difficult to divide the task. In fact, there are two places where you could assign the task to different threads.

1. When calling `mergeSort` recursively, it's quite natural to consider assigning the two `mergeSort` operations to two different threads to increase parallelism. This approach is indeed effective. However, you should also **maintain control over the total number of threads** in your program. Creating and destroying threads isn't inexpensive, so creating new threads for very lightweight tasks is unnecessary. Additionally, if the number of threads exceeds the number of CPU cores, introducing new threads may even result in performance degradation. Therefore, it's advisable to implement a mechanism within your recursive function to control the total number of threads.

2. `merge` function is the core of merge sort. If `merge` isn't parallel, the parallelism of merge sort is limited. However, making the `merge` function parallel isn't that easy. Here are some guidance for you to understand how to make `merge` parallel.

   1. Firstly, let's discuss the simplest situation: how to make two threads cooperate to execute `merge`. The `merge` function combines two sorted arrays into a larger sorted array. Let's consider the following:

      1. The middle point of the resulting larger sorted array.
      2. For the first sorted array, which elements are less (or equal) to the middle point, and which elements are larger than the middle point.
      3. For the second sorted array, which elements are less (or equal) to the middle point, and which elements are larger than the middle point.

      If we know all of these, we can assign the first thread the responsibility of merging the first half of the array and the second thread the responsibility of merging the second half.

   2. If you grasp the high-level concept explained above, the next step is to determine how to find the middle point of the resulting larger sorted array. This process essentially involves an algorithm for finding the median point between two sorted arrays. You may refer to [LeetCode 4](https://leetcode.com/problems/median-of-two-sorted-arrays/) for learning and practice. It's important to note that your algorithm should have a time complexity of O(log(m+n)) to avoid significant performance degradation. Additionally, once you completely understand the algorithm, you'll realize that determining which elements of the two subarrays are less than, equal to, or larger than the middle points is a natural outcome of the algorithm.
   3. When it comes to using more threads, the general idea remains very similar. If you have k threads, you should first find the points at (1/k), (2/k), and so on, up to ((k-1)/k) of the resulting merged array (the algorithm should be almost the same as in the case of 2 threads). Afterward, you can divide the merging tasks and assign them to different threads.
   4. Remember that creating and destroying threads is costly, and parallelizing the `merge` operation incurs additional overhead (e.g., finding the middle points) when compared to the sequential version. Therefore, you should consider whether it's worthwhile to use parallel `merge` based on the task size and current threads number. For instance, it is entirely unnecessary to assign multiple threads to merge [0] and [1].

## References:

[OpenMP Task Basics](https://hpc2n.github.io/Task-based-parallelism/branch/master/task-basics-1/)

[QuickSort with OpenMP by Mohd Ehtesham Shareef from The State University of New York](https://cse.buffalo.edu/faculty/miller/Courses/CSE702/Mohd-Ehtesham-Shareef-Fall-2020.pdf)

[Advanced Programming with OpenMP (Quick Sort as one Example)](https://cw.fel.cvut.cz/old/_media/courses/b4m35pag/lab6_slides_advanced_openmp.pdf)

[Medium: Parallel QuickSort using OpenMP](https://mcbeukman.medium.com/parallel-quicksort-using-openmp-9d18d7468cac)

[SC'13 Talk on OpenMP Tasking](https://openmp.org/wp-content/uploads/sc13.tasking.ruud.pdf)

[OpenMP Tutorial by Blaise Barney from Lawrence Livermore National Laboratory](https://hpc-tutorials.llnl.gov/openmp/)

## Task #2: Parallel Quick Sort with Parallel Partitioning on CPU

 To parallelize quick sort, we simply create a thread for each quick-sort recursive function call and execute sequential quick-sort in each process. However, it is actually possible to make the quick-sort computation part itself go in parallel to get extra speedup. The solution is to do dynamic thread creation to deal with the recursive function call during sorting. To implement this dynamic thread creation, you can use the OpenMP task pragma as you did in Task 1. Note that you also need to make the partitioning process parallel to gain full points.

 Your task is to implement a **thread-level parallel quick sort with parallel partitioning** in `src/quicksort/quicksort.cpp`. You need to use OpenMP to complete the template.

#### Hints:

It may feel very difficult to divide the task. However similar to merge sort, there are two places where you could assign the task to different threads.

1. When calling `quickSort` recursively, you could assign new threads to increase parallelism. However, as with the same requirement for `mergeSort`, you should **maintain control over the total number of threads** in your program.
2. Make the `partition` operation parallel. The high-level idea of the `partition` is to select a pivot and divide the array into two subarrays: one containing elements less than the pivot, and the other containing elements larger than the pivot. The sequential version of `partition` operation achieves the purpose without introducing any extra space. However, in the parallel version, you can achieve the same goal but **using extra space** to allow for a parallel process.  There are various methods to achieve this. If you don't have your own approach, you can refer to this [Video](https://www.youtube.com/watch?v=yD_pg34xhIs&t=340s). If you decide to follow the algorithm in the video, you will also need to understand how to implement a **parallel PrefixSum algorithm**. You may visit to the [Wiki](https://en.wikipedia.org/wiki/Prefix_sum) on PrefixSum for a reference. 
3. Remember that creating and destroying threads is costly, and parallelizing the `partition` operation incurs additional overhead (e.g., PrefixSum) when compared to the sequential version. Therefore, you should consider whether it's worthwhile to use parallel `partition` based on the task size and current threads number . For instance, it is entirely unnecessary to assign multiple threads for the partition of [0, 1].

## Task #3: Parallel Radix Sort on GPU

Radix sort is a non-comparative sorting algorithm that distributes elements into buckets based on the value of their individual digits. By sorting the elements digit by digit, starting from the least significant to the most significant, radix sort eventually produces a fully sorted array.

Here is a step-by-step explaination of how radix sort works:

1. **Find the Largest Element**: The first step is to identify the largest element in the array. This allows us to determine how many digits we need to process.
2. **Sort by Digits**: The array is sorted based on individual digits, starting from the least significant digit (LSD) to the most significant digit (MSD). For example, given the array \([805, 122]\), we first sort by the unit digits ('5' and '2'), resulting in \([122, 805]\). This digit-based sorting is typically done using **counting sort**.
3. **Repeat**: The process is repeated for each digit, moving from the least significant to the most significant digit, until the entire array is sorted.

```c++
#define BASE 256
#define BASE_BITS 8

void radixSort(std::vector<int> &vec) {
    int n = vec.size();

    // Get the pointer of element
    int *vec_raw = vec.data();

    // Counting sort for each digit
    int *output = new int[n];
    int count[BASE];
    int start_pos[BASE];
    int offset[BASE];

    for (int shift = 0; shift < 32; shift += BASE_BITS) {
        
        memset(count, 0, sizeof(int) * BASE);
        for (int i = 0; i < n; i++)
            count[(vec_raw[i] >> shift) & (BASE - 1)]++;

        memset(start_pos, 0, sizeof(int) * BASE);
        for (int d = 1; d < BASE; d++) {
            start_pos[d] = start_pos[d - 1] + count[d - 1];
        }

        memset(offset, 0, sizeof(int) * BASE);
        for (int i = 0; i < n; i++) {
            int digit = (vec_raw[i] >> shift) & (BASE - 1);
            int pos = start_pos[digit] + offset[digit];
            output[pos] = vec_raw[i];
            offset[digit]++;
        }

        // Assign elements back to the input vector
        memcpy(vec_raw, output, sizeof(int) * n);
    }
    delete[] output;
}
```

The sequential version of radix sort has been provided in `src/radixsort-gpu/sequential.cpp`, and your task is to implement parallel radix sort using **OpenACC** in `src/radixsort-gpu/radixsort.cpp`.

In this task, please implement a radix sort for GPU accelerator in OpenACC which could support sorting for an array of length $N = 10^8$. 

#### Requirements

You need to learn about how to implement a OpenACC scan, reduce, and radix sort efficiently in this task. You can apply decoupled look-back method to implement your code. 

**NOTE:** To obtain the credit, you need to report your program structure (the histogram/scan/reduce/scatter/... passes of different levels in your algorithm and how they are organized) associated with parallel methods (e.g. reduce-then-scan, scan-scan-add, decoupled look-back), and your speedup factor comparing with `std::sort`. 

Related techniques about OpenACC programming (including Q&A of this task) is planning to be introduced in the tutorial on the 15th of November. You can self-study on OpenACC implementation of scan, reduce, and radix sort by yourself in advance to complete this task. 

#### References 

You can take reference on the following links to complete this task. 

[OpenACC 3.3 Specification](https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.3-final.pdf)

[UMich EECS 570: Parallel Prefix Sum (Scan) with CUDA](https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf)

[UC Davis CS 223 Guest Lecture: GPU Reduce, Scan, & Sort](https://www.cs.ucdavis.edu/~amenta/f15/amenta-reduce-scan-sort.pdf)

[Nvidia 2016: Single-pass-Parallel-Prefix with Decoupled Look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf)

[Onesweep: A Faster Least Significant Digit Radix Sort for GPUs](https://arxiv.org/pdf/2206.01784)

[Nvidia GTC 2020: A Faster Radix Sort Implementation - video](https://developer.nvidia.com/gtc/2020/video/s21572-vid)

[Nvidia GTC 2020: A Faster Radix Sort Implementation - slides](https://developer.download.nvidia.cn/video/gputechconf/gtc/2020/presentations/s21572-a-faster-radix-sort-implementation.pdf)

[Thrust/CUB Library Implementation](https://github.com/thrust/cub/tree/master/cub)s

## Task #4: Parallel Searching for Data Array on CPU

This task focuses on a practical scenario where we need to search for many different targets simultaneously. This is a common pattern in database systems, data analytics, and scientific computing applications.

In many practical applications, we need to search for multiple different targets in a large sorted array. This task handles an **array of different search targets**. You will search for many different values in parallel. Given a sorted array of 200,000,000 elements and a search target array of 20,000,000 targets (10% of the array size), your goal is to efficiently find the index of each target using parallel processing. 

Here's the sequential approach for searching multiple targets:

```c++
// Sequential search for multiple targets
for (int i = 0; i < search_size; i++) {
    results[i] = binarySearch(vec, search_targets[i]);
}
```

Above is the sequential version for searching multiple targets, and your task is to implement **parallel binary search for data array** in `src/searching-cpu/parallel_array.cpp`. Both the data array and the array to be searched are sorted. You need to use OpenMP to complete the template. To be correct, your implementation should yield the same results as `std::lower_bound`.

#### Hints:

1. **Exploiting Sorted Targets**: Since search targets are sorted, maintain a "hint" position from the previous search. Use exponential search (steps 1, 2, 4, 8, ...) to quickly narrow the search range.

2. **Cache Optimization**: Process multiple searches in synchronized batches to improve cache locality. Searches in a batch access similar memory regions. 

3. **Algorithm vs Parallelization**: Good algorithm design can outperform simple parallelization by 40-50%. Consider data characteristics (sorted targets) beyond just thread-level parallelism.

## Task #5: Parallel Searching for Data Array on GPU
This task is a GPU version of task #4. However, if you simply port the CPU version to GPU, you will find the execution speed extremely slow. This is because GPU is unable to deal with branch divergence well. That means, if different threads in one Warp (Streaming Multiprocessor) takes different branches, they will have to execute sequentially. Therefore, it greatly affects the performance.

Here is a naïve GPU version of binary search:

```c++
#pragma acc data copyin(vec[0:n], search_targets[0:search_size]) copyout(results[0:search_size])
#pragma acc parallel loop
for (int i = 0; i < search_size; i++) {
    results[i] = binarySearch(vec, n, search_targets[i]);
}
```
You need to perform optimizations to the code above and possibly the `binarySearch` function to make the performance better.

Your task is to implement **parallel binary search for data array** in `src/searching-gpu/array_gpu.cpp`. Both the data array and the array to be searched are sorted, as in task 4. You need to use OpenACC to complete the template. To be correct, your implementation should yield the same results as `std::lower_bound`.
#### Hints:
1. **Cache Optimization**: Process multiple searches in synchronized batches to improve cache locality. Searches in a batch access similar memory regions. Also, use the `register` statement to avoid unnecessary memory operations for some variables.

2. **Eliminate Branch Divergence**: Use conditional moves and ternary operators to avoid branching in GPU. Also, make smart use of arithmetic operations. For example, the following code snippet:
```c++
if (a < b) {
    sum += value;
}
```
can be turned into:
```c++
sum += (a < b) * value;
```

3. **Manual Task Scheduling**: Finding OpenACC schedule too much for you? Want the freedom and performance of CUDA? You don't need to switch back to CUDA! In OpenACC, you can still schedule works by yourself, by using `parallel` construct (Not `parallel loop`), and identify the number of gangs, workers, or vectors. Then, you can use `__pgi_gangidx()`, `__pgi_workeridx()`, and `__pgi_vectoridx()` inside the construct to get the ID of the current execution unit. Also, `__pgi_blockidx()` and `__pgi_threadidx()` can provide compabilities with CUDA.

#### References 

You can take reference on the following links to complete this task. 

[SIMD/GPU Friendly Binary Search](https://blog.demofox.org/2017/06/20/simd-gpu-friendly-branchless-binary-search/)

[Parallel Search on Video Cards](https://www.usenix.org/legacy/event/hotpar09/tech/full_papers/kaldeway/kaldeway.pdf)

[Thrust/CUB Library Implementation](https://github.com/thrust/cub/tree/master/cub)

## Requirements & Grading Policy

- **Parallel Sorting and Searching Algorithms (50%)**
  - Task #1: Parallel Merge Sort with Parallel Merging on CPU (10%)
    - Parallel Merge Sort (5%)
    - Parallel Merging (5%)
  - Task #2: Parallel Quick Sort with Parallel Partitioning on CPU (10%)
    - Dynamic Parallel Quick Sort (5%)
    - Parallel Partitioning (5%)
  - Task #3: Parallel Radix Sort on GPU (10%)
  <!-- - Task #4: Parallel Searching for Single Data on CPU (5%) - COMMENTED OUT -->
  - Task #4: Parallel Searching for Data Array on CPU (10%)
  - Task #5: Parallel Searching for Data Array on GPU (10%)

  Your programs should be able to compile & execute to get the expected computation result to get full grade in this part.

- **Performance of Your Program (30%)**
  6% for each task (5 tasks total), in total 30%.

  Try your best to optimize your parallel programs for higher speedup. If your programs shows similar performance to the baseline, then you can get full mark for this part. Points will be deduted if your parallel programs perform poorer while no justification can be found in your report.

- **One Report in PDF (20%, No Page Limit)**
  - **Regular Report (10%)**
  
      The report does not have to be very long and beautiful to help you get good grade, but you need to include what you have done and what you have learned in this project. The following components should be included in the report:

    - How to compile and execute your program to get the expected output on the cluster.
    - Explain clearly how did you design and implement each parallel sorting algorithm?
    - Show the experiment results you get, and do some numerical analysis, such as calculating the speedup and efficiency, demonstrated with tables and figures.
    - What kinds of optimizations have you tried to speed up your parallel program, and how do them work?
    - Any interesting discoveries you found during the experiment?

  - **Profiling Results & Analysis with `perf` and `nsys` (10%)**
  
      Please follow the [Instruction on Profiling with perf and nsys](https://github.com/tonyyxliu/CSC4005-2023Fall-internal/blob/main/docs/Instruction%20on%20Profiling%20with%20perf%20and%20nsys.md) to profile all of your parallel programs for the four tasks with `perf`, and do some analysis on the profiling results before & after the implementation or optimization. For example, you can use the profiling results from `perf` to do quantitative analysis that how many cache misses or page faults can be reduced with your optimization. Always keep your mind open, and try different profiling metrics in `perf` and see if you can find any interesting thing during experiment.
      
      **Note:** The raw profiling results may be very long. Please extract some of the useful items to show in your report, and remember to carry all the raw profiling results for your programs when you submit your project on BB.

- **Extra Credits (10%)**
We don't have any explicit instructions for extra credits. You may do anything you like related to this project. Be sure to include how to execute your bonus part and implementation details in your report. Here are some directions for your reference if you don't know what to do:
  - Optimize the programs for much better performance. 
  - Use CUDA for GPU tasks.
  - Use Triton for GPU tasks.
  - Anything else!
  - <span style='color:red'>IMPORTANT WARNING: Using CUDA or Triton for GPU tasks DOES NOT mean that you can omit the basic OpenACC part. Otherwise, you'll automatically get a 0 for this part and the extra credits will also not be given.</span>


### The Extra Credit Policy

According to the professor, the extra credits in this project cannot be added to other projects to make them full mark. The credits are the honor you received from the professor and the teaching stuff, and the professor may help raise you to a higher grade level if you are at the boundary of two grade levels and he think you deserve a better grade with your extra credits. For example, if you are the top students with B+ grade, and get enough extra credits, the professor may raise you to A- grade. Furthermore, professor will invite a few students with high extra credits to have dinner with him and all other teaching stuff.

### Grading Policy for Late Submission

1. Late submission for less than 15 minutes after the deadline is tolerated for possible issues during submission.
2. 10 Points deduction for each day after the deadline (16 minutes late will be considered as one day, so be careful)
3. Zero point if you submitted your project late for more than two days
If you have some special reasaons for late submission, please send email to the professor and c.c to TA Liu Yuxuan and USTF Fang Zihao.

### File Structure to Submit on BlackBoard

```bash
122090330.pdf  # Report
122090330.zip  # Codes
|-
|--- src/           # Where your source codes lie in
|--- CMakeLists.txt # Root CMakeLists.txt
|-
|--- profiling/     # Where your perf profiling raw results lie in
|--- bonus/         # (Optional) Bonus parts
```

## How to Execute the Program

### Compilation

```bash
cd /path/to/project3
mkdir build && cd build
# Change to -DCMAKE_BUILD_TYPE=Debug for debug build error message logging
# Here, use cmake on the cluster and cmake3 in your docker container
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

Compilation with `cmake` may fail in docker container, if so, please compile with `gcc`, `mpic++`, `nvcc` and `pgc++` in the terminal with the correct optimization options.

### Job Submission

Please submit the job with `sbatch.sh` through `sbatch`.

```bash
# Use sbatch
cd /path/to/project3/src
sbatch ./sbatch.sh
```

## Performance Evaluation

### Correctness Verification

After performing your implemented sorting algorithm, we will compare your sorting results with the `std::sort` function, both for correctness and performance. If your sorting results are correct, you will see the output suggesting that you have passed the test, as follows:

```
Quick Sort Complete!
Execution Time: 0 milliseconds
std::sort Time: 0 milliseconds
Pass the sorting result check!
```

Otherwise, it will point out where your sorting results are incorrect. Please ensure the correctness of your program, as failing to do so will result in a loss of points.

```
Quick Sort Complete!
Execution Time: 0 milliseconds
std::sort Time: 0 milliseconds
Fail to pass the sorting result check!
4th element of the sorted vector is expected to be 5
But your 4th element is 2
```

### Performance Baseline

Before executing the sorting program, you should set the length of the vector as suggested in the previous section, and a random vector with the given size will be generated. 

**Experiment Setup**
- Vector Size for Sorting: 100,000,000 (1E+08)
- Vector Size for Searching: 200,000,000 (2E+08, larger for better benchmarking)
- Search Array Fraction: 10% (20,000,000 searches)

Here are the performance baselines (in milliseconds) for Project 3:

*Params: low = 0, high = SIZE*

| Workers | std::sort | Merge Sort | Quick Sort | Radix Sort | std::lower_bound (Array, CPU) | Searching (Array, CPU) | Searching (Array, GPU) |
| :-----: | :-------: | :-------------: | :------------: | :--------: | :---: |:----------------: | :-------------:|
|    1    | 10077     | 18587           | 12394          | 618        | 1880  | 2117          |     352     |
|    4    | 5031      | 8295            | 6913           | N/A        | N/A  | 645           |      N/A    |
|    8    | 2537      | 5276            | 4008           | N/A        | N/A  | 357           |      N/A    |
|   16    | 1288      | 2940            | 2796           | N/A        | N/A  | 212           |      N/A    |
|   32    | 696       | 1987            | 2259           | N/A        | N/A  | 150           |      N/A    |
