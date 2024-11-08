# Project 3: Distributed/Parallel Sorting Algorithms with MPI

## This project weights 12.5% for your final grade (4 Projects for 50%)

### Release Date:

October 30th，2024 (Beijing Time, UTC+08:00)

### Deadline:

11:59 P.M., November 17th, 2024 (Beijing Time, UTC+08:00)

### Teaching Stuff In Charge of This Assignment

TA Mr. Liu Yuxuan (刘宇轩先生) (yuxuanliu1@link.cuhk.edu.cn)

USTF Mr. You Lyu (吕游先生) (youlyu@link.cuhk.edu.cn)

USTF Mr. Yuan Xu (徐源先生) (yuanxu1@link.cuhk.edu.cn)

## Introduction

Sorting algorithms are of great importance in computer science. In data structure and algorithm courses, we have learned about various sorting algorithms. However, have you ever considered making these algorithms run in parallel on multi-core CPUs or even distributing them in a cluster? In fact, distributed and parallel sorting algorithms are more challenging than the projects we have completed for two main reasons. Firstly, it is not that easy to divide the entire task into different threads or processes. Secondly, the work of different threads or processes is no longer independent, which means that they must communicate with each other to synchronize their states and ensure that the algorithms run correctly.

In this project, we have prepared four classic sorting algorithms for you. You need to implement them with a step-by-step instruction from easy to hard, and finally achieve both multi-process and multi-thread parallelization of sorting. Get started and do your best!

1. **Process-Level Parallel Bucket Sort in MPI**
2. **Process-Level Parallel Quick Sort with K-Way Merge in MPI**
3. **Process-Level Parallel Sorting with Regular Sampling (PSRS) in MPI**
4. **Thread-Level Parallel Merge Sort in OpenMP**
5. (Optional) **Dynamic Thread-Level Parallel Quick Sort**
6. (Optional) **Parallel Radix Sort on CPU**
7. (Undergraduate Optional, Graduate Required) **GPU Radix Sort with Decoupled Look-back**

## Task #1: Process-Level Parallel Bucket Sort (Assume Uniformly Distributed Data)

Bucket Sort is a sorting algorithm that falls under the category of distribution sort algorithms. It is particularly useful when sorting a large number of elements with known or limited range values, such as integers or floating-point numbers. The primary idea behind Bucket Sort is to distribute elements into a finite number of buckets and then sort the individual buckets using another sorting algorithm, typically insertion sort. Once the buckets are sorted, their contents are concatenated to obtain the final sorted result. *However, it is intuitive that the expected speedup can only be achieved when the data size in each bucket is close to each other. Otherwise, load imbalance occurs and there will be performance degration.*

Here's a step-by-step explanation of how Bucket Sort works:

1. **Bucket Creation**: Create a fixed number of buckets (or containers) based on the range of input values. The range of values should be known or bounded in advance. For example, if you are sorting a list of integers between 1 and 100, you can create 100 buckets.
2. **Distribution**: Iterate through the input list and place each element into its corresponding bucket based on a mapping function. This mapping function should distribute elements uniformly among the buckets.
3. **Sort Each Bucket**: After all elements are distributed into the buckets, sort each individual bucket. This step can use any sorting algorithm, but typically, insertion sort is chosen because it's simple and efficient for small lists.
4. **Concatenation**: Finally, concatenate the sorted buckets in order to obtain the fully sorted list. The order of concatenation depends on the order of the buckets, and it typically follows the order of the buckets' indices.

![Bucket sort - Wikipedia](../docs/images/bucket-sort.png)

```c++
void insertionSort(std::vector<int>& bucket) {
    for (int i = 1; i < bucket.size(); ++i) {
        int key = bucket[i];
        int j = i - 1;

        while (j >= 0 && bucket[j] > key) {
            bucket[j + 1] = bucket[j];
            j--;
        }

        bucket[j + 1] = key;
    }
}

void bucketSort(std::vector<int>& vec, int num_buckets) {
    int max_val = *std::max_element(vec.begin(), vec.end());
    int min_val = *std::min_element(vec.begin(), vec.end());

    int range = max_val - min_val + 1;
    int small_bucket_size = range / num_buckets;
    int large_bucket_size = small_bucket_size + 1;
    int large_bucket_num = range - small_bucket_size * num_buckets;
    int boundary = large_bucket_num * large_bucket_size;

    std::vector<std::vector<int>> buckets(num_buckets);
    // Pre-allocate space to avoid re-allocation
    for (std::vector<int>& bucket : buckets) {
        bucket.reserve(large_bucket_num);
    }

    // Place each element in the appropriate bucket
    for (int num : vec) {
        int index;
        if (num < boundary) {
            index = (num - min_val) / large_bucket_size;
        } else {
            index = large_bucket_num + (num - boundary) / small_bucket_size;
        }
        if (index >= num_buckets) {
            // Handle elements at the upper bound
            index = num_buckets - 1;
        }
        buckets[index].push_back(num);
    }

    // Sort each bucket using insertion sort
    for (std::vector<int>& bucket : buckets) {
        insertionSort(bucket);
    }

    // Combine sorted buckets to get the final sorted array
    int index = 0;
    for (const std::vector<int>& bucket : buckets) {
        for (int num : bucket) {
            vec[index++] = num;
        }
    }
}
```

The sequential version of bucket sort has been provided in `src/bucketsort/sequential.cpp`, and your task is to implement a **process-level parallel bucket sort** using **MPI** in `src/bucketsort/mpi.cpp`.

### Hints:

1. The implementation of process-level bucket sort is relatively easy. The general idea is to assign each MPI worker the task of sorting several buckets and then send the sorted buckets back to the main worker to create the final sorted array.
2. The number of buckets is crucial for performance (You may also consider why bucket sort conventionally uses insertion sort for sorting each bucket). Therefore, when comparing your performance with the baseline, you need to try to **determine the optimal number of buckets**.
3. You can apply parallel bucket sort on dataset that is not uniform to see how much performance degration will there be. 

## Task #2: Process-Level Parallel Quick Sort with K-Way Merge

In order to solve the load imbalance problem for dataset not uniformly distributed in Task 1, a naive solution is to evenly deliver the dataset into each process, sorting each partition locally with quick sort, one of the faster average-case sorting algorithm, and finally merge all the partitions into the final sorted array. *However, it is important that we are not really parallelizing the quick sort itself, but simply applying it on multiple processors.*

Quick Sort is a highly efficient and widely used sorting algorithm in computer science. It is known for its fast average-case performance and is often the sorting algorithm of choice in many applications. Quick Sort operates on the principle of a divide-and-conquer strategy, where it repeatedly divides the unsorted list into two sublists: elements smaller than a chosen pivot and elements larger than the pivot. The algorithm sorts these sublists and combines them to create the final sorted list.

Here's a step-by-step explanation of how parallel quick sort works:

1. **Pivot Selection**: The algorithm selects a pivot element from the unsorted list. The choice of the pivot can significantly affect the algorithm's efficiency, and various methods are used to select the pivot, such as selecting the first, last, middle, or a random element from the list.
2. **Partitioning**: The list is then partitioned into two sublists: elements less than the pivot and elements greater than the pivot. This is done by comparing each element in the list with the pivot and moving elements to the appropriate sublist.
3. **Recursion**: Quick Sort is applied recursively to both sublists, which means that the algorithm is called on the sublists created in the previous step. This process continues until the sublists are small enough to be considered sorted.
4. **Combining**: Once the recursion reaches small enough sublists, no further action is needed. The sorted sublists are then combined to form the final sorted list.

![Unraveling QuickSort: The Fast and Versatile Sorting Algorithm | by Nathal  Dawson | Medium](../docs/images/quick-sort.png)

```c++
int partition(std::vector<int> &vec, int low, int high) {
    int pivot = vec[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (vec[j] <= pivot) {
            i++;
            std::swap(vec[i], vec[j]);
        }
    }

    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

void quickSort(std::vector<int> &vec, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(vec, low, high);
        quickSort(vec, low, pivotIndex - 1);
        quickSort(vec, pivotIndex + 1, high);
    }
}
```

The sequential version of quick sort has been provided in `src/quicksort/sequential.cpp`, and your task is to implement a **process-level parallel quick sort** using **MPI** in `src/quicksort/mpi.cpp`. 

To make life easier, we simply partition the data into several blocks and send each data block to one MPI process, which performs sequential quick-sort inside to make that data block in order. After all the processes finish sorting their data, a sequential merge is done to construct the final sorted list for output.

Note that this task actually has nothing to do with making the quick sort computation in parallel. We simply call the sequential quick-sort in each MPI process for efficient sorting and do a sequential reduction.

### Hints:

1. The sequential merging is referred as the [k-way merge problem](https://en.wikipedia.org/wiki/K-way_merge_algorithm), and you should think about how to merge multiple ordered lists into one list. One way may be merging from bottom to top (for example, 8 to 4, 4 to 2, 2 to 1). Another way may take advantage of some specific data structure (heap, for example) to do merging for all the ordered lists together, as long as you can know which list has the smallest/highest value as its head element.
2. Can the local quick sort be further parallelized given more threads? Please check the details in the extra credits part.

### Topics for Extra Credits and Deeper Invectigation:

1. It is possible to make the quick-sort computation part go in parallel to get extra speedup on top of the basic implementation in Task 1. The solution is to do dynamic thread creation to deal with the recursive function call during sorting. To implement this dynamic thread creation, you can use either Pthread, std::thread, or simply OpenMP. Please see the [Extra Credits](#extra-credits-thread-level-parallel-merge-sort-with-dynamic-thread-creation) section for more details.
2. Another optimization may be making the sequential merging (the [k-way merge problem](https://en.wikipedia.org/wiki/K-way_merge_algorithm)) go in parallel. See if you can think of any solution for this.

## Task #3: Process-Level Parallel Sorting with Regular Sampling (PSRS)

In task 2, although the data is evenly delivered to each process to achieve load balance, it comes with the cost that the final merging process can only be sequential since there is no constraint on the range of data in each process like that did in parallel bucket sort any more. According to the [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law), the speedup will be bounded by the sequential portion of the program. To achieve further speedup, we have to parallelize the merging process, and one of the popular solutions is the parallel sorting with regular sampling (PSRS) algorithm.

[Parallel Sorting by Regular Sampling (PSRS)](https://medium.com/@dhairyapatelad/parallel-sorting-using-regular-sampling-psrs-4e27eb63b307) is a new MIMD parallel sorting algorithm. The regular sampling load balancing heuristic distinguishes PSRS from other parallel sorting algorithms. Regular sampling has good analytical and empirical properties. Also, PSRS has modest communication needs and exhibits good per task locality of reference, reducing memory and communication contention. In theory, if there are no duplicate data
values, PSRS guarantees to distribute the work among processors within a factor of two of optimal load balancing, regardless of the data value distribution. In practice, PSRS achieves near-perfect load balancing [[cite]](https://www.sciencedirect.com/science/article/abs/pii/016781919390019H).

PSRS is a combination of a sequential sort, a load balancing phase, a data exchange and a parallel merge. Although any sequential sorting and merge algorithm can be used, PSRS is demonstrated using quick sort and successive 2-way merge. Given n data items (indices 1, 2, 3, ..., n) and p processors (1, 2, 3, ..., p), PSRS consists of the following four phases:

### Phase One: Sort Local Data.
Each processor is assigned a contiguous block of $\frac{n}{p}$ items. The blocks assigned to different processors are disjoint. Each processor, in parallel, sorts their local block of items using sequential quicksort. Begin the regular sampling load balancing heuristic. All p processors, in parallel, select the data items at local indices $1, w + 1, 2w + 1, \ldots; (p - 1)w + 1$ to form a representative sample of the locally sorted block. The $p^{2}$ selected data items, p from each of processors, are a regular sample of the entire data array. The local regular samples represent the keys and their value distribution at each processor.

In the following figure, each processor is assigned $\frac{n}{p} = 12$ contiguous keys to sort. Each processor takes three samples, at the 1st, 5th and 9th indices since $w = \frac{n}{p^{2}} = 4$, to form the local regular sample. Note that the distance between the indices of the samples is of fixed size.

### Phase Two: Find Pivots then Partition.

One designated processor gathers and sorts the local regular samples. $p - 1$ pivots are selected from the sorted regular sample, at indices $p + \rho$; $2p + \rho$; $3p + \rho$, ..., $(p - 1)p + \rho$. Each processor receives a copy of the pivots and forms p partitions from their sorted local blocks. A partition is contiguous internally and is disjoint from the other partitions.

In the following figure, the 9 samples are collected together and sorted. From this list (0, 3, 7, 10, 13, 16, 22, 23, 27), $p - 1 = 2$ pivots are selected. The pivots are 10 and 22 (at the 4th and 7th indices), because $\rho = floor(\frac{p}{2}) = 1$. All processors then create three partitions.

### Phase 3: Exchange Partitions.

In parallel, each processor i keeps the ith partition for itself and assigns the jth partition to the jth processor. For example, processor 1 receives partition 1 from all of the processors. Therefore, each processor keeps one partition, and reassigns $p - 1$ partitions. For example, in the figure, processor 3 sends the list (3, 4, 5, 6, 10) to processor 1, sends (14, 15, 20, 22) to processor 2, and keeps (26, 31, 32) for itself.

### Phase 4: Merge Partitions.

Each processor, in parallel, merges its p partitions into a single list that is disjoint from the merged lists of the other processors. The concatenation of all the lists is the final sorted list.

Note that in the figure, the keys in the final merged partitions at each processor are also partitioned by the pivots 10 and 22. The final sorted list is distributed over the 3 processors.

![Demonstration of a PSRS Sorting](../docs/images/PSRS-Algorithm.png)

There is no sequential version PSRS, and your task is to implement it using **MPI** in `src/psrs/main.cpp`. 

## Task #4: Dynamic Thread-Level Parallel Merge Sort

All the 3 tasks above are exhausting process-level parallelism for sorting, and the problem is that the number of processes should be determined and fixed in advance. This is not aligned with the common philosophy *divide and conquer* in many popular sorting algorithms. To achieve parallel divide and conquer, the number of threads should be dynamically determined during the run-time, and the OpenMP tasking directive can help.

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

Above is the sequential version of merge sort, and your task is to implement a **thread-level parallel merge sort** in `src/mergesort/bonus.cpp`. You can choose to use `OpenMP`, `Pthreads`, `std::thread`, or even a combination of them to complete the task.

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

[Medium: Parallel Soring using Regular Sampling (PSRS)](https://medium.com/@dhairyapatelad/parallel-sorting-using-regular-sampling-psrs-4e27eb63b307)

[On the versatility of parallel sorting by regular sampling](https://www.sciencedirect.com/science/article/abs/pii/016781919390019H)

[OpenMP Task Basics](https://hpc2n.github.io/Task-based-parallelism/branch/master/task-basics-1/)

[QuickSort with OpenMP by Mohd Ehtesham Shareef from The State University of New York](https://cse.buffalo.edu/faculty/miller/Courses/CSE702/Mohd-Ehtesham-Shareef-Fall-2020.pdf)

[Advanced Programming with OpenMP (Quick Sort as one Example)](https://cw.fel.cvut.cz/old/_media/courses/b4m35pag/lab6_slides_advanced_openmp.pdf)

[Medium: Parallel QuickSort using OpenMP](https://mcbeukman.medium.com/parallel-quicksort-using-openmp-9d18d7468cac)

[SC'13 Talk on OpenMP Tasking](https://openmp.org/wp-content/uploads/sc13.tasking.ruud.pdf)

[OpenMP Tutorial by Blaise Barney from Lawrence Livermore National Laboratory](https://hpc-tutorials.llnl.gov/openmp/)

## Extra Credits: Dynamic Parallel Sorting Algorithms

### Task #5: Dynamic Thread-Level Parallel Quick Sort

In [Task 1: Process-Level Parallel Quick Sort with MPI](#task-1-process-level-parallel-quick-sort-with-mpi), we simply called sequential quick-sort in each process. However, it is actually possible to make the quick-sort computation part itself go in parallel to get extra speedup on top of the basic implementation in Task 1. The solution is to do dynamic thread creation to deal with the recursive function call during sorting. To implement this dynamic thread creation, there are briefly two ways: you can still use the OpenMP task pragma as you did in Task 4 or explore the manual control on the creation of threads.

#### Hints:

It may feel very difficult to divide the task. However similar to merge sort, there are two places where you could assign the task to different threads.

1. When calling `quickSort` recursively, you could assign new threads to increase parallelism. However, as with the same requirement for `mergeSort`, you should **maintain control over the total number of threads** in your program.
2. Make the `partition` operation parallel. The high-level idea of the `partition` is to select a pivot and divide the array into two subarrays: one containing elements less than the pivot, and the other containing elements larger than the pivot. The sequential version of `partition` operation achieves the purpose without introducing any extra space. However, in the parallel version, you can achieve the same goal but **using extra space** to allow for a parallel process.  There are various methods to achieve this. If you don't have your own approach, you can refer to this [Video](https://www.youtube.com/watch?v=yD_pg34xhIs&t=340s). If you decide to follow the algorithm in the video, you will also need to understand how to implement a **parallel PrefixSum algorithm**. You may visit to the [Wiki](https://en.wikipedia.org/wiki/Prefix_sum) on PrefixSum for a reference. 
3. Remember that creating and destroying threads is costly, and parallelizing the `partition` operation incurs additional overhead (e.g., PrefixSum) when compared to the sequential version. Therefore, you should consider whether it's worthwhile to use parallel `partition` based on the task size and current threads number . For instance, it is entirely unnecessary to assign multiple threads for the partition of [0, 1].

### Task #6: Parallel Radix Sort on CPU

Radix sort is a non-comparative sorting algorithm that distributes elements into buckets based on the value of their individual digits. By sorting the elements digit by digit, starting from the least significant to the most significant, radix sort eventually produces a fully sorted array.

Here is a step-by-step explaination of how radix sort works:

1. **Find the Largest Element**: The first step is to identify the largest element in the array. This allows us to determine how many digits we need to process.
2. **Sort by Digits**: The array is sorted based on individual digits, starting from the least significant digit (LSD) to the most significant digit (MSD). For example, given the array \([805, 122]\), we first sort by the unit digits ('5' and '2'), resulting in \([122, 805]\). This digit-based sorting is typically done using **counting sort**.
3. **Repeat**: The process is repeated for each digit, moving from the least significant to the most significant digit, until the entire array is sorted.

```c++
#define BASE 256


void radixSort(std::vector<int> &vec) {
    int n = vec.size();
    int i;
    long exp;

    // Get the largest element
    int max_element = vec[0];

    for (i = 0; i < n; ++i) {
        if (vec[i] > max_element) {
            max_element = vec[i];
        }
    }

    // Counting sort for each digit
    for (exp = 1; max_element / exp > 0; exp *= BASE) {
        std::vector<int> output(n);
        std::vector<int> count(BASE, 0);

        for (i = 0; i < n; i++)
            count[(vec[i] / exp) % BASE]++;

        std::vector<int> start_pos(BASE, 0);
        for (int d = 1; d < BASE; d++) {
            start_pos[d] = start_pos[d - 1] + count[d - 1];
        }

        std::vector<int> offset(BASE, 0);
        for (int i = 0; i < n; i++) {
            int digit = (vec[i] / exp) % BASE;
            int pos = start_pos[digit] + offset[digit];
            output[pos] = vec[i];
            offset[digit]++;
        }

        // Assign elements back to the input vector
        for (i = 0; i < n; i++) {
            vec[i] = output[i];
        }
    }

    return;
}
```

The sequential version of radix sort has been provided in `src/radixsort-cpu/sequential.cpp`, and your task is to implement parallel radix sort using **MPI** and **OpenMP** in `src/radixsort-cpu/mpi.cpp` and `src/radixsort-cpu/omp.cpp` respectively.

### Hints:

1. The MPI-based parallel radix sort will involve distributing parts of the array across multiple processes. Each process can sort its portion of the array independently, and they are merged to produce the final vector, similar to how process-level quicksort is implemented.
2. Implementing radix sort with OpenMP is more complex, where you will need to parallelize each step of the radix sort algorithm. You should use larger bases in order to see greater speedup (for example, 16384).

### Task #7: Parallel Radix Sort on GPU

In this task, please implement a radix sort for GPU accelerator in CUDA which could support sorting for an array of length $N = 10^8$. Undergraduate student can earn extra credit from this task and graduate students are required to finish the task. 

#### Requirements

For undergraduate students, you need to complete the radix sort or counting sort (i.e. the version with radix = $N$). You can apply multiple parallel methods (e.g. reduce-then-scan method) to implement your code. (Undergraduates can get the credit with speedup about $30$ comparing with `std::sort`. )

For graduate students, you need to learn about how to implement a CUDA scan, reduce, and radix sort efficiently in this task. You can apply decoupled look-back method to implement your code. (Graduates can get the credit with speedup about $60$ comparing with `std::sort`. )

**NOTE:** To obtain the credit, you need to report your program structure (the histogram/scan/reduce/scatter/... passes of different levels in your algorithm and how they are organized) associated with parallel methods (e.g. reduce-then-scan, scan-scan-add, decoupled look-back), and your speedup factor comparing with `std::sort`. Below is the example code of warp-level inclusive scan operation implementing in the warp-scan parallel method. (Maybe this help you get some insight to optimize your sorting in different levels.)

```cuda
// scan 32 elements in warp (in place)
#define warpSize 32
__device__ __forceinline__ void warp_scan_inclusive(volatile uint* psum) {
    uint i = threadIdx.x, lane = i & (warpSize - 1);
    if (lane >= 1) psum[i] = psum[i - 1] + psum[i];
    if (lane >= 2) psum[i] = psum[i - 2] + psum[i];
    if (lane >= 4) psum[i] = psum[i - 4] + psum[i];
    if (lane >= 8) psum[i] = psum[i - 8] + psum[i];
    if (lane >= 16) psum[i] = psum[i - 16] + psum[i];
}
```

Related techniques about CUDA programming (including Q&A of this task) is planning to be introduced in the tutorial on the 15th of November. You can self-study on CUDA implementation of scan, reduce, and radix sort by yourself in advance to complete this task. 

For any question to this task, welcome to Yuan's office hour at USTF Zone (Room 103 in Start-up Zone Library) on Tuesday 18:00 - 19:00. 

#### References 

You can take reference on the following links to complete this task. 

[UMich EECS 570: Parallel Prefix Sum (Scan) with CUDA](https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf)

[UC Davis CS 223 Guest Lecture: GPU Reduce, Scan, & Sort](https://www.cs.ucdavis.edu/~amenta/f15/amenta-reduce-scan-sort.pdf)

[Nvidia 2016: Single-pass-Parallel-Prefix with Decoupled Look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf)

[Onesweep: A Faster Least Significant Digit Radix Sort for GPUs](https://arxiv.org/pdf/2206.01784)

[Nvidia GTC 2020: A Faster Radix Sort Implementation - video](https://developer.nvidia.com/gtc/2020/video/s21572-vid)

[Nvidia GTC 2020: A Faster Radix Sort Implementation - slides](https://developer.download.nvidia.cn/video/gputechconf/gtc/2020/presentations/s21572-a-faster-radix-sort-implementation.pdf)

[Thrust/CUB Library Implementation](https://github.com/thrust/cub/tree/master/cub)

## Requirements & Grading Policy

- **Three Process-Level Parallel Sorting Algorithms (60%)**
  - Task #1: Parallel Bucket Sort using MPI (10%)
  - Task #2: Parallel Quick Sort with K-Way Merge using MPI (10%)
  - Task #3: Parallel Sort with Regular Sampling using MPI (20%)
  - Task #4: Parallel Merge Sort using OpenMP (20%)

  Your programs should be able to compile & execute to get the expected computation result to get full grade in this part.

- **Performance of Your Program (30%)**
  - 5% for Task #1 on uniformly distributed dataset
  - 5% for Task #2 on both uniformly and normally distributed dataset
  - 10% for Task #3 on both uniformly and normally distributed dataset
  - 10% for Task #4 on both uniformly and normally distributed dataset

  Try your best to optimize your parallel programs for higher speedup. If your programs shows similar performance to the baseline, then you can get full mark for this part. Points will be deduted if your parallel programs perform poor while no justification can be found in your report.

- **One Report in PDF (20%, No Page Limit)**
  - **Regular Report (5%)**
  
      The report does not have to be very long and beautiful to help you get good grade, but you need to include what you have done and what you have learned in this project. The following components should be included in the report:

    - How to compile and execute your program to get the expected output on the cluster.
    - Explain clearly how did you design and implement each parallel sorting algorithm?
    - Show the experiment results you get, and do some numerical analysis, such as calculating the speedup and efficiency, demonstrated with tables and figures.
    - What kinds of optimizations have you tried to speed up your parallel program, and how do them work?
    - Any interesting discoveries you found during the experiment?

  - **Profiling Results & Analysis with `perf` (5%)**
  
      Please follow the [Instruction on Profiling with perf and nsys](https://github.com/tonyyxliu/CSC4005-2023Fall-internal/blob/main/docs/Instruction%20on%20Profiling%20with%20perf%20and%20nsys.md) to profile all of your parallel programs for the four tasks with `perf`, and do some analysis on the profiling results before & after the implementation or optimization. For example, you can use the profiling results from `perf` to do quantitative analysis that how many cache misses or page faults can be reduced with your optimization. Always keep your mind open, and try different profiling metrics in `perf` and see if you can find any interesting thing during experiment.
      
      **Note:** The raw profiling results may be very long. Please extract some of the useful items to show in your report, and remember to carry all the raw profiling results for your programs when you submit your project on BB.

- **Extra Credits (10%)**
  - Task #5: Dynamic Thread-Level Parallel Quick Sort(5%)
  - Task #6: Parallel Radix Sort on CPU (5%)
  - Task #7: Parallel Radix Sort on GPU (10%)
  
  Extra optimizations or interesting discoveries in the first three tasks may also earn you some extra credits.

### The Extra Credit Policy

According to the professor, the extra credits in this project cannot be added to other projects to make them full mark. The credits are the honor you received from the professor and the teaching stuff, and the professor may help raise you to a higher grade level if you are at the boundary of two grade levels and he think you deserve a better grade with your extra credits. For example, if you are the top students with B+ grade, and get enough extra credits, the professor may raise you to A- grade. Furthermore, professor will invite a few stundets with high extra credits to have dinner with him and all other teaching stuff.

### Grading Policy for Late Submission

1. late submission for less than 10 minutes after then DDL is tolerated for possible issues during submission.
2. 10 Points deduction for each day after the DDL (11 minutes late will be considered as one day, so be careful)
3. Zero point if you submitted your project late for more than two days
If you have some special reasaons for late submission, please send email to the professor and c.c to TA Liu Yuxuan.

### File Structure to Submit on BlackBoard

```bash
118010200.pdf  # Report
118010200.zip  # Codes
|-
|--- src/           # Where your source codes lie in
|--- CMakeLists.txt # Root CMakeLists.txt
|-
|--- profiling/     # Where your perf profiling raw results lie in
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

Remember to execute the scripts for two dataset with different distribution.

**Important**: Change the directory of output file in `sbatch-uniform.sh` and `sbatch-normal.sh` first, and you can also change the matrix files for different testing.

```bash
# Use sbatch
cd /path/to/project3
sbatch ./src/sbatch-uniform.sh
sbatch ./src/sbatch-normal.sh
```

## Performance Evaluation

### Correctness Verification

After performing your implemented sorting algorithm, we will compare your sorting results with the `std::sort` function, both for correctness and performance. If your sorting results are correct, you will see the output suggesting that you have passed the test, as follows:

```bash
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

Before executing the sorting program, you should set the length of the vector as suggested in the previous section, and a random vector with the given size will be generated. For Quick Sort, Merge Sort, and Bucket Sort, we use a vector size of 100,000,000 (10E+08) and bucket size of 1000000 (10E+06) to verify the performance due to their relatively low time complexity.

**Experiment Setup**
- Vector Size: 100,000,000 (10E+08)
- Bucket Size (for bucket sort only): 1,000,000 (10E+06)

Here are the performance baselines (in milliseconds) for Project 3:

#### Uniformly Distributed Dataset

*Params: low = 0, high = SIZE*

| Workers | std::sort | BucketSort(MPI) | QuickSort(MPI) | PSRS (MPI) | MergeSort (OpenMP) | QuickSort(Threads) | RadixSort (MPI) | RadixSort (OpenMP) |
| :-----: | :-------: | :-------------: | :------------: | :--------: | :----------------: | :----------------: | :-------------: | :----------------: |
|    1    | 11777     |     10104       |      13583     | 12639      |       19535        |       13006        |      10012      |        2300        |
|    4    | N/A       |      4592       |      9187      | 4771       |       9312         |        7804        |       8529      |        1375        |
|   16    | N/A       |      2457       |      8403      | 1696       |        3242        |        3808        |       8730      |         763        |
|   32    | N/A       |      2213       |      9133      | 1453       |        2099        |        3597        |       9283      |         624        |

#### Normally Distributed Dataset

*Params: mean = size / 2.0, stddev = size / 10.0*

Note: According to the empirical law of normal distribution:
- Around 68% of data appear within [mean - stddev, mean + stddev]
- Around 95% of data appear within [mean - 2 * stddev, mean + 2 * stddev]
- Around 99.7% of data appear within [mean - 3 * stddev, mean + 3 * stddev]

| Workers | std::sort | BucketSort(MPI) | QuickSort(MPI) | PSRS (MPI) | MergeSort (OpenMP) | QuickSort(Threads) | RadixSort (MPI) | RadixSort (OpenMP) |
| :-----: | :-------: | :-------------: | :------------: | :--------: | :----------------: | :----------------: | :-------------: | :----------------: |
|    1    | 11740     |     12375       |      13440     | 12620      |       19450        |       N/A          |      N/A        |         N/A        |
|    4    | N/A       |      9174       |      9479      | 4752       |       9784         |        N/A         |      N/A        |         N/A        |
|   16    | N/A       |      5904       |      9022      | 1695       |        3179        |        N/A         |      N/A        |         N/A        |
|   32    | N/A       |      4040       |      9985      | 1456       |        2027        |        N/A         |      N/A        |         N/A        |
