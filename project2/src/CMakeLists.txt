# Required packages and flags
## OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message("OPENMP FOUND, Version: " ${OpenMP_CXX_VERSION})
endif ()

## MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI FOUND, Version: ${MPI_CXX_VERSION}")
endif ()

# Add Executables

##################
## [Basic Part] ##
##################

## Naive
add_executable(naive
        naive.cpp
        matrix.cpp matrix.hpp)
target_compile_options(naive PRIVATE -O2)

## Task 1.1: Fused Multiply and Add by declaring __restrict__ pointer
add_executable(fma_restrict
        task1_fma_restrict_ptr.cpp
        matrix.cpp matrix.hpp)
target_compile_options(fma_restrict PRIVATE -O2 -mfma -ffp-contract=fast)

## Task 1.2: Fused Multiply and Add by adding a standalone variable
add_executable(fma_standalone_var
        task1_fma_standalone_var.cpp
        matrix.cpp matrix.hpp)
target_compile_options(fma_standalone_var PRIVATE -O2 -mfma -ffp-contract=fast)

## Task 2: Declaring Constant Pointers
add_executable(const_ptr_decl
        task2_const_ptr_decl.cpp
        matrix.cpp matrix.hpp)
target_compile_options(const_ptr_decl PRIVATE -O2 -mfma -ffp-contract=fast)

## Task 3.1: Improve Memory Locality with Matrix Transposition
add_executable(transpose
        task3_transpose.cpp
        matrix.cpp matrix.hpp)
target_compile_options(transpose PRIVATE -O2 -mfma -ffp-contract=fast)

## Task 3.2: Improve Memory Locality with Loop Re-Ordering
add_executable(loop_interchange
        task3_loop_interchange.cpp
        matrix.cpp matrix.hpp)
target_compile_options(loop_interchange PRIVATE -O2 -mfma -ffp-contract=fast)

## Task 4.1: Improve Memory Localtity with Tiling (+ Transposition)
add_executable(tiling_transpose
        task4_tiling_transpose.cpp
        matrix.cpp matrix.hpp)
target_compile_options(tiling_transpose PRIVATE -O2 -mfma -ffp-contract=fast)

## Task 4.1: Improve Memory Localtity with Tiling (+ Loop-interchange)
add_executable(tiling_loop_interchange
        task4_tiling_loop_interchange.cpp
        matrix.cpp matrix.hpp)
target_compile_options(tiling_loop_interchange PRIVATE -O2 -mfma -ffp-contract=fast)

## Task 5: Auto Vectorization
add_executable(autovec
        task5_auto_vec.cpp
        matrix.cpp matrix.hpp)
target_compile_options(autovec PRIVATE -O2 -ftree-vectorize -fopt-info-vec -mavx2 -mfma -ffp-contract=fast)

## Task 6: OpenMP for Multi-Threaded
add_executable(openmp
        task6_openmp.cpp
        matrix.cpp matrix.hpp)
target_compile_options(openmp PRIVATE -O2 -ftree-vectorize -fopt-info-vec -fopenmp -mavx2 -mfma -ffp-contract=fast)
target_include_directories(openmp PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(openmp PRIVATE ${OpenMP_CXX_LIBRARIES})


##########################
## [Extra Credits Part] ##
##########################

# ## SIMD Vectorization (AVX2)
# add_executable(simd
#         extra_simd.cpp
#         matrix.cpp matrix.hpp)
# target_compile_options(simd PRIVATE -O2 -mavx2)

# ## MPI + OpenMP + Everything
# add_executable(mpi
#         extra_mpi.cpp
#         matrix.cpp matrix.hpp)
# target_compile_options(mpi PRIVATE -O2 -fopenmp -mavx2)
# target_include_directories(mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})
# target_link_libraries(mpi ${MPI_LIBRARIES} ${OpenMP_CXX_LIBRARIES})

# add_subdirectory(gpu)
