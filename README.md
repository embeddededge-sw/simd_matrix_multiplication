SIMD Matrix Multiplicaion


Code compilation: 

gcc -O3 -march=native assignement_final_code.c -lm -o assignment_bin

About the code: -

This C program implements high-performance matrix multiplication using SIMD intrinsics on x86-64 processors.
It provides three implementations: a naive triple-loop baseline, a basic AVX2 dot-product version using fused
multiply-add instructions, and an advanced blocked implementation inspired by the GotoBLAS approach described
in "Anatomy of High-Performance Matrix Multiplication." The blocked version optimizes memory access through
cache blocking (dividing matrices into L1/L2/L3 cache-friendly blocks), data packing (reorganizing matrix B
panels into contiguous memory to eliminate strided access), and a 6×16 micro-kernel that uses AVX2 intrinsics
with 12 accumulators to compute 96 elements simultaneously. The program validates correctness by comparing
optimized results against the naive version with a tolerance of 1e-3, measures execution time using nanosecond
precision timestamps, and reports performance in GFlops/sec, demonstrating that combining SIMD vectorization
with cache-aware algorithms can achieve >2× speedup over the baseline implementation.

