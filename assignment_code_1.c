/*
 * assignment.c
 * M.Tech GPU Course Assignment: SIMD Matrix Multiplication
 * Compile: gcc -O3 -march=native -o run assignment.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>      // AVX, FMA, _mm_malloc/_mm_free

/* -------------------- Configuration -------------------- */
#define MC  64     // block size for rows of A (L1)
#define KC  256    // block size for columns of A / rows of B (L2)
#define NC  4096   // block size for columns of B (L3)
#define MR  6      // micro‑kernel rows
#define NR  16     // micro‑kernel columns

/* -------------------- Memory helpers -------------------- */
static float* alloc_aligned(size_t n) {
    return (float*)_mm_malloc(n * sizeof(float), 32);
}
static void free_aligned(float* p) {
    _mm_free(p);
}

static void init_mat(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = (float)(rand() % 100) / 10.0f;
}

/* ---------- 1. NAIVE BASELINE (triple loop) ---------- */
void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

/* ----- 2. SIMD DOT‑PRODUCT (basic AVX + FMA) ------ */
void matmul_simd_dot(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256 sum = _mm256_setzero_ps();
            int k;
            for (k = 0; k <= K - 8; k += 8) {
                __m256 a = _mm256_load_ps(&A[i * K + k]);
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);   // strided access
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            /* horizontal reduction of 8 values */
            float tmp[8];
            _mm256_storeu_ps(tmp, sum);
            float total = tmp[0] + tmp[1] + tmp[2] + tmp[3]
                        + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            /* remainder (K % 8) */
            for (; k < K; ++k)
                total += A[i * K + k] * B[k * N + j];
            C[i * N + j] = total;
        }
    }
}

/* --------- 3. HIGH‑PERFORMANCE BLOCKED (GotoBLAS) --------- */

/* pack a panel of B into contiguous memory (column‑major within block) */
static void pack_B(float* B, float* packed, int kc, int nc, int N) {
    for (int kk = 0; kk < kc; ++kk)
        for (int jj = 0; jj < nc; ++jj)
            packed[kk * nc + jj] = B[kk * N + jj];
}

/* micro‑kernel: computes a 6×16 block of C */
static void micro_kernel_6x16(float* C, float* A, float* packedB,
                              int m, int n, int k, int ldC, int ldA) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int kk = 0; kk < k; kk += 8) {
        __m256 a0 = _mm256_load_ps(&A[0 * ldA + kk]);
        __m256 a1 = _mm256_load_ps(&A[1 * ldA + kk]);
        __m256 a2 = _mm256_load_ps(&A[2 * ldA + kk]);
        __m256 a3 = _mm256_load_ps(&A[3 * ldA + kk]);
        __m256 a4 = _mm256_load_ps(&A[4 * ldA + kk]);
        __m256 a5 = _mm256_load_ps(&A[5 * ldA + kk]);

        __m256 b0 = _mm256_load_ps(&packedB[kk * NR + 0]);
        __m256 b1 = _mm256_load_ps(&packedB[kk * NR + 8]);

        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    /* store results (up to 16 columns) */
    _mm256_storeu_ps(&C[0 * ldC + 0], c00);
    if (n > 8) _mm256_storeu_ps(&C[0 * ldC + 8], c01);
    _mm256_storeu_ps(&C[1 * ldC + 0], c10);
    if (n > 8) _mm256_storeu_ps(&C[1 * ldC + 8], c11);
    _mm256_storeu_ps(&C[2 * ldC + 0], c20);
    if (n > 8) _mm256_storeu_ps(&C[2 * ldC + 8], c21);
    _mm256_storeu_ps(&C[3 * ldC + 0], c30);
    if (n > 8) _mm256_storeu_ps(&C[3 * ldC + 8], c31);
    _mm256_storeu_ps(&C[4 * ldC + 0], c40);
    if (n > 8) _mm256_storeu_ps(&C[4 * ldC + 8], c41);
    _mm256_storeu_ps(&C[5 * ldC + 0], c50);
    if (n > 8) _mm256_storeu_ps(&C[5 * ldC + 8], c51);
}

/* fallback scalar for remainders (m%MR != 0 or n%NR != 0) */
static void micro_kernel_fallback(float* C, float* A, float* packedB,
                                  int m, int n, int k, int ldC, int ldA) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk)
                sum += A[i * ldA + kk] * packedB[kk * NR + j];
            C[i * ldC + j] += sum;
        }
}

/* main blocked multiplication routine */
void matmul_blocked(float* A, float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    float* packedB = (float*)_mm_malloc(KC * NC * sizeof(float), 32);

    for (int j = 0; j < N; j += NC) {
        int nc = (j + NC > N) ? N - j : NC;
        for (int p = 0; p < K; p += KC) {
            int kc = (p + KC > K) ? K - p : KC;
            pack_B(&B[p * N + j], packedB, kc, nc, N);

            for (int i = 0; i < M; i += MC) {
                int mc = (i + MC > M) ? M - i : MC;
                int mb = mc / MR;
                int mr_rem = mc % MR;
                int nb = nc / NR;
                int nr_rem = nc % NR;

                /* full micro‑kernel blocks */
                for (int r = 0; r < mb; ++r) {
                    for (int c = 0; c < nb; ++c) {
                        float* C_ptr = &C[(i + r * MR) * N + (j + c * NR)];
                        float* A_ptr = &A[(i + r * MR) * K + p];
                        float* B_ptr = &packedB[c * KC * NR];
                        micro_kernel_6x16(C_ptr, A_ptr, B_ptr, MR, NR, kc, N, K);
                    }
                    if (nr_rem) {
                        float* C_ptr = &C[(i + r * MR) * N + (j + nb * NR)];
                        float* A_ptr = &A[(i + r * MR) * K + p];
                        float* B_ptr = &packedB[nb * KC * NR];
                        micro_kernel_fallback(C_ptr, A_ptr, B_ptr, MR, nr_rem, kc, N, K);
                    }
                }
                /* remainder rows */
                if (mr_rem) {
                    for (int c = 0; c < nb; ++c) {
                        float* C_ptr = &C[(i + mb * MR) * N + (j + c * NR)];
                        float* A_ptr = &A[(i + mb * MR) * K + p];
                        float* B_ptr = &packedB[c * KC * NR];
                        micro_kernel_fallback(C_ptr, A_ptr, B_ptr, mr_rem, NR, kc, N, K);
                    }
                    if (nr_rem) {
                        float* C_ptr = &C[(i + mb * MR) * N + (j + nb * NR)];
                        float* A_ptr = &A[(i + mb * MR) * K + p];
                        float* B_ptr = &packedB[nb * KC * NR];
                        micro_kernel_fallback(C_ptr, A_ptr, B_ptr, mr_rem, nr_rem, kc, N, K);
                    }
                }
            }
        }
    }
    _mm_free(packedB);
}

/* ------------------- TIMESTAMP FUNCTION (as requested) ------------------- */
uint64_t get_timestamp_ns() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
        perror("clock_gettime failed");
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* ------------------- validation & performance calculation ------------------- */
static double compute_gflops(uint64_t elapsed_ns, int M, int N, int K) {
    double seconds = (double)elapsed_ns / 1e9;
    return (2.0 * M * N * K) / (seconds * 1e9);
}

static int validate(float* C1, float* C2, int M, int N) {
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(C1[i] - C2[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff < 1e-3f;
}

/* ------------------------ main --------------------------- */
int main(void) {
    int M = 1024, N = 1024, K = 1024;   // change to 2048 for better measurement
    printf("Matrix Size: %d x %d x %d\n", M, N, K);

    float *A = alloc_aligned(M * K);
    float *B = alloc_aligned(K * N);
    float *C_naive = alloc_aligned(M * N);
    float *C_simd   = alloc_aligned(M * N);
    float *C_blocked= alloc_aligned(M * N);

    srand(42);
    init_mat(A, M, K);
    init_mat(B, K, N);

    uint64_t start, end, elapsed;

    // 1. Naive
    start = get_timestamp_ns();
    matmul_naive(A, B, C_naive, M, N, K);
    end = get_timestamp_ns();
    elapsed = end - start;
    printf("Naive:      %llu ns (%.6f s), %.2f GFlops\n",
           (unsigned long long)elapsed, (double)elapsed/1e9,
           compute_gflops(elapsed, M, N, K));

    // 2. SIMD Dot
    start = get_timestamp_ns();
    matmul_simd_dot(A, B, C_simd, M, N, K);
    end = get_timestamp_ns();
    elapsed = end - start;
    printf("SIMD Dot:   %llu ns (%.6f s), %.2f GFlops\n",
           (unsigned long long)elapsed, (double)elapsed/1e9,
           compute_gflops(elapsed, M, N, K));

    // 3. Blocked
    start = get_timestamp_ns();
    matmul_blocked(A, B, C_blocked, M, N, K);
    end = get_timestamp_ns();
    elapsed = end - start;
    printf("Blocked:    %llu ns (%.6f s), %.2f GFlops\n",
           (unsigned long long)elapsed, (double)elapsed/1e9,
           compute_gflops(elapsed, M, N, K));

    // Validation
    printf("\nValidation (vs Naive):\n");
    printf("SIMD Dot:   %s\n", validate(C_naive, C_simd, M, N) ? "PASS" : "FAIL");
    printf("Blocked:    %s\n", validate(C_naive, C_blocked, M, N) ? "PASS" : "FAIL");

    free_aligned(A); free_aligned(B);
    free_aligned(C_naive); free_aligned(C_simd); free_aligned(C_blocked);
    return 0;
}
