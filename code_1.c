#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h> // Header for AVX2 intrinsics

#define N 1024 // Must be a multiple of 8 for basic alignment/vectorization

// Optimized SIMD Matrix Multiplication (i, k, j order)
void matmul_simd(const float *restrict A, const float *restrict B, float *restrict C) {
    // Initialize the result matrix C to 0
    for (int i = 0; i < N * N; i++) {
        C[i] = 0.0f;
    }

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            // 1. Broadcast a single scalar element A[i][k] into all 8 slots of an AVX register
            __m256 a_vec = _mm256_set1_ps(A[i * N + k]);

            // 2. Vectorized inner loop processing 8 elements of row B[k] at a time
            for (int j = 0; j < N; j += 8) {
                // Load 8 contiguous floats from B[k][j] to B[k][j+7]
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);

                // Load 8 existing floats from C[i][j]
                __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);

                // Perform Fused Multiply-Add: C = (A * B) + C
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                // Store the updated 8 elements back into C
                _mm256_storeu_ps(&C[i * N + j], c_vec);
            }
        }
    }
}

int main() {
    // Dynamic allocation ensures cache friendliness or optional alignment
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));

    // Seed dummy data
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    matmul_simd(A, B, C);

    // Verify a sample output cell
    printf("C[0] expected: %f, got: %f\n", (float)N * 2.0f, C[0]);

    free(A);
    free(B);
    free(C);
    return 0;
}
