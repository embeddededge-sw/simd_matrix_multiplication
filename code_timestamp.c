

#include <stdio.h>
#include <time.h>
#include <stdint.h>

// User-defined function to get the current timestamp in nanoseconds
uint64_t get_timestamp_ns() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
        perror("clock_gettime failed");
        return 0; // Return 0 if the clock read fails
    }
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// Simulated workload to benchmark
void target_function() {
    volatile unsigned long long count = 0;
    for (count = 0; count < 10000000ULL; count++) {
        // Simple loop to consume CPU cycles
    }
}

int main() {
    // 1. Capture the start timestamp using the user-defined function
    uint64_t start_ns = get_timestamp_ns();

    // 2. Execute the target function
    target_function();

    // 3. Capture the end timestamp using the user-defined function
    uint64_t end_ns = get_timestamp_ns();

    // 4. Compute the delta
    uint64_t elapsed_ns = end_ns - start_ns;

    // Print the results
    printf("Start Timestamp: %lu ns\n", start_ns);
    printf("End Timestamp:   %lu ns\n", end_ns);
    printf("Execution Time:  %lu nanoseconds\n", elapsed_ns);
    printf("Execution Time:  %.6f milliseconds\n", (double)elapsed_ns / 1000000.0);

    return 0;
}
