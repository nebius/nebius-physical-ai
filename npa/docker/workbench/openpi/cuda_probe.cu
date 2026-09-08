// Verify that the selected CUDA architecture executes actual device code.
#include <cuda_runtime.h>
#include <cstdio>

#ifndef EXPECTED_MAJOR
#error "Compile with the required CUDA architecture major version"
#endif

__global__ void write_result(int* result) {
    *result = 42;
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count != 1) return 2;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 3;
    if (properties.major != EXPECTED_MAJOR || properties.minor != 0) return 4;
    int* device_result = nullptr;
    if (cudaMalloc(&device_result, sizeof(int)) != cudaSuccess) return 5;
    write_result<<<1, 1>>>(device_result);
    if (cudaDeviceSynchronize() != cudaSuccess) return 6;
    int result = 0;
    if (cudaMemcpy(&result, device_result, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 7;
    if (cudaFree(device_result) != cudaSuccess || result != 42) return 8;
    std::printf("devices=%d cc=%d.%d result=%d\n", count, properties.major, properties.minor, result);
    return 0;
}
