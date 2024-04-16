// CUDA运行时头文件
#include <cuda_runtime.h>

#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <chrono>

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
    if (code != cudaSuccess) {
        const char* err_name    = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

void checkMemcpyAsyncTime(int* d_src, int* d_dst, int size) {
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpyAsync(d_dst, d_src, size, cudaMemcpyDeviceToDevice, stream);
    checkRuntime(cudaStreamSynchronize(stream));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("MemcpyAsync time: %f ms\n", milliseconds);
    checkRuntime(cudaStreamDestroy(stream));
}

void checkMemcpyTime(int* d_src, int* d_dst, int size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Memcpy time: %f ms\n", milliseconds);
}

void checkCpuMemcpyTime(int* src, int* dst, int size) {
    auto start = std::chrono::high_resolution_clock::now();
    memcpy(dst, src, size);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "CPU Memcpy time: " << duration.count() << " milliseconds\n";
}

int app_cuda() {
    int device_id = 0;
    checkRuntime(cudaSetDevice(device_id));

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    // 测试耗时
    // int N    = 1 << 20;  // Number of elements
    int N    = 3 * 1920 * 1080;
    int size = N * sizeof(int);

    // Allocate memory on host
    int* h_src = (int*)malloc(size);
    int* h_dst = (int*)malloc(size);

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_src[i] = i;
    }
    checkCpuMemcpyTime(h_src, h_dst, size);

    // Allocate memory on device
    int *d_src, *d_dst;
    checkRuntime(cudaMalloc(&d_src, size));
    checkRuntime(cudaMalloc(&d_dst, size));

    // Copy data from host to device
    checkRuntime(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice));

    // Test cudaMemcpyAsync time
    checkMemcpyAsyncTime(d_src, d_dst, size);

    // Test cudaMemcpy time
    checkMemcpyTime(d_src, d_dst, size);

    // Free device memory
    checkRuntime(cudaFree(d_src));
    checkRuntime(cudaFree(d_dst));

    // Free host memory
    free(h_src);
    free(h_dst);
    // end

    // 在GPU上开辟空间
    float* memory_device = nullptr;
    checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float)));

    // 在CPU上开辟空间并且放数据进去，将数据复制到GPU
    float* memory_host = new float[100];
    memory_host[2]     = 520.25;
    checkRuntime(cudaMemcpyAsync(memory_device, memory_host, sizeof(float) * 100, cudaMemcpyHostToDevice,
                                 stream));  // 异步复制操作，主线程不需要等待复制结束才继续

    // 在CPU上开辟pin memory,并将GPU上的数据复制回来
    float* memory_page_locked = nullptr;
    checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float)));
    checkRuntime(cudaMemcpyAsync(memory_page_locked, memory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost,
                                 stream));  // 异步复制操作，主线程不需要等待复制结束才继续
    checkRuntime(cudaStreamSynchronize(stream));

    printf("%f\n", memory_page_locked[2]);

    // 释放内存
    checkRuntime(cudaFreeHost(memory_page_locked));
    checkRuntime(cudaFree(memory_device));
    checkRuntime(cudaStreamDestroy(stream));
    delete[] memory_host;
    return 0;
}