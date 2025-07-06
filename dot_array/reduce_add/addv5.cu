// shffle
#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "config.h"

__global__ void dot_array_p1(const int *dinput, int *doutput) {
    __shared__ int shared_output[THREAD_PRE_BLOCK]; 
    int idx = blockIdx.x * blockDim.x * 2+ threadIdx.x;
    
    shared_output[threadIdx.x] = dinput[idx] + dinput[idx + blockDim.x]; 
    __syncthreads();

    // thread num ＝> 32時會有多個warp交替運行，所以需要__syncthreads()
    for(int stride = blockDim.x / 2; stride >= 32; stride /= 2) {
        if(threadIdx.x < stride) {
            shared_output[threadIdx.x] += shared_output[threadIdx.x + stride];
        }
        __syncthreads();
    }
    //shuffle做加法直接在register中完成，不用經過shrared memory
    //shuffle只能處理32個值，所以上面要多做一回
    if (threadIdx.x < 32){
        //讀進去register
        int val = shared_output[threadIdx.x];
        for (int offset = warpSize/2; offset > 0; offset /= 2){
             val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) {
            doutput[blockIdx.x] = val;
        }
    }
   
    
    

    
}

int dot_product_cpu(const int* input, int N) {
    int result = 0;
    for (int i = 0; i < N; i++) {
        result += input[i];
    }
    return result;
}

int main(){
    // Align input size to multiple of THREAD_PRE_BLOCK
    int extra = ((DATA_NUM + THREAD_PRE_BLOCK - 1) / THREAD_PRE_BLOCK) * THREAD_PRE_BLOCK;
    int* hinput = (int*)malloc(extra * sizeof(int));
    srand(time(NULL));
    for (int i = 0; i < DATA_NUM; i++) {
        hinput[i] = rand() % 10;
    }
    for (int i = DATA_NUM; i < extra; i++) {
        hinput[i] = 0;
    }

    int numOfPartialsum = extra / (2 * THREAD_PRE_BLOCK);
    int* houtput = (int*)malloc(numOfPartialsum * sizeof(int));

    // CPU result
    int cpu_result = dot_product_cpu(hinput, DATA_NUM);
    printf("CPU dot product result: %d\n", cpu_result);

    // GPU memory allocation
    int *dinput, *doutput;
    cudaMalloc((void**)&dinput, extra * sizeof(int));
    cudaMalloc((void**)&doutput, numOfPartialsum * sizeof(int));
    cudaMemcpy(dinput, hinput, extra * sizeof(int), cudaMemcpyHostToDevice);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int repeat = 10;
    float total_kernel_ms = 0.0f;
    float total_bandwidth = 0.0f;
    int last_gpu_result = 0;

    for (int r = 0; r < repeat; ++r) {
        cudaEventRecord(start);
        dot_array_p1<<<numOfPartialsum, THREAD_PRE_BLOCK>>>(dinput, doutput);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, start, stop);

        // Copy back and sum partial results
        cudaMemcpy(houtput, doutput, numOfPartialsum * sizeof(int), cudaMemcpyDeviceToHost);
        int gpu_result = 0;
        for(int i = 0; i < numOfPartialsum; i++) {
            gpu_result += houtput[i];
        }

        float total_bytes = DATA_NUM * sizeof(int) + numOfPartialsum * sizeof(int);
        float bandwidth = total_bytes / (kernel_ms / 1000.0f) / 1e9f; // GB/s

        total_kernel_ms += kernel_ms;
        total_bandwidth += bandwidth;
        last_gpu_result = gpu_result; // 保留最後一次的結果
        
    }

    printf("Dot product result (last run): %d\n", last_gpu_result);
    printf("Average kernel time: %.3f ms\n", total_kernel_ms / repeat);
    printf("Average memory bandwidth: %.3f GB/s\n", total_bandwidth / repeat);

    // Output average performance to file
    std::ofstream perf_file("performance.csv", std::ios::app);
    perf_file << "addv2.cu," << (total_kernel_ms / repeat) << "," << (total_bandwidth / repeat) << std::endl;
    perf_file.close();

    cudaFree(dinput);
    cudaFree(doutput);
    free(hinput);
    free(houtput);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}