//使用share memory 的 baseline
#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "config.h"

__global__ void dot_array_p1(int *doutput) {
    int* outputc_begin = blockDim.x * blockIdx.x + doutput;

    __shared__ int shared_output[THREAD_PRE_BLOCK];

    shared_output[threadIdx.x] = outputc_begin[threadIdx.x];

    //相加個點
    for(int i=1;i<blockDim.x;i*=2){
        if(threadIdx.x%(2*i)==0){
            outputc_begin[threadIdx.x] += outputc_begin[threadIdx.x+i];
        }
        __syncthreads();
    }
}

int dot_product_cpu(int* input, int N) {
    int result = 0;
    for (int i = 0; i < N; i++) {
        result += input[i];
    }
    return result;
}

int main(){
    //宣告陣列
    int* hinput = (int*)malloc(DATA_NUM*sizeof(int));
    srand(time(NULL));  // 只要呼叫一次
for (int i = 0; i < DATA_NUM; i++) {
    hinput[i] = rand() % 10;
}
    int* houtput = (int*)malloc(DATA_NUM*sizeof(int));

    //CPU計算
    int result = dot_product_cpu(hinput, DATA_NUM);
    printf("CPU dot product result: %d\n", result);

    //宣告gpu記憶體空間
    int* doutput;
    cudaMalloc( (void**)&doutput, DATA_NUM*sizeof(int));
    cudaMemcpy( doutput, hinput, DATA_NUM*sizeof(int), cudaMemcpyHostToDevice);

    //評估參數
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    
    float kernel1_ms = 0.0f;

    //Kernel運算
    int numOfBlocks = DATA_NUM / THREAD_PRE_BLOCK;
    //Kernel 1
    cudaEventRecord(start1);
    dot_array_p1<<<numOfBlocks, THREAD_PRE_BLOCK>>>(doutput);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&kernel1_ms, start1, stop1);


    //回傳結果並列印
    cudaMemcpy( houtput, doutput, DATA_NUM*sizeof(int), cudaMemcpyDeviceToHost);

    //將partial sum的結果加總
    int res = 0;
    for(int i = 0; i < DATA_NUM; i+= THREAD_PRE_BLOCK) {
        res += houtput[i];
    }


    
    printf("Dot product result: %d\n", res);

    float total_bytes = 2 * DATA_NUM * sizeof(int); 
    float bandwidth = total_bytes / (kernel1_ms / 1000.0f) / 1e9f; // GB/s
    printf("Kernel 1 time: %.3f ms\n", kernel1_ms);
    printf("Total GPU time: %.3f ms\n", kernel1_ms);
    printf("Memory bandwidth: %.3f GB/s\n", bandwidth);

    // 輸出效能到檔案
    std::ofstream perf_file("performance.csv", std::ios::app);
    perf_file << "addv0.cu," << kernel1_ms << "," << bandwidth << std::endl;
    perf_file.close();

    
    cudaFree(doutput );
    free(hinput);
    free(houtput);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
}