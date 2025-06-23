#include <iostream>
#include <cuda_runtime.h>

__global__ void arrmul(int* a, int* b, int* c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 100) {
        c[idx] = a[idx] * b[idx];
    }
}

int main() {
    const int size = 100;
    int block_num = 10;
    int thread_per_block = 10;

    // Host memory
    int* ha = (int*)malloc(size * sizeof(int));
    int* hb = (int*)malloc(size * sizeof(int));
    int* hc = (int*)malloc(size * sizeof(int));

    // 初始化資料
    for (int i = 0; i < size; i++) {
        ha[i] = rand() % 10;
        hb[i] = rand() % 10;
    }

    // Device memory
    int* da;
    int* db;
    int* dc;
    cudaMalloc((void**)&da, size * sizeof(int));
    cudaMalloc((void**)&db, size * sizeof(int));
    cudaMalloc((void**)&dc, size * sizeof(int));

    // 複製資料到裝置
    cudaMemcpy(da, ha, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size * sizeof(int), cudaMemcpyHostToDevice);

    // 執行 CUDA kernel
    arrmul<<<block_num, thread_per_block>>>(da, db, dc);
    cudaDeviceSynchronize();

    // 複製結果回主機
    cudaMemcpy(hc, dc, size * sizeof(int), cudaMemcpyDeviceToHost);

    // 印出結果
    for (int i = 0; i < size; i++) {
        printf("%d * %d = %d\n", ha[i], hb[i], hc[i]);
    }

    // 釋放記憶體
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(ha);
    free(hb);
    free(hc);

    return 0;
}
