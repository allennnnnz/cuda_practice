#include <iostream>
#include <cuda_runtime.h>

#define R 16  // rows
#define C 16  // columns

__global__ void matrixAdd(int* a, int* b, int* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];

}

int main(){
    int size = R * C * sizeof(int);

    // host memory
    int* ha = (int*)malloc(size);
    int* hb = (int*)malloc(size);
    int* hc = (int*)malloc(size);

    // initialize ha & hb
    for (int i = 0; i < R * C; i++) {
        ha[i] = rand() % 10;
        hb[i] = rand() % 10;
    }
    printf("=== Matrix A ===\n");
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            printf("%4d ", ha[i * C + j]);
        }
        printf("\n");
    }

    printf("\n=== Matrix B ===\n");
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            printf("%4d ", hb[i * C + j]);
        }
        printf("\n");
    }
    // device memory
    int *da, *db, *dc;
    cudaMalloc((void**)&da, size);
    cudaMalloc((void**)&db, size);
    cudaMalloc((void**)&dc, size);

    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    // kernel 呼叫配置
    int threadsPerBlock = 256;
    int numBlocks = (R * C + threadsPerBlock - 1) / threadsPerBlock;
    matrixAdd<<<numBlocks, threadsPerBlock>>>(da, db, dc);
    cudaDeviceSynchronize();

    // device → host
    cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);

    // 印出矩陣結果
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            printf("%4d ", hc[i * C + j]);
        }
        printf("\n");
    }

    // free
    free(ha); free(hb); free(hc);
    cudaFree(da); cudaFree(db); cudaFree(dc);

    return 0;
}
