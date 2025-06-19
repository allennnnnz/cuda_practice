#include <stdio.h>
#include <stdlib.h>

__global__ void add(int *a, int *b, int *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}
int main() {
    int N = 10;
    int *ha = (int *)malloc(N * sizeof(int));
    int *hb = (int *)malloc(N * sizeof(int));
    int *hc = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        ha[i] = i;
        hb[i] = i * 2;
    }

    int *da, *db, *dc;
    cudaMalloc((void**)&da, N * sizeof(int));
    cudaMalloc((void**)&db, N * sizeof(int));
    cudaMalloc((void**)&dc, N * sizeof(int));

    cudaMemcpy(da, ha, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(int), cudaMemcpyHostToDevice);
    // 不需要複製 hc 到 dc

    add<<<2, 5>>>(da, db, dc);
    cudaDeviceSynchronize();

    cudaMemcpy(hc, dc, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("hc[%d] = %d\n", i, hc[i]);
    }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(ha);
    free(hb);
    free(hc);

    return 0;
}
