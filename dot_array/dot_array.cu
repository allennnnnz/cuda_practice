#include <iostream>
#include <cuda_runtime.h>

__global__ void dot_array(int *a, int *b, int *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] * b[idx];
}

int main(){
    int* ha = (int*)malloc(20*sizeof(int));
    int* hb = (int*)malloc(20*sizeof(int));
    int* hc = (int*)malloc(20*sizeof(int));

    for (int i = 0; i < 20; i++) {
        ha[i] = rand() % 10;
        hb[i] = rand() % 10;
        printf("a[%2d] = %d\tb[%2d] = %d\n", i, ha[i], i, hb[i]);
    }

    int* da;
    int* db;
    int* dc;

    cudaMalloc( (void**)&da, 20*sizeof(int));
    cudaMalloc( (void**)&dc, 20*sizeof(int));
    cudaMalloc( (void**)&db, 20*sizeof(int));

    cudaMemcpy( da, ha, 20*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( db, hb, 20*sizeof(int), cudaMemcpyHostToDevice);

    dot_array<<<2,10>>>(da,db,dc);

    cudaMemcpy( hc, dc, 20*sizeof(int), cudaMemcpyDeviceToHost);

    int res = 0;
    for(int i=0;i<20;i++){
        res = res + hc[i];
    }
    
    printf("Dot product result: %d\n", res);

    cudaFree(da );
    cudaFree(db );
    cudaFree(dc );
    free(ha);
    free(hb);
    free(hc);
}