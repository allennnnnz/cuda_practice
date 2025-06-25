#include <iostream>
#include <cuda_runtime.h>

__global__ void dot_array_p1(int *inputa, int *inputb, int *outputc) {
    int* inputa_begin = blockDim.x * blockIdx.x + inputa;
    int* inputb_begin = blockDim.x * blockIdx.x + inputb;
    int* outputc_begin = blockDim.x * blockIdx.x + outputc;

    //各element相乘
    for(int i =0 ; i<blockDim.x ;i++){
        outputc_begin[i] = inputa_begin[i] * inputb_begin[i];
    } 
    //相加個點
    for(int i=1;i<blockDim.x;i*=2){
        if(threadIdx.x%2*i==0){
            outputc_begin[threadIdx.x] += outputc_begin[threadIdx.x+i];
        }
        
    }
}

__global__ void dot_array_p2(int *result) {
    int id = blockDim.x * threadIdx.x ;
    if(threadIdx.x == 0 || 2 || 4){
        result[id] += result[id+1*blockDim.x];
    }
    __syncthreads();
    if(threadIdx.x == 0 ){
        result[id] += result[id+2*blockDim.x];
        result[id] += result[id+4*blockDim.x];
    }

}


/* 共36個數字分成6block的狀況
__global__ void dot_array_p1(int *inputa, int *inputb, int *outputc) {
    int* inputa_begin = blockDim.x * blockIdx.x + inputa;
    int* inputb_begin = blockDim.x * blockIdx.x + inputb;
    int* outputc_begin = blockDim.x * blockIdx.x + outputc;

    //各element相乘
    for(int i =0 ; i<blockDim.x ;i++){
        outputc_begin[i] = inputa_begin[i] * inputb_begin[i];
    } 
    //相加個點
    if(threadIdx.x == 0 || 2 || 4){
        outputc_begin[threadIdx.x] += outputc_begin[threadIdx.x+1];
    }
    __syncthreads();
    if(threadIdx.x == 0 ){
        outputc_begin[threadIdx.x] += outputc_begin[threadIdx.x+2];
        outputc_begin[threadIdx.x] += outputc_begin[threadIdx.x+4];
    }
}

__global__ void dot_array_p2(int *result) {
    int id = blockDim.x * threadIdx.x ;
    if(threadIdx.x == 0 || 2 || 4){
        result[id] += result[id+1*blockDim.x];
    }
    __syncthreads();
    if(threadIdx.x == 0 ){
        result[id] += result[id+2*blockDim.x];
        result[id] += result[id+4*blockDim.x];
    }

}
*/
int dot_product_cpu(int* a, int* b, int N) {
    int result = 0;
    for (int i = 0; i < N; i++) {
        result += a[i] * b[i];
    }
    return result;
}

int main(){
    int* ha = (int*)malloc(36*sizeof(int));
    int* hb = (int*)malloc(36*sizeof(int));
    int* hc = (int*)malloc(36*sizeof(int));

    for (int i = 0; i < 36; i++) {
        ha[i] = rand() % 10;
        hb[i] = rand() % 10;
        
    }

    int result = dot_product_cpu(ha, hb, 36);
    printf("CPU dot product result: %d\n", result);

    int* da;
    int* db;
    int* dc;

    cudaMalloc( (void**)&da, 36*sizeof(int));
    cudaMalloc( (void**)&dc, 36*sizeof(int));
    cudaMalloc( (void**)&db, 36*sizeof(int));

    cudaMemcpy( da, ha, 36*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( db, hb, 36*sizeof(int), cudaMemcpyHostToDevice);

    dot_array_p1<<<6,6>>>(da,db,dc);
    dot_array_p2<<<1,6>>>(dc);

    cudaMemcpy( hc, dc, 1*sizeof(int), cudaMemcpyDeviceToHost);

    int res = hc[0];

    
    
    printf("Dot product result: %d\n", res);

    cudaFree(da );
    cudaFree(db );
    cudaFree(dc );
    free(ha);
    free(hb);
    free(hc);
}