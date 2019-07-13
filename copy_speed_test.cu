#include <iostream>
#include <random>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void mkarry(float *a){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    a[index] = 0;
}


int main(){
    float *a;
    int N = 8*32*1000*1000*sizeof(float);
    float milliseconds;
    cudaMalloc(&a, N);
    cudaMemset(&a, 0, N);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    milliseconds = 0;
    cudaEventRecord(start);
    cudaMemset(&a, 0, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<N<<"\t"<<milliseconds<<std::endl;

    milliseconds = 0;
    cudaEventRecord(start);
    mkarry<<<N/32, 32>>>(a);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<N<<"\t"<<milliseconds<<std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}