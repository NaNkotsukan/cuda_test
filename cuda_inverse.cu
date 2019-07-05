#include <iostream>
#include <random>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>



void inv(float *a, float *b, int n){
    cusolverStatus_t status;
    cusolverDnHandle_t handle;
    status = cusolverDnCreate(&handle);
    float* A;
    cudaMalloc(&A, sizeof(float)*n*n);
    cudaMemcpy(A, A, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    int worksize;
    status = cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, 
        n,
        A,
        n,
        &worksize);
    std::cout << "worksize:" << worksize << std::endl;
    std::cout << "status:" << status << std::endl;
    float *workspace;
    cudaMalloc(&workspace, sizeof(float)*worksize);
    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));
    status = cusolverDnSpotrf(handle,
           CUBLAS_FILL_MODE_LOWER,
           n,
           A,
           n,
           workspace,
           worksize,
           devInfo);
    std::cout << "status:" << status << std::endl;

           
    status = cusolverDnSpotri_bufferSize(handle,
                 CUBLAS_FILL_MODE_LOWER,
                 n,
                 A,
                 n,
                 worksize);
    std::cout << "worksize:" << worksize << std::endl;
    std::cout << "status:" << status << std::endl;
                 
    status = cusolverDnSpotri(handle,
            CUBLAS_FILL_MODE_LOWER,
            n,
            A,
            n,
            workspace,
            worksize,
            devInfo);
    std::cout << "status:" << status << std::endl;
    cudaMemcpy(b, A, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
}
                
int main(){
    int N = 1000;
    std::mt19937 mt(982359349);
    std::uniform_real_distribution<> MyRand(-1.0, 1.0);
    
    float *A = (float*)calloc(N*N, sizeof(float));
    float *B = (float*)calloc(N*N, sizeof(float));
    for(int i = 0;i<N*N;i++){
        A[i] = MyRand(mt);
    }

    for(int j = 10;j<N;j*=10){
        for(int i=1;i<10;++i){
            int n = j*i;
            if(n>N) break;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            inv(A, B, n);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            std::cout<<n<<"\t"<<milliseconds<<std::endl;
        }
    }


    return 0;
}