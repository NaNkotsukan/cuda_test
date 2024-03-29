#include <iostream>
#include <random>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "matrix_file.cpp"

void showArray(float *a, int n){
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            std::cout<<a[i*n+j]<<" ";
        }
        std::cout<<std::endl;
    }
}

__global__ void eye(float *a, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=n)return;
    for(int i = 0;i<n;++i){
        a[i*n+index] = i==index ? 1 : 0;
    }
}

inline void mkide(float *a, int n){
    eye<<<n/32 + (n%32 ? 1 : 0), 32>>>(a, n);
}

void inv(float *A, float *B, int n){
    cusolverStatus_t status, status0, status1, status2;
    cusolverDnHandle_t handle;
    status = cusolverDnCreate(&handle);
    // float* A;
    // float* B;
    // cudaMalloc(&B, sizeof(float)*n*n);
    mkide(B, n);
    int worksize;
    float *workspace;
    int *devInfo;
    int *devIpiv;
    cudaMalloc(&devInfo, sizeof(int));
    cudaMalloc(&devIpiv, sizeof(int)*n);

    status0 = cusolverDnSgetrf_bufferSize(handle,
        n, n,
        A,
        n,
        &worksize);
    cudaMalloc(&workspace, sizeof(float)*worksize);
    // std::cout << "worksize:" << worksize << std::endl;
    // std::cout << "status:" << status << std::endl;
    
    status1 = cusolverDnSgetrf(handle,
        n, n,
        A,
        n,
        workspace,
        devIpiv,
        devInfo);
    // std::cout << "status:" << status << std::endl;
    cudaDeviceSynchronize();

    status2 = cusolverDnSgetrs(handle,
        CUBLAS_OP_N,
        n,
        n,
        A,
        n,
        devIpiv,
        B,
        n,
        devInfo);
    // std::cout << "status:" << status << status0 << status1 << status2 << std::endl;
    // std::cout << CUSOLVER_STATUS_SUCCESS<<" "<< CUSOLVER_STATUS_NOT_INITIALIZED<<" "<<CUSOLVER_STATUS_INVALID_VALUE <<" "<<CUSOLVER_STATUS_ARCH_MISMATCH<<" "<<CUSOLVER_STATUS_EXECUTION_FAILED<<" " <<CUSOLVER_STATUS_INTERNAL_ERROR << std::endl;
 
}
                
int main(){
    int N = 10000;
    std::mt19937 mt(982359349);
    std::uniform_real_distribution<> MyRand(-1.0, 1.0);
    
    float *A_ = (float*)calloc(N*N, sizeof(float));
    float *B_ = (float*)calloc(N*N, sizeof(float));
    float *A;
    float *B;
    for(int i = 0;i<N*N;i++){
        A_[i] = MyRand(mt);
    }

    // for(int j = 1;j<N;j*=10){
    //     for(int i=1;i<10;++i){
    //         int n = j*i;
    //         if(n>N) break;
    //         cudaEvent_t start, stop;
    //         cudaEventCreate(&start);
    //         cudaEventCreate(&stop);
    //         cudaEventRecord(start);

    //         cudaMalloc(&A, sizeof(float)*n*n);
    //         cudaMalloc(&B, sizeof(float)*n*n);
    //         cudaMemcpy(A, A_, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    //         inv(A, B, n);
    //         cudaMemcpy(B_, B, sizeof(float)*n*n, cudaMemcpyDeviceToHost);

    //         cudaEventRecord(stop);
    //         cudaEventSynchronize(stop);
    //         float milliseconds = 0;
    //         cudaEventElapsedTime(&milliseconds, start, stop);
    //         cudaEventDestroy(start);
    //         cudaEventDestroy(stop);
    //         std::cout<<n<<"\t"<<milliseconds<<std::endl;
    //     }
    // }

    N = 5000;
    // showArray(A, N);
    cudaMalloc(&A, sizeof(float)*N*N);
    cudaMalloc(&B, sizeof(float)*N*N);
    cudaMemcpy(A, A_, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    inv(A, B, N);
    cudaDeviceSynchronize();
    cudaMemcpy(B_, B, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
    save("A.csv", A_, N, N);
    save("B.csv", B_, N, N);
    // showArray(B, N);

    return 0;
}