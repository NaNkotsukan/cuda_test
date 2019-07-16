#include <iostream>
#include <random>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>


void showArray(float *a, int n){
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            std::cout<<a[i*n+j]<<" ";
        }
        std::cout<<std::endl;
    }
}

int main(){
    cublasHandle_t cublas_handle;
    int n_samples, n_features, n_targets;
    n_samples = 5;
    n_features = 10;
    n_targets = 15;
    float *W;
    float *X, *Y;
    cudaMallocManaged(&X, n_samples*n_features*sizeof(float));
    cudaMallocManaged(&Y, n_samples*n_targets*sizeof(float));
    cudaMallocManaged(&W, n_features*n_targets*sizeof(float));
    float *coef_matrix;
    cudaMallocManaged(&coef_matrix, n_features * n_features * sizeof(float));
    for(int i = 0;i<n_samples*n_features;++i){
        X[i] = i;
        // Y[i] = i;
    }
    for(int i = 0;i<n_features*n_targets;++i){
        W[i] = i;
        // Y[i] = i;
    }

    for(int i = 0;i < n_features * n_features;++i){
        coef_matrix[i]= 0;
    }
    float rho = 1.0;            

    float inv_n_samp = 1.0f/n_samples;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        n_features, n_features, n_samples,
        1.0f/n_samples,
        X, n_samples,
        W, n_samples,
        rho,
        Y,
        n_features);
    cudaDeviceSynchronize();
    showArray(Y, n_features);

    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        n_features, n_features, n_samples,
        1.0f/n_samples,
        X, n_samples,
        W, n_samples,
        rho,
        Y,
        n_features);

    cudaDeviceSynchronize();
    showArray(Y, n_features);
    

    return 0;
}

