#include <iostream>
#include <chrono>
#include <random>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void inv(float *a, float *b, int n){

    cublasHandle_t handle; 
    cublasCreate_v2(&handle);

    float *A[1] = {a};
    float *B[1] = {b};
    int *info;
    int *p;

    cudaMalloc(&info, sizeof(int));
    cudaMalloc(&p, n * sizeof(int));
    cublasSgetrfBatched(handle, n, A, n, p, info, 1);
    cublasSgetriBatched(handle, n, A, n, p, B, n, info, 1);
    cudaFree(info);
    cudaFree(p);
    cublasDestroy_v2(handle);
}

void dot(float *a, float *b, float *y, int n){
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            double r = 0;
            for(int k = 0; k < n; ++k)
            {
                r += a[i*n+k] * b[k*n+j];
            }
            y[i*n+j] = abs(r)<0.001?0: r;
        }
    }
}

int main(){
    int n = 100;
    // std::cin>>n;

    std::chrono::system_clock::time_point  start, end;
    std::mt19937 mt(982359349);
    std::uniform_real_distribution<> MyRand(-1.0, 1.0);
    float *ref = (float*)calloc(n*n, sizeof(float));
    for(int i = 0;i<n*n;i++){
        ref[i] = MyRand(mt);
        // std::cout<<ref[i]<<" ";
    }
    std::cout<<std::endl;

    float *a, *b;
    // *a = (float*)calloc(n*n, sizeof(float));
    // *b = (float*)calloc(n*n, sizeof(float));
    cudaMalloc(&a, n*n*sizeof(float));
    cudaMalloc(&b, n*n*sizeof(float));
    float *buf = (float*)calloc(n*n, sizeof(float));
    float *a_ = (float*)calloc(n*n, sizeof(float));
    float *b_ = (float*)calloc(n*n, sizeof(float));
    float *y_ = (float*)calloc(n*n, sizeof(float));
    // cudaMallocManaged(&a, n*n*sizeof(float));
    // cudaMallocManaged(&b, n*n*sizeof(float));
    for(int i = 0;i<n*n;i++){
        a_[i] = MyRand(mt);
        // std::cout<<ref[i]<<" ";
    }
    // for(int i = 0;i<n;i++){
    //     a__[i] = MyRand(mt);
    // }

    // float *a;
    // CHECK(cudaMallocManaged(&a, n*n*sizeof(float)));
    // float *b;
    // CHECK(cudaMallocManaged(&b, n*n*sizeof(float)));




    // for(int j = 10;j<n;j*=10){
    //     for(int i=1;i<10;++i){
    //         int N = j*i;
    //         cudaMemcpy(ref, a, n*n*sizeof(float), cudaMemcpyHostToDevice);
    //         start = std::chrono::system_clock::now();
    //         inv(a, b, N);
    //         end = std::chrono::system_clock::now();


    //         double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    //         std::cout<<N<<"\t"<<elapsed<<std::endl;


    //     }

    // }

    double sum=0;
    cudaMemcpy(a, a_, n*n*sizeof(float), cudaMemcpyHostToDevice);
    
    for(int i = 0;i<n*n;++i)sum+=a_[i];
    std::cout<<sum<<std::endl;
    // showArray(a_, n);
    inv(a, b, n);
    cudaDeviceSynchronize();
    cudaMemcpy(buf, a, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    // showArray(buf, n);
    cudaMemcpy(b_, b, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    // showArray(b_, n);
    std::cout<<"meow"<<std::endl;
    dot(a_, b_, y_, n);
    sum = 0;
    for(int i = 0;i<n*n;++i)sum+=buf[i];
    std::cout<<sum<<std::endl;
    sum = 0;
    for(int i = 0;i<n*n;++i)sum+=y_[i];
    std::cout<<sum<<std::endl;
    // showArray(y_, n);




    return 0;
}