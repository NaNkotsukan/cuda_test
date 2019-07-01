#include <iostream>
#include <chrono>
#include <random>
#include <math.h>



int main(){
    int n = 10000;
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




    for(int j = 10;j<n;j*=10){
        for(int i=1;i<10;++i){
            int N = j*i;
            int blockSize = N/32 + (N%32 ? 1 : 0);
            cudaMemcpy(ref, a, n*n*sizeof(float), cudaMemcpyHostToDevice);
            start = std::chrono::system_clock::now();
            
            cublasSgetriBatched(cublasHandle_t handle,
                int n,
                float *Aarray[],
                int lda,
                int *PivotArray,
                float *Carray[],
                int ldc,
                int *infoArray,
                int batchSize);



            cudaDeviceSynchronize();

            end = std::chrono::system_clock::now();


            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
            std::cout<<N<<"\t"<<elapsed<<std::endl;


        }

    }

    return 0;
}