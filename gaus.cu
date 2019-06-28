#include <iostream>
#include <chrono>
#include <random>


#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

void inputArray(float *a, int n){
    for(int i=0;i<n*n;++i){
        std::cin >> a[i];
    }
}

void showArray(float *a, int n){
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            std::cout<<a[i*n+j]<<" ";
        }
        std::cout<<std::endl;
    }
}


__global__ void mkide(float *a, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    a[index*n+index] = 1;
}

__global__ void divRow(float *a, float *b, float t, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d\n", index);
    if(index<n){
        a[index] /= t;
        b[index] /= t;
    }
}

// __global__ void GaussElimination(float *a, float *b, int n, int i){
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if(index != i && index < n){
//         float t = a[index*n+i];
//         for(int k = 0; k < n; ++k){
//             a[index*n+k] -= a[i*n+k]*t;
//             b[index*n+k] -= b[i*n+k]*t;
//         }
//     }
// }

__global__ void GaussElimination(float *a, float *b, float *t, int n, int i){
    //-- a?b[i*n:i*n+k]はキャッシュorシェアード --、tはコンスタントメモリを使うべき?
    //無駄になるスレッド多いし要修正
    int col = blockIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.x;
    if(col >= n || row >= n || i == col || i == row) return;
    int index = col * n + row;

    printf("%d %d\n", col, row);
    // a[index*n+k] -= __ldg(a[i*n+k])*t[col];
    // b[index*n+k] -= __ldg(b[i*n+k])*t[col];
    a[index*n+row] -= a[i*n+row]*t[col];
    b[index*n+row] -= b[i*n+row]*t[col];
}

__constant__ float t[10000];
__host__ void GaussJordanGpuOptimize(float *a, float *b, int n){
    int blockSize = n/32 + (n%32 ? 1 : 0);
    mkide<<<blockSize,32>>>(b, n);
    cudaDeviceSynchronize();
    // for(int i = 0; i < n; ++i){
    //     b[i*n+i] = 1;
    // }
    dim3 thread(32);
    dim3 block(n, n/32 + n%32!=0);

    printf("%d, %d, %d\n", n, n/32 + n%32!=0, 32);
    // float *t_;
    // cudaMalloc(&t_, sizeof(float)*n);
    for(int i = 0;i<n; ++i){
        int in = i*n;
        divRow<<<blockSize, 32>>>(&a[in], &b[in], a[in+i], n);
        cudaDeviceSynchronize();
        cudaMemcpyToSymbol(t, a, n, in, cudaMemcpyDeviceToDevice);
        GaussElimination<<<block, thread>>>(a, b, t, n, i);
        cudaDeviceSynchronize();
        std::cout<<i<<" "<<in<<std::endl;
        std::cout<<thread.x<<" "<<thread.y<<" "<<thread.z<<std::endl;
        std::cout<<block.x<<" "<<block.y<<" "<<block.z<<std::endl;
    }
}

// __global__ void GaussJordanGpuOptimize(float *a, float *b, int n){
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if(index>=n)return;
//     a[index*n+index] = 1;
//     __syncthreads(); 
//     for(int i = 0;i<n; ++i){
//         int in = i*n;
//         float t = a[in+i];
//         a[in+index] /= t;
//         b[in+index] /= t;
//         __syncthreads();
//         for(int j=0;j<n;++j){
//             if(j != i){
//                 float t = a[j*n+i];
//                 a[j*n+index] -= a[in+index]*t;
//                 b[j*n+index] -= b[in+index]*t;
//             }
//         }
//         __syncthreads();
//     }
// }

__global__ void GaussJordanGpu(float *a, float *b, int n){
    for(int i = 0; i < n; ++i){
        b[i*n+i] = 1;
    }

    for(int i = 0; i < n; ++i){
        float t = a[i*n+i];
        for(int j = 0; j < n; ++j){
            a[i*n+j] /= t;
            b[i*n+j] /= t;
        }
        for(int j = 0; j < n; ++j){
            if(i != j){
                float t = a[j*n+i];
                for(int k = 0; k < n; ++k){
                    a[j*n+k] -= a[i*n+k]*t;
                    b[j*n+k] -= b[i*n+k]*t;
                }
            }
        }
    }
}

void GaussJordan(float *a, float *b, int n){
    for(int i = 0; i < n; ++i){
        b[i*n+i] = 1;
    }

    for(int i = 0; i < n; ++i){
        float t = a[i*n+i];
        for(int j = 0; j < n; ++j){
            a[i*n+j] /= t;
            b[i*n+j] /= t;
        }
        for(int j = 0; j < n; ++j){
            if(i != j){
                float t = a[j*n+i];
                for(int k = 0; k < n; ++k){
                    a[j*n+k] -= a[i*n+k]*t;
                    b[j*n+k] -= b[i*n+k]*t;
                }
            }
        }
    }
}


int main(){
    int n = 10;
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
    // cudaMalloc(&a, n*n*sizeof(float));
    // cudaMalloc(&b, n*n*sizeof(float));
    cudaMallocManaged(&a, n*n*sizeof(float));
    cudaMallocManaged(&b, n*n*sizeof(float));
    for(int i = 0;i<n*n;i++){
        a[i] = MyRand(mt);
        // std::cout<<ref[i]<<" ";
    }

    
    // float *a;
    // CHECK(cudaMallocManaged(&a, n*n*sizeof(float)));
    // float *b;
    // CHECK(cudaMallocManaged(&b, n*n*sizeof(float)));

    
    
    
    // for(int j = 10;j<10000;j*=10){
    //     for(int i=1;i<9;++i){
    //         int N = j*i;
    //         int blockSize = N/32 + (N%32 ? 1 : 0);
    //         cudaMemcpy(ref, a, n*n*sizeof(float), cudaMemcpyHostToDevice);
    //         start = std::chrono::system_clock::now();
    //         // GaussJordan(a, b, N);
    //         GaussJordanGpuOptimize(a, b, N);
    //         // GaussJordanGpuOptimize<<<blockSize, 32>>>(a, b, N);
    //         cudaDeviceSynchronize();
            
    //         end = std::chrono::system_clock::now();
            
            
    //         double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    //         std::cout<<N<<"\t"<<elapsed<<std::endl;
            
            
    //     }
        
    // }
    // std::cout<<"hoge"<<std::endl;
    // inputArray(a, n);
    // std::cout<<"hoge"<<std::endl;
    // cudaMemcpy(ref, a, n*n*sizeof(float), cudaMemcpyHostToDevice);
    showArray(a, n);
    GaussJordanGpuOptimize(a, b, n);
    cudaDeviceSynchronize();
    showArray(a, n);
    showArray(b, n);


    return 0;
}