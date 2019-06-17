#include <iostream>

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

// __global__ void GaussJordan(float *a, float *b, int n){
//     for(int i = 0; i < n; ++i){
//         b[i*n+i] = 1;
//     }

//     for(int i = 0; i < n; ++i){
//         float t = a[i*n+i];
//         for(int j = 0; j < n; ++j){
//             a[i*n+j] /= t;
//             b[i*n+j] /= t;
//         }
//         for(int j = 0; j < n; ++j){
//             if(i != j){
//                 float t = a[j*n+i];
//                 for(int k = 0; k < n; ++k){
//                     a[j*n+k] -= a[i*n+k]*t;
//                     b[j*n+k] -= b[i*n+k]*t;
//                 }
//             }
//         }
//     }
// }

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
    int n;
    std::cin>>n;

    // float *a = (float*)calloc(n*n, sizeof(float));
    // float *b = (float*)calloc(n*n, sizeof(float));
    
    float *a;
    CHECK(cudaMallocManaged(&a, n*n*sizeof(float)));
    float *b;
    CHECK(cudaMallocManaged(&b, n*n*sizeof(float)));

    inputArray(a, n);
    GaussJordan(a, b, n);
    showArray(b, n);


    return 0;
}