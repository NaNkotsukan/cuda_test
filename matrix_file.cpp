#include<iostream>
#include<fstream>

template<typename T>
void save(const char* filename, T data, int m, int n){
    std::ofstream f(filename);
    for(int i = 0;i < m;++i){
        int j;
        for(j = 0;j < n-1;++j){
            f<<data[i*n+j]<<",";
        }
        f<<data[i*n+j]<<"\n";
    }
    f.close();
}