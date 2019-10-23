#include <stdlib.h>
#include <stdio.h>

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)


__global__ void kernel(double *vector, int n) {
    int offset = blockDim.x * gridDim.x;
    
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += offset) {
        vector[i] *= vector[i] < 0 ? -1 : 1;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    
    int size = n * sizeof(double);
    double *vector = (double *) malloc(size);
    for (int i = 0; i < n; ++i) {
        scanf("%lf", &vector[i]);
    }
    
    double *device_vector;

    CSC(cudaMalloc(&device_vector, size));
    CSC(cudaMemcpy(device_vector, vector, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, end;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&end));
    CSC(cudaEventRecord(start));

    kernel<<<1024, 1024>>>(device_vector, n);
    
    CSC(cudaGetLastError());

    CSC(cudaEventRecord(end));
    CSC(cudaEventSynchronize(end));
    
    float time;
    CSC(cudaEventElapsedTime(&time, start, end));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(end));

    printf("Time = %f ms\n", time);

    CSC(cudaMemcpy(vector, device_vector, size, cudaMemcpyDeviceToHost));
    CSC(cudaFree(device_vector));

    /*
    for (int i = 0; i < n; ++i) {
        printf("%.10e ", vector[i]);
    }
    */

    printf("\n");
    free(vector);
    
    return 0;
}
