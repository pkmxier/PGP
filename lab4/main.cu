#include <stdio.h>
#include <math.h>
#include <float.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#define CSC(call)  					\
do {								\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);					\
	}								\
} while(0)

__global__ void subtract_row(double *matrix, int n, int column) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int i, j;
    
    double coeff;
    double divisor = matrix[column * n + column];
    for (i = 1 + column + idx; i < n; i += offsetx) {
        coeff = matrix[column * n + i] / divisor;
        for (j = 1 + column + idy; j < n + 1; j += offsety) {
            matrix[j * n + i] -= coeff * matrix[j * n + column];
        }
    }
}

__global__ void reverse_subtract_row(double *matrix, int n, int column) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    int i;
    
    double coeff;
    double divisor = matrix[column * n + column];
    for (i = idx; i < column; i += offsetx) {
        coeff = matrix[column * n + i] / divisor;
        matrix[n * n + i] -= coeff * matrix[n * n + column];
    }
}

__global__ void normalize(double *matrix, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    int i;
    
    for (i = idx; i < n; i += offsetx) {
        matrix[n * n + i] /= matrix[i * n + i];
    }
}

__global__ void swap_rows(double *matrix, int n, int column, int max_row) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    int i;
    double tmp;
    
    for (i = idx + column; i < n + 1; i+= offsetx) {
        tmp = matrix[i * n + column];
        matrix[i * n + column] = matrix[i * n + max_row];
        matrix[i * n + max_row] = tmp;
    }
}

struct compare {
	__host__ __device__ bool operator ()(double lhs, double rhs) {
        return fabs(lhs) < fabs(rhs);
	}
};

void solve(double *matrix, int n) {
    for (int column = 0; column < n; ++column) {
        thrust::device_ptr<double> thrust_matrix =
            thrust::device_pointer_cast(matrix) + n * column;
        int max_row = thrust::max_element(thrust_matrix + column,
                                          thrust_matrix + n, compare()) - thrust_matrix;
        if (max_row >= n) {
            continue;
        }
        
        swap_rows<<<32, 32>>>(matrix, n, column, max_row);
        subtract_row<<<dim3(32, 32), dim3(32, 32)>>>(matrix, n, column);
    }
    
    for (int column = n - 1; column >= 0; --column) {
        reverse_subtract_row<<<32, 32>>>(matrix, n, column);
    }
    
    normalize<<<32, 32>>>(matrix, n);
}

int main() {
    int n;
    scanf("%d", &n);
    double *matrix = (double *) malloc(sizeof(double *) * (n + 1) * n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%lf", matrix + j * n + i);
        }
    }
    
    for (int i = 0; i < n; ++i) {
        scanf("%lf", matrix + n * n + i);
    }
	
	double *device_matrix;
	
	CSC(cudaMalloc(&device_matrix, sizeof(double) * (n + 1) * n));
    CSC(cudaMemcpy(device_matrix, matrix, sizeof(double) * (n + 1) * n, cudaMemcpyHostToDevice));
	
	solve(device_matrix, n);
	
	CSC(cudaMemcpy(matrix + n * n, device_matrix + n * n, sizeof(double) * n, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < n; ++i) {
        printf("%.10e ", matrix[n * n + i]);
    }
    
    printf("\n");
    
	CSC(cudaFree(device_matrix));
	free(matrix);
	return 0;
}
