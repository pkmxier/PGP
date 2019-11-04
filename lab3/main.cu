#include <stdio.h>
#include <math.h>
#include <float.h>


typedef struct {
    int x, y;
} Point;

typedef struct {
    float4 avg;
    double inverse_cov[3][3];
    double log_det;
} Class;

__constant__ Class dev_class[32];

float4 Average(uchar4 *data, int w, int h, Point *class_points, int point_n) {
    float4 result = make_float4(0, 0, 0, 0);
    
    for (int i = 0; i < point_n; ++i) {
        Point p = class_points[i];
        uchar4 pixel = data[p.y * w + p.x];
        
        result.x += pixel.x;
        result.y += pixel.y;
        result.z += pixel.z;
    }
    
    result.x /= point_n;
    result.y /= point_n;
    result.z /= point_n;
    
    return result;
}

void CalculateCovariance(double cov[3][3], uchar4 *data, int w, int h,
                         Point *class_points, int point_n, float4 avg) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cov[i][j] = 0;
        }
    }
    
    for (int i = 0; i < point_n; ++i) {
        Point p = class_points[i];
        uchar4 pixel = data[p.y * w + p.x];
        double delta[3] = {pixel.x - avg.x, pixel.y - avg.y, pixel.z - avg.z};
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                cov[i][j] += delta[i] * delta[j];
            }
        }
    }
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cov[i][j] /= point_n - 1;
        }
    }
}

double Determinant(double cov[3][3]) {
    double det = 0;
    
    for (int i = 0; i < 3; ++i) {
        det += cov[0][i] *
               cov[1][(i + 1) % 3] *
               cov[2][(i + 2) % 3];
        det -= cov[0][(i + 2) % 3] *
               cov[1][(i + 1) % 3] *
               cov[2][i];
    }
    
    return det;
}

void Inverse(double in[3][3], double out[3][3]) {
    double det = Determinant(in);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = in[(j + 1) % 3][(i + 1) % 3] * in[(j + 2) % 3][(i + 2) % 3] -
                        in[(j + 1) % 3][(i + 2) % 3] * in[(j + 2) % 3][(i + 1) % 3];
            out[i][j] /= det;
        }
    }
}

__device__ double MaxLikehoodEstimation(uchar4 p, int class_idx) {
    Class c = dev_class[class_idx];
    double delta[3] = {p.x - c.avg.x, p.y - c.avg.y, p.z - c.avg.z};
    
    double temp[3] = {0,};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            temp[i] += delta[j] * c.inverse_cov[j][i];
        }
    }
    
    double result = -c.log_det;
    
    for (int i = 0; i < 3; ++i) {
        result -= temp[i] * delta[i];
    }
    
    return result;
}

__global__ void kernel(uchar4 *image, int w, int h, int class_count) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int i, j;
	
	int class_idx;
	int max_idx = 0;
	double value;
	double max_value;
	
	for (i = idx; i < w; i += offsetx) {
	    for (j = idy; j < h; j += offsety) {
	        uchar4 pixel = image[j * w + i];
	        
	        max_value = INT_MIN;
	        
	        for (class_idx = 0; class_idx < class_count; ++class_idx) {
	            value = MaxLikehoodEstimation(pixel, class_idx);
	            if (value > max_value) {
	                max_idx = class_idx;
	                max_value = value;
	            }
	        }
	        
	        image[j * w + i] = make_uchar4(pixel.x, pixel.y, pixel.z, max_idx);
	    }
	}
}

int main() {
    char input_file[256], output_file[256];
    int class_count;

    scanf("%s", input_file);
    scanf("%s", output_file);
    scanf("%d", &class_count);
    
    Point *class_points[class_count];
    
	int w, h;
	FILE *in = fopen(input_file, "rb");
	fread(&w, sizeof(uchar4), 1 , in);
	fread(&h, sizeof(uchar4), 1 , in);
	uchar4 *data = (uchar4*) malloc(sizeof(uchar4) * h * w);
	fread(data, sizeof(uchar4), h * w, in);
	fclose(in);
	
    int point_n[class_count];
    for (int i = 0; i < class_count; ++i) {
        scanf("%d", &point_n[i]);
        class_points[i] = (Point *) malloc(sizeof(Point) * point_n[i]);
        
        for (int j = 0; j < point_n[i]; ++j) {
            scanf("%d%d", &class_points[i][j].x, &class_points[i][j].y);
        }
    }
	
	Class class_arr[class_count];
	double cov[3][3];
	for (int i = 0; i < class_count; ++i) {
	    class_arr[i].avg = Average(data, w, h, class_points[i], point_n[i]);
	    CalculateCovariance(cov, data, w, h, class_points[i], point_n[i], class_arr[i].avg);
	    Inverse(cov, class_arr[i].inverse_cov);
	    class_arr[i].log_det = log(Determinant(cov));
	}
	
	uchar4 *dev_data;
	cudaMalloc(&dev_data, sizeof(uchar4) * h * w);
	cudaMemcpy(dev_data, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dev_class, class_arr, sizeof(Class) * class_count);
	
	kernel<<<dim3(32,32), dim3(32, 32)>>>(dev_data, w, h, class_count);
	
	cudaMemcpy(data, dev_data, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost);
	
	FILE *out = fopen(output_file, "wb");
	fwrite(&w, sizeof(uchar4), 1, out);
	fwrite(&h, sizeof(uchar4), 1, out);
	fwrite(data, sizeof(uchar4), h * w, out);
	fclose(out);

	cudaFree(dev_data);
	
    for (int i = 0; i < class_count; ++i) {
        free(class_points[i]);
    }
	
	free(data);

	return 0;
}
