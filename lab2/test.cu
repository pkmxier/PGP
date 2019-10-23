#include <stdio.h>
#include <math.h>


texture<uchar4, 2, cudaReadModeElementType> tex;

__device__ int Intensity(uchar4 p) {
	return p.x * 0.299 + p.y * 0.587 + p.z * 0.114;
}

__global__ void kernel(uchar4 *dst, int w, int h) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	int Gx, Gy;
	int grad;
	for (x = idx; x < w; x += offsetx) {
		for (y = idy; y < h; y += offsety) {
			Gx = Intensity(tex2D(tex, x, y)) - 
			     Intensity(tex2D(tex, x + 1, y + 1));
			Gy = Intensity(tex2D(tex, x, y + 1)) - 
			     Intensity(tex2D(tex, x + 1, y));
			grad = min(int(sqrt(float(Gx * Gx + Gy * Gy))), 255);
			dst[y * w + x] = make_uchar4(grad, grad, grad, 1);
		}
	}
}

int main() {
    char input_file[] = "in.data", output_file[] = "out.data";

	int w, h;
	FILE *in = fopen(input_file, "rb");
	fread(&w, sizeof(uchar4), 1 , in);
	fread(&h, sizeof(uchar4), 1 , in);
	uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * h * w);
	fread(data, sizeof(uchar4), h * w, in);
	fclose(in);
	
	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	cudaMallocArray(&arr, &ch, w, h);
	cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice);

	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false; 

	cudaBindTextureToArray(tex, arr, ch);
	uchar4 *dev_data;
	cudaMalloc(&dev_data, sizeof(uchar4) * h * w);
	
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
	
	kernel<<<dim3(512, 512), dim3(16, 16)>>>(dev_data, w, h);
	
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    float time;
    cudaEventElapsedTime(&time, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    
    printf("%lf\n", time);
	
	
	cudaMemcpy(data, dev_data, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost);

	FILE *out = fopen(output_file, "wb");
	fwrite(&w, sizeof(uchar4), 1, out);
	fwrite(&h, sizeof(uchar4), 1, out);
	fwrite(data, sizeof(uchar4), h * w, out);
	fclose(out);

	cudaUnbindTexture(tex);
	cudaFreeArray(arr);
	cudaFree(dev_data);
	free(data);

	return 0;
}
