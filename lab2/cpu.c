#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>


int min(int a, int b) {
    return a > b ? b : a;
}

struct uchar4 {
    char x, y, z, alpha;
};

typedef struct uchar4 uchar4;

int Intensity(uchar4 p) {
	return p.x * 0.299 + p.y * 0.587 + p.z * 0.114;
}

void convert(uchar4 *data, uchar4 *out, int w, int h) {
    int Gx, Gy, grad;
    uchar4 p;
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            int i1 = min(i, h - 1),
                i2 = min(i + 1, h - 1),
                j1 = min(j, w - 1),
                j2 = min(j + 1, w - 1);
            
			Gx = Intensity(data[j1 * w + i1]) - 
			     Intensity(data[j2 * w + i2]);
			Gy = Intensity(data[j2 * w + i1]) - 
			     Intensity(data[j1 * w + i2]);
			grad = min((int)sqrt(Gx * Gx + Gy * Gy), 255);
			p.x = grad;
			p.y = grad;
			p.z = grad;
			p.alpha = 1;
			out[j * w + i] = p;
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
	uchar4 *out_data = (uchar4*)malloc(sizeof(uchar4) * h * w);
	fread(data, sizeof(uchar4), h * w, in);
	fclose(in);
	
    clock_t begin = clock();
	convert(data, out_data, w, h);
    clock_t end = clock();
 
    printf("%.2lf\n", (double)(end - begin) / CLOCKS_PER_SEC * 1000);

	FILE *out = fopen(output_file, "wb");
	fwrite(&w, sizeof(uchar4), 1, out);
	fwrite(&h, sizeof(uchar4), 1, out);
	fwrite(data, sizeof(uchar4), h * w, out);
	fclose(out);

	free(data);

	return 0;
}
