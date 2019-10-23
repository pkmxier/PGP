#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int n;
    scanf("%d", &n);
    
    int size = n * sizeof(double);
    double *vector = (double *) malloc(size);
    for (int i = 0; i < n; ++i) {
        scanf("%lf", &vector[i]);
    }

    clock_t begin = clock();

    for (int i = 0; i < n; ++i) {
        vector[i] *= vector[i] > 0 ? 1 : -1;
    }

    clock_t end = clock();
 
    printf("%lf\n", (double)(end - begin) / CLOCKS_PER_SEC * 1000);


    return 0;
}
