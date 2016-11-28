#include <stdio.h>
#include <algorithm>
#include <getopt.h>
#include <time.h>
#include <string.h>
#include <cmath>

#include "CycleTimer.h"

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

extern void writePPMImage(int* data, int width, int height, const char *filename, int maxIterations);

int verifyResult (int *gold, int *result, int width, int height) {

    int i, j;
    int count = 0;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (abs(gold[i * width + j] - result[i * width + j]) > 15) {
                printf ("Mismatch : [%d][%d], Expected : %d, Actual : %d\n",i, j, result[i * width + j], gold[i * width + j]);
                count++;
            }
        }
    }
    return count;
}

__global__ void mandel(
    int *d_out, float x0, float y0, float d_dx, float d_dy,
    int height, int width, int count)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= height || j >= width) return;

    int index = (j * width + i);

    float c_re = x0 + i * d_dx;
    float c_im = y0 + j * d_dy;
    float z_im = c_im;
    float z_re = c_re;
    float new_re, new_im;
    int k = 0;

    for (; k < count; ++k)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        new_re = z_re*z_re - z_im*z_im;
        new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    d_out[index] = k;
}

void mandelbrotCuda(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations,
    int *output)
{
    const int ARRAY_BYTES = width * height * sizeof(int);
    int *d_out;
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    // allocate GPU memory
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

	// launch the kernel
    mandel<<<dim3(width), dim3(height)>>>(d_out, x0, y0, dx, dy, height, width, maxIterations);

	// copy back the result array to the CPU
	cudaMemcpy(output, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
}

int main(int argc, char** argv) {

    const unsigned int width = 200;
    const unsigned int height = 200;
    const int maxIterations = 256;

    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;

    // Serial
    int* output_serial = new int[width*height];
    memset(output_serial, 0, width * height * sizeof(int));
    double minSerial = 1e30;
    for(int i = 0; i < 3; ++i)
    {
        double startTime = CycleTimer::currentSeconds();
        mandelbrotSerial(x0, y0, x1, y1, width, height, 0, height, maxIterations, output_serial);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std:min(minSerial, endTime - startTime);
    }

    printf("[mandelbrot serial]:\t\t[%.3f] ms\n", minSerial * 1000);
    writePPMImage(output_serial, width, height, "mandelbrot-serial.ppm", maxIterations);

    // CUDA

    int *output_cuda = new int[width*height];
    memset(output_cuda, 0, width * height * sizeof(int));
    double minCuda = 1e30;
    for (int i = 0; i < 3; ++i)
    {
        double startTime = CycleTimer::currentSeconds();
        mandelbrotCuda(x0, y0, x1, y1, width, height, maxIterations, output_cuda);
        double endTime = CycleTimer::currentSeconds();
        minCuda = std::min(minCuda, endTime - startTime);
    }

    printf("[mandelbrot CUDA]:\t\t[%.3f] ms\n", minCuda * 1000);
    writePPMImage(output_cuda, width, height, "mandelbrot-cuda.ppm", maxIterations);

    // compute speedup
    printf("\t\t\t\t(%.2fx speedup from CUDA implementation)\n", minSerial/minCuda);

    printf("Number of mismatches in Serial Vs CUDA: %d\n", verifyResult(output_cuda, output_serial, width, height));

    delete[] output_serial;
    delete[] output_cuda;
    return 0;
}