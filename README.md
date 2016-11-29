# Raghav Pandya - Assignment 4



### 1.  
GPU data as input: The data (array) is initialized inside GPU array and thus there is no cost associated with transferring data from CPU memory to GPU memory. Also the algorithm is 23X faster compared to serial implementation.
VS
Element wise operation: As for every pixel same algorithm is followed to calculate the color based on number of iterations till the magnitude of the complex number is within 2.
We seperate a kernel function and every thread works on a pixel in parallel.
This approach is much faster compared to previous operation as less number of GPU operations are required and it is much faster (200X) compared to eh serial implementation

 ### 2. Mandelbrot set implementation using CUDA
 
 We can parallelize the part where pixel color is caluclated based on the number of iteration taken by an element to diverge above magnitude 2.
 
 First we allocate the memory on GPU for output image:
 
```
cudaMalloc((void**) &d_out, ARRAY_BYTES); 
```
Then we launch the kernel with number of threads equal to the number of pixels as every threads works on a single pixel

```
mandel<<<dim3(width), dim3(height)>>>(d_out, x0, y0, dx, dy, height, width, maxIterations);
```
Finally we copy back the output image from the GPU to CPU
```
cudaMemcpy(output, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
```

The kernel function looks like:
```
__global__ void mandel(
    int *d_out, float x0, float y0, float d_dx, float d_dy,
    int height, int width, int count)
{
    int i = blockIdx.x;
    int j = threadIdx.x;

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
```

[Source Code]


### Analysis

![General](https://raw.githubusercontent.com/rpandya1990/Gauss-seidel-Parallel-Implementation/master/images/Image%206.png)

![Correctness](https://raw.githubusercontent.com/rpandya1990/Gauss-seidel-Parallel-Implementation/master/images/Image%207.png)

### Charts

![SerialvsParallel](https://raw.githubusercontent.com/rpandya1990/Gauss-seidel-Parallel-Implementation/master/images/Image%201.png)

![StrongScaling](https://raw.githubusercontent.com/rpandya1990/Gauss-seidel-Parallel-Implementation/master/images/Image%202.png)

![Correctness](https://raw.githubusercontent.com/rpandya1990/Gauss-seidel-Parallel-Implementation/master/images/Image%203.png)


### Running the code

To run the Parallel CUDA Mandelbrot set on ELF(Assuming CUDA has been installed):

```sh
$ nvcc mandelbrot_cuda.cu mandelbrotSerial.cpp ppm.cpp tasksys.cpp
```

[Source Code]: <https://github.com/rpandya1990/Parallel-Mandelbrot-set-using-CUDA>