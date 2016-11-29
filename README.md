# Mandelbrot set implementation using CUDA
 
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

![General](https://raw.githubusercontent.com/rpandya1990/Parallel-Mandelbrot-set-using-CUDA/master/Images/Analysis.png)

### Charts
Time taken by CUDA implementation:

![Parallel](https://raw.githubusercontent.com/rpandya1990/Parallel-Mandelbrot-set-using-CUDA/master/Images/Screen%20Shot%202016-11-28%20at%209.02.25%20PM.png)

CUDA speedup comapred to serial implementation: 

![Speedup](https://raw.githubusercontent.com/rpandya1990/Parallel-Mandelbrot-set-using-CUDA/master/Images/Graph%202.png)



### Running the code

To run the Parallel CUDA Mandelbrot set on ELF(Assuming CUDA has been installed):

```sh
$ nvcc mandelbrot_cuda.cu mandelbrotSerial.cpp ppm.cpp tasksys.cpp
```

[Source Code]: <https://github.com/rpandya1990/Parallel-Mandelbrot-set-using-CUDA>