#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void sobel_kernel(unsigned char *input, unsigned char *output,
							 int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
		float Gx =
			-input[(y - 1) * width + (x - 1)] +
			input[(y - 1) * width + (x + 1)] - 2 * input[y * width + (x - 1)] +
			2 * input[y * width + (x + 1)] - input[(y + 1) * width + (x - 1)] +
			input[(y + 1) * width + (x + 1)];

		float Gy =
			-input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] -
			input[(y - 1) * width + (x + 1)] +
			input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] +
			input[(y + 1) * width + (x + 1)];

		output[y * width + x] =
			(unsigned char)fminf(255.0f, sqrtf(Gx * Gx + Gy * Gy));
	}
}

void launch_sobel(unsigned char *input, unsigned char *output, int width,
				  int height) {
	unsigned char *d_in, *d_out;
	size_t size = width * height * sizeof(unsigned char);

	cudaMalloc(&d_in, size);
	cudaMalloc(&d_out, size);

	cudaMemcpy(d_in, input, size, cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridSize((width + 15) / 16, (height + 15) / 16);

	sobel_kernel<<<gridSize, blockSize>>>(d_in, d_out, width, height);

	cudaMemcpy(output, d_out, size, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}
