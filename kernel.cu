#include "kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define gpuErrchk(ans)                                                         \
	{                                                                          \
		gpuAssert((ans), __FILE__, __LINE__);                                  \
	}
inline void gpuAssert(cudaError_t code, const char *file, int line,
					  bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		if (abort)
			exit(code);
	}
}

__constant__ float gaussian_5x5[25] = {
	0.003, 0.013, 0.022, 0.013, 0.003, 0.013, 0.059, 0.097, 0.059,
	0.013, 0.022, 0.097, 0.159, 0.097, 0.022, 0.013, 0.059, 0.097,
	0.059, 0.013, 0.003, 0.013, 0.022, 0.013, 0.003};

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

__global__ void gaussian_blur_kernel(unsigned char *input,
									 unsigned char *output, int width,
									 int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
		float blur_val = 0.0f;

		for (int i = -2; i <= 2; i++) {
			for (int j = -2; j <= 2; j++) {
				float weight = gaussian_5x5[(i + 2) * 5 + (j + 2)];
				blur_val += weight * input[(y + i) * width + (x + j)];
			}
		}
		output[y * width + x] = (unsigned char)blur_val;
	}
}

__global__ void invert_kernel(unsigned char *input, unsigned char *output,
							  int width, int height) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < width * height) {
		output[tid] = 255 - input[tid];
	}
}

__global__ void copy_kernel(unsigned char *input, unsigned char *output,
							int width, int height) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < width * height) {
		output[tid] = input[tid];
	}
}

// helper procs
void allocate_buffers(unsigned char **d_in_ptr, unsigned char **d_out_ptr,
					  size_t size) {
	gpuErrchk(cudaMalloc((void **)d_in_ptr, size));
	gpuErrchk(cudaMalloc((void **)d_out_ptr, size));
}

void free_buffers(unsigned char *d_in, unsigned char *d_out) {
	cudaFree(d_in);
	cudaFree(d_out);
}

void upload_to_gpu(unsigned char *d_in, unsigned char *h_in, size_t size) {
	gpuErrchk(cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice));
}

void download_from_gpu(unsigned char *h_out, unsigned char *d_out,
					   size_t size) {
	gpuErrchk(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
}

// launcher execs
void launch_sobel_exec(unsigned char *d_in, unsigned char *d_out, int width,
					   int height) {
	dim3 blockSize(16, 16);
	dim3 gridSize((width + 15) / 16, (height + 15) / 16);
	sobel_kernel<<<gridSize, blockSize>>>(d_in, d_out, width, height);
}

void launch_blur_exec(unsigned char *d_in, unsigned char *d_out, int width,
					  int height) {
	dim3 blockSize(16, 16);
	dim3 gridSize((width + 15) / 16, (height + 15) / 16);
	gaussian_blur_kernel<<<gridSize, blockSize>>>(d_in, d_out, width, height);
}

void launch_invert_exec(unsigned char *d_in, unsigned char *d_out, int width,
						int height) {
	int totalPixels = width * height;
	int blockSize = 256;
	int gridSize = (totalPixels + blockSize - 1) / blockSize;
	invert_kernel<<<gridSize, blockSize>>>(d_in, d_out, width, height);
}

void launch_copy_exec(unsigned char *d_in, unsigned char *d_out, int width,
					  int height) {
	int totalPixels = width * height;
	int blockSize = 256;
	int gridSize = (totalPixels + blockSize - 1) / blockSize;
	copy_kernel<<<gridSize, blockSize>>>(d_in, d_out, width, height);
}
