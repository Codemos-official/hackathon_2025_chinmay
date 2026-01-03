#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

void allocate_buffers(unsigned char **d_in_ptr, unsigned char **d_out_ptr,
					  size_t size);
void free_buffers(unsigned char *d_in, unsigned char *d_out);

void upload_to_gpu(unsigned char *d_in, unsigned char *h_in, size_t size);
void download_from_gpu(unsigned char *h_out, unsigned char *d_out, size_t size);

void launch_sobel_exec(unsigned char *d_in, unsigned char *d_out, int width,
					   int height, int threshold);
void launch_blur_exec(unsigned char *d_in, unsigned char *d_out, int width,
					  int height, int threshold, int strength);
void launch_invert_exec(unsigned char *d_in, unsigned char *d_out, int width,
						int height);
void launch_copy_exec(unsigned char *d_in, unsigned char *d_out, int width,
					  int height);

#endif
