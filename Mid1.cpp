#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include "bmpfuncs.h"
#include <iostream>
#include <chrono> 
using namespace std::chrono;


static float theta = 3.14159 / 6;

void rotate(float *input_image, float *output_image,
	int image_width, int image_height,
	float sin_theta, float cos_theta);

void GaussianFilter(float *input_image, float *output_image,
	int image_width, int image_height);

void Sobel(float *input_image, float *output_image,
	int image_width, int image_height);

void Sobel_OpenCL(float *input_image, float *output_image,
	int image_width, int image_height);

int main(int argc, char *argv[]) {
	if (argc < 3) {
		printf("Usage: %s <src file> <dest file>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	float sin_theta = sinf(theta);
	float cos_theta = cosf(theta);

	int image_width, image_height;
	float *input_image = readImage(argv[1], &image_width, &image_height);
	float *output_image = (float*)malloc(sizeof(float) * image_width * image_height);
	 
	//Sequential
	Sobel(input_image, output_image, image_width, image_height);
	//OpenCL
	Sobel_OpenCL(input_image, output_image, image_width, image_height);

	storeImage(output_image, argv[2], image_height, image_width, argv[1]);
	return 0;
}

#define  CHECK_ERROR(err) \
	if (err != CL_SUCCESS) { \
	printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
	exit(EXIT_FAILURE); \
	}

char *get_source_code(const char *file_name, size_t *len) {
	char *source_code;
	size_t length;
	FILE *file;
	errno_t err = fopen_s(&file, file_name, "rb");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);

	source_code = (char *)malloc(length + 1);
	fread(source_code, length, 1, file);
	source_code[length] = '\0';

	fclose(file);
	*len = length;
	return source_code;
}

void GaussianFilter(float *input_image, float *output_image,
	int image_width, int image_height)
{
	float kernelWeights[9] = { 1.0f, 2.0f, 1.0f,
							  2.0f, 4.0f, 2.0f,
							  1.0f, 2.0f, 1.0f };

	for (int y = 0; y < image_height; ++y) {
		for (int x = 0; x < image_width; ++x) {
			if (x >= 1 && x < image_width - 1 && y >= 1 && y < image_height - 1) {
				int k = 0;
				float outColor = 0.0f;
				for (int j = -1; j < 2; ++j)
					for (int i = -1; i < 2; ++i)
						outColor += input_image[(y + j) * image_width + (x + i)] * kernelWeights[k++];
				output_image[y * image_width + x] = outColor / 16.0f;
			}
			else {
				output_image[y * image_width + x] = 0.0f;
			}
		}
	}
}

// Sobel Sequential
void Sobel(float *input_image, float *output_image,
	int image_width, int image_height)
{
	auto start = high_resolution_clock::now();
	float m_y[9] = { -1.0f, -2.0f, -1.0f,
					 0.0f, 0.0f, 0.0f,
					 1.0f, 2.0f, 1.0f };

	float m_x[9] = { -1.0f, 0.0f, 1.0f,
					 -2.0f, 0.0f, 2.0f,
					 -1.0f, 0.0f, 1.0f };

	for (int y = 0; y < image_height; ++y) {
		for (int x = 0; x < image_width; ++x) 
		{
			if (x >= 1 && x < image_width - 1 && y >= 1 && y < image_height - 1) 
			{
				int k = 0;
				float output_dy = 0.0f;	float output_dx = 0.0f;
				for (int j = -1; j < 2; ++j) 
				{
					for (int i = -1; i < 2; ++i) 
					{
						output_dy += input_image[(y + j) * image_width + (x + i)] * m_y[k];
						output_dx += input_image[(y + j) * image_width + (x + i)] * m_x[k];
						k++;
					}
				}
				
				float magnitude = abs(sqrtf(pow(output_dy, 2) + pow(output_dx, 2)));
				

				if (magnitude > 255) 
				{
					magnitude = 255;
				}
				else if (magnitude < 0) 
				{
					magnitude = 0;
				}

				output_image[y * image_width + x] = magnitude;
			} 
			else 
			{	// 가장자리 예외처리
				output_image[y * image_width + x] = 0.0f;
			}
		}
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Sequential Time: " << duration.count() << " ms" << std::endl;

}

void Sobel_OpenCL(float *input_image, float *output_image,
	int image_width, int image_height)
{
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	char *kernel_source;
	size_t kernel_source_size;
	cl_kernel kernel;
	cl_int err;

	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	queue = clCreateCommandQueue(context, device, 0, &err);
	CHECK_ERROR(err);

	kernel_source = get_source_code("mid1.cl", &kernel_source_size);
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char *log;

		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		CHECK_ERROR(err);
		log = (char*)malloc(log_size + 1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		printf("Compiler error : \n%s\n", log);
		free(log);
		exit(0);
	}
	CHECK_ERROR(err);

	kernel = clCreateKernel(program, "img_sobel", &err);
	CHECK_ERROR(err);

	size_t image_size = sizeof(float) * image_width * image_height;
	cl_mem src, dest;
	src = clCreateBuffer(context, CL_MEM_READ_ONLY, image_size, NULL, &err);
	CHECK_ERROR(err);
	dest = clCreateBuffer(context, CL_MEM_READ_WRITE, image_size, NULL, &err);
	CHECK_ERROR(err);

	err = clEnqueueWriteBuffer(queue, src, CL_FALSE, 0, image_size,
		input_image, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dest);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &src);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(cl_int), &image_width);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 3, sizeof(cl_int), &image_height);
	CHECK_ERROR(err);

	size_t global_size[2] = { image_width, image_height };
	size_t local_size[2] = { 16, 16 };
	global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0]
		* local_size[0];
	global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1]
		* local_size[1];

	auto t1 = std::chrono::high_resolution_clock::now();
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size,
		local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueReadBuffer(queue, dest, CL_TRUE, 0, image_size
		, output_image, 0, NULL, NULL);
	CHECK_ERROR(err);
	auto t2 = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(t2 - t1);
	std::cout << "OpenCL eplased Time: " << duration.count() << " ms" << std::endl;

	clReleaseMemObject(src);
	clReleaseMemObject(dest);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

}
