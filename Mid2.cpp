#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <CL/cl.h>

#include <chrono> 
using namespace std::chrono;


#define  CHECK_ERROR(err) \
	if (err != CL_SUCCESS) { \
	printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
	exit(EXIT_FAILURE); \
	}

double* Matmul_Opencl(double *input, double *w1, double *w2);

using namespace std;

// Testing image file name
const string testing_image_fn = "mnist/t10k-images.idx3-ubyte";

// Testing label file name
const string testing_label_fn = "mnist/t10k-labels.idx1-ubyte";

// Weights file name
const string model_fn = "model-neural-network.dat";

// Number of testing samples
const int nTesting = 10000;

// Image size in MNIST database
const int width = 28;
const int height = 28;

// n1 = Number of input neurons
// n2 = Number of hidden neurons
// n3 = Number of output neurons

const int n1 = width * height; // = 784, without bias neuron 
const int n2 = 128;
const int n3 = 10; // Ten classes: 0 - 9

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double *w1, *out1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double *w2, *in2, *out2;

// Layer 3 - Output layer
double *in3, *out3;
double expected[n3];

// Image. In MNIST: 28x28 gray scale images.
int d[width][height];

// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
char *kernel_source;
size_t kernel_source_size;
cl_kernel kernel;
cl_int err;

// Declare Batch Size
const int BATCH_SIZE = 10000;

// +--------------------+
// | About the software |
// +--------------------+

void about() {
	// Details
	cout << "*************************************************" << endl;
	cout << "*** Testing Neural Network for MNIST database ***" << endl;
	cout << "*************************************************" << endl;
	cout << endl;
	cout << "No. input neurons: " << n1 << endl;
	cout << "No. hidden neurons: " << n2 << endl;
	cout << "No. output neurons: " << n3 << endl;
	cout << endl;
	cout << "No. testing sample: " << nTesting << endl << endl;
}

// +-----------------------------------+
// | Memory allocation for the network |
// +-----------------------------------+

void init_array() {
	// Layer 1 - Layer 2 = Input layer - Hidden layer
	w1 = new double[n1 * n2];

	out1 = new double[n1];

	// Layer 2 - Layer 3 = Hidden layer - Output layer
	w2 = new double[n2 * n3];

	in2 = new double[n2];
	out2 = new double[n2];

	// Layer 3 - Output layer
	in3 = new double[n3];
	out3 = new double[n3];
}

// +----------------------------------------+
// | Load model of a trained Neural Network |
// +----------------------------------------+

void load_model(string file_name) {
	ifstream file(file_name.c_str(), ios::in);

	// Input layer - Hidden layer
	for (int i = 0; i < n1; ++i) {
		for (int j = 0; j < n2; ++j) {
			file >> w1[i * n2 + j];
		}
	}

	// Hidden layer - Output layer
	for (int i = 0; i < n2; ++i) {
		for (int j = 0; j < n3; ++j) {
			file >> w2[i * n3 + j];
		}
	}

	file.close();
}

// +------------------+
// | Sigmoid function |
// +------------------+

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

// +------------------+
// |   Mat multiply   |
// +------------------+

void mat_mul(double *A, double *B, double *C,
	int ROW_A, int COL_A, int COL_B) {
	int i, j, k;
	for (i = 0; i < ROW_A; ++i) {
		for (j = 0; j < COL_B; ++j) {
			C[i * COL_B + j] = 0.0;
			for (k = 0; k < COL_A; ++k) {
				C[i * COL_B + j] += A[i * COL_A + k] * B[k * COL_B + j];
			}
		}
	}

}

int input() {
	char number;

	// Reading image
	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			image.read(&number, sizeof(char));
			if (number == 0) {
				d[i][j] = 0;
			}
			else {
				d[i][j] = 1;
			}
		}
	}

	// 28x28을 1x784로 1차원 변경함.
	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			int pos = i + j * width;
			out1[pos] = d[i][j];
		}
	}

	// Reading label
	label.read(&number, sizeof(char));
	for (int i = 0; i < n3; ++i) {
		expected[i] = 0.0;
	}
	expected[number] = 1.0;

	return (int)(number);
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


void kernelInit()
{
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	queue = clCreateCommandQueue(context, device, 0, &err);
	CHECK_ERROR(err);

	kernel_source = get_source_code("mid2.cl", &kernel_source_size);
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
}

int main()
{
	about();
	kernelInit();

	image.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
	label.open(testing_label_fn.c_str(), ios::in | ios::binary); // Binary label file
	
																 // Reading file headers
	char number;
	for (int i = 1; i <= 16; ++i) {
		image.read(&number, sizeof(char));
	}
	for (int i = 1; i <= 8; ++i) {
		label.read(&number, sizeof(char));
	}

	// !!!! Neural Network Initialization !!!!
	init_array(); // Memory allocation
	load_model(model_fn); // Load model (weight matrices) of a trained Neural Network

	int nCorrect = 0;
	
	

	int labels[nTesting];

	double *out_batch = new double[n1 * nTesting];

	for (int sample = 0; sample < nTesting; ++sample)
	{
		int size = n1 * sample;
		labels[sample] = input();		// label = 정답값 | 정답값 읽어오기.		
		// 784 x batch_size로 memcopy해주기.
		memcpy(out_batch + size, out1, sizeof(double) * n1);
	}

	auto time_start = high_resolution_clock::now();	
	double *out = Matmul_Opencl(out_batch, w1, w2);
	auto time_end = high_resolution_clock::now();

	for (int sample = 0; sample < nTesting; ++sample)
	{
		int size = sample * 10;

		double out3[10];
		memcpy(out3, out + size, sizeof(double) * 10);

		// Prediction
		int predict = 0;
		for (int i = 1; i < n3; ++i) {
			if (out3[i] > out3[predict]) {
				predict = i;
			}
		}

		if (labels[sample] == predict) {	// 정답값과 예측값 비교하기.
			++nCorrect;
		}
	}	

	
	
	auto duration = duration_cast<milliseconds>(time_end - time_start);
	std::cout << "Opencl Time: " << duration.count() << " ms" << std::endl;

	double accuracy = (double)(nCorrect) / nTesting * 100.0;
	cout << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
	printf("Accuracy: %0.2lf\n", accuracy);

	report << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
	report << "Accuracy: " << accuracy << endl;

	delete[] in2;
	delete[] in3;
	delete[] w1;
	delete[] w2;
	delete[] out1;
	delete[] out2;
	delete[] out3;
	delete[] out;
	delete[] out_batch;

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	report.close();
	image.close();
	label.close();

	return 0;
}

double* Matmul_Opencl(double *input, double *w1, double *w2)
{
	kernel = clCreateKernel(program, "mat_mul", &err);
	CHECK_ERROR(err);

	size_t _in_size = sizeof(double) * BATCH_SIZE * n1;
	size_t _w1_size = sizeof(double) * n1 * n2;
	size_t _w2_size = sizeof(double) * n2 * n3;

	size_t _h1_size = sizeof(double) * BATCH_SIZE * n2;

	size_t _out_size = sizeof(double) * BATCH_SIZE * n3;
	
	cl_mem _in, _w1, _w2, _h1, _out;
	_in = clCreateBuffer(context, CL_MEM_READ_ONLY, _in_size, NULL, &err);			CHECK_ERROR(err);
	_w1 = clCreateBuffer(context, CL_MEM_READ_ONLY, _w1_size, NULL, &err);  		CHECK_ERROR(err);
	_w2 = clCreateBuffer(context, CL_MEM_READ_ONLY, _w2_size, NULL, &err);			CHECK_ERROR(err);
	_h1 = clCreateBuffer(context, CL_MEM_READ_WRITE, _h1_size, NULL, &err);  		CHECK_ERROR(err);
	_out = clCreateBuffer(context, CL_MEM_READ_WRITE, _out_size, NULL, &err);		CHECK_ERROR(err);

	err = clEnqueueWriteBuffer(queue, _in, CL_FALSE, 0, _in_size, input, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clEnqueueWriteBuffer(queue, _w1, CL_FALSE, 0, _w1_size, w1, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clEnqueueWriteBuffer(queue, _w2, CL_FALSE, 0, _w2_size, w2, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &_in);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &_w1);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &_h1);
	CHECK_ERROR(err);	
	err = clSetKernelArg(kernel, 3, sizeof(cl_int), &n1);
	CHECK_ERROR(err);	

	size_t global_size[2] = { n2 , BATCH_SIZE };	// 128 x 10000
	size_t local_size[2] = { 32, 10 };		// 
	global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0]
		* local_size[0];
	global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1]
		* local_size[1];

	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size,
		local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
	
	err = clFinish(queue);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &_h1);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &_w2);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &_out);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 3, sizeof(cl_int), &n2);
	CHECK_ERROR(err);

	size_t global_size2[2] = { n3 , BATCH_SIZE };	// 10 x 10000
	size_t local_size2[2] = { 5, 10 };		// 
	global_size2[0] = (global_size2[0] + local_size2[0] - 1) / local_size2[0]
		* local_size2[0];
	global_size2[1] = (global_size2[1] + local_size2[1] - 1) / local_size2[1]
		* local_size2[1];

	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size2,
		local_size2, 0, NULL, NULL);
	CHECK_ERROR(err);

	double *out = new double[n3 * nTesting];

	err = clEnqueueReadBuffer(queue, _out, CL_TRUE, 0, _out_size, out, 0, NULL, NULL);
	CHECK_ERROR(err);

	clReleaseMemObject(_in);
	clReleaseMemObject(_w1);
	clReleaseMemObject(_w2);
	clReleaseMemObject(_h1);
	clReleaseMemObject(_out);

	return out;
}					   
