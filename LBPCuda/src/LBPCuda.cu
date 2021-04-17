#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>
#include <cuda.h>
#include <random>
#include <unistd.h>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "writeCsv.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

__constant__ int weights[8];

 static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

//---------------------------------------------------------------------------------------------------------------------------------------

__global__ void warm_up_gpu(){  // this kernel avoids cold start when evaluating duration of kmeans exec.
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

//----------------------------------------------------------------------------------------------------------------------------------------

__global__ void lbpApply(unsigned char *imgIn_d, unsigned char *imgOut_d, int *histogram_d, int rows, int cols){

	int i = blockIdx.y * blockDim.y + threadIdx.y; //row
	int j = blockIdx.x * blockDim.x + threadIdx.x; //col

	//+++++++++++++++++++++++++++++++++++++++++++missing histogram and tiling++++++++++++++++++++++++++++++++++++++++++++++++++


	if (i < rows && j < cols){
		int neighbors[8];

		// remember that imgOut_d[i * cols + j] -> imgIn_d[ (i + 1) * (cols + 2) + j + 1 ];

		neighbors[0] = imgIn_d[(i) * (cols + 2) + j]; // (i - 1, j - 1);
		neighbors[1] = imgIn_d[(i) * (cols + 2) + j + 1]; // (i - 1, j);
		neighbors[2] = imgIn_d[(i) * (cols + 2) + j + 2]; // (i - 1, j + 1);
		neighbors[3] = imgIn_d[(i + 1) * (cols + 2) + j + 2]; // (i, j + 1);
		neighbors[4] = imgIn_d[(i + 2) * (cols + 2) + j + 2]; // (i + 1, j + 1);
		neighbors[5] = imgIn_d[(i + 2) * (cols + 2) + j + 1]; // (i + 1, j);
		neighbors[6] = imgIn_d[(i + 2) * (cols + 2) + j]; // (i + 1, j - 1);
		neighbors[7] = imgIn_d[(i + 1) * (cols + 2) + j]; // (i, j - 1);

		int oldVal = imgIn_d[ (i + 1) * (cols + 2) + j + 1 ]; // (i, j);

		int newVal = 0;
		for (int k = 0; k < 8; k++) {
			if (neighbors[k] >= oldVal)
				newVal += weights[k];
		}
		imgOut_d[i * cols + j] = newVal;

		//histogram[newVal]++;

	}

}


// TODO insert main code in this function:
//__host__ Mat localBinaryPattern(const Mat &imgIn_h) {}


int main(int argc, char **argv){
	String imgName = argv[1];
	Mat imgIn_h = cv::imread("input/" + imgName, 0);

	//imshow("Image before LBP", imgIn_h);
	//warm_up_gpu<<<128, 128>>>();  // avoids cold start for testing purposes
	auto start = chrono::high_resolution_clock::now();

	// output image
	unsigned char *imgOut_d;
	size_t imgOutSize = imgIn_h.step * imgIn_h.rows;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&imgOut_d, imgOutSize));
	Mat imgOut_h = Mat::zeros(imgIn_h.rows, imgIn_h.cols, CV_8UC1);


	//input image
	unsigned char *imgIn_d;
	copyMakeBorder(imgIn_h, imgIn_h, 1, 1, 1, 1, BORDER_CONSTANT, 0);
	size_t imgInSize = imgIn_h.step * imgIn_h.rows;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&imgIn_d, imgInSize));
	CUDA_CHECK_RETURN(cudaMemcpy(imgIn_d, imgIn_h.data, imgInSize, cudaMemcpyHostToDevice));

	//histogram
	int *histogram_h, *histogram_d;
	histogram_h = (int *) malloc(sizeof(int) * 256);
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&histogram_d, sizeof(int) * 256));
	CUDA_CHECK_RETURN(cudaMemset(histogram_d, 0, sizeof(int) * 256 ));

	//weights
	int weights_h[8] = {1, 2, 4, 8, 16, 32, 64, 128};
	cudaMemcpyToSymbol(weights, &weights_h, sizeof(int) * 8);


	dim3 blockDim(16,16);
	dim3 gridDim(ceil( (float) imgOut_h.cols / blockDim.x), ceil( (float) imgOut_h.rows / blockDim.y) );
	lbpApply<<<gridDim, blockDim>>>(imgIn_d, imgOut_d, histogram_d, imgOut_h.rows, imgOut_h.cols);
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpy(imgOut_h.data, imgOut_d, imgOutSize, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(histogram_h, histogram_d, sizeof(int) * 256, cudaMemcpyDeviceToHost));

	//writeCsv(histogram_h);


	auto end = chrono::high_resolution_clock::now();
	auto ms_int = duration_cast<chrono::milliseconds>(end - start);

	imshow("Image after LBP", imgOut_h);
	waitKey(0);

	return ms_int.count();

}
