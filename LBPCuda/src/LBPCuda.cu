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

__constant__ int weights[3][3];
#define BLOCK_WIDTH 16

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
__global__ void lbpApplyS(unsigned char *imgIn_d, unsigned char *imgOut_d, int *histogram_d, int rows, int cols){
	int i = blockIdx.y * blockDim.y + threadIdx.y; //row of imgOut
	int j = blockIdx.x * blockDim.x + threadIdx.x; //col of imgOut
	int bi = threadIdx.y;
	int bj = threadIdx.x;
	int colsB = cols + BLOCK_WIDTH; //columns number considering border


	__shared__ unsigned char imgIn_s[BLOCK_WIDTH + 2][BLOCK_WIDTH + 2];
	__shared__ int histogram_s[256];

	int tid = bi * BLOCK_WIDTH + bj;
	if(tid < 256)
		histogram_s[tid] = 0; // NOTE: if BLOCK_WIDTH < 16 does not work!

	//load one part of image in shared memory
	imgIn_s[bi][bj] = imgIn_d[ i * (colsB) + j];
	if (bj < 2)
		imgIn_s[bi][bj + BLOCK_WIDTH] = imgIn_d[ i * (colsB) + j + BLOCK_WIDTH];
	if (bi < 2)
		imgIn_s[bi + BLOCK_WIDTH][bj] = imgIn_d[ (i + BLOCK_WIDTH) * (colsB) + j];
	if (bi >= BLOCK_WIDTH - 2 && bj >= BLOCK_WIDTH - 2)
		imgIn_s[bi + 2][bj + 2] = imgIn_d[(i + 2) * (colsB) + j + 2];

	__syncthreads();


	if (i < rows && j < cols){
		int oldVal = imgIn_s[bi + 1][bj + 1]; // fuori dall' if?
		int newVal = 0;
		for (int u = 0; u < 3; u++)
			for (int v = 0; v < 3; v++)
				if (imgIn_s[bi + u][bj + v] >= oldVal)
					newVal += weights[u][v];

		imgOut_d[i * cols + j] = newVal;
		atomicAdd(&histogram_s[newVal], 1);
	}
	__syncthreads();

	//commit histogram to global memory
	atomicAdd(&histogram_d[bi * 16 + bj], histogram_s[bi * 16 + bj]);
}


//same function with no use of shared memory
__global__ void lbpApply(unsigned char *imgIn_d, unsigned char *imgOut_d, int *histogram_d, int rows, int cols){

	int i = blockIdx.y * blockDim.y + threadIdx.y; //row of imgOut
	int j = blockIdx.x * blockDim.x + threadIdx.x; //col of imgOut
	int colsB = cols + BLOCK_WIDTH; //columns number considering border

	if (i < rows && j < cols){
		int neighbors[3][3];

		// remember that imgOut_d[i * cols + j] -> imgIn_d[ (i + 1) * (cols + 2) + j + 1 ];

		neighbors[0][0] = imgIn_d[(i) * (colsB) + j]; // (i - 1, j - 1);
		neighbors[0][1] = imgIn_d[(i) * (colsB) + j + 1]; // (i - 1, j);
		neighbors[0][2] = imgIn_d[(i) * (colsB) + j + 2]; // (i - 1, j + 1);
		neighbors[1][0] = imgIn_d[(i + 1) * (colsB) + j]; // (i, j - 1);
		neighbors[1][1] = 0;
		neighbors[1][2] = imgIn_d[(i + 1) * (colsB) + j + 2]; // (i, j + 1);
		neighbors[2][0] = imgIn_d[(i + 2) * (colsB) + j]; // (i + 1, j - 1);
		neighbors[2][1] = imgIn_d[(i + 2) * (colsB) + j + 1]; // (i + 1, j);
		neighbors[2][2] = imgIn_d[(i + 2) * (colsB) + j + 2]; // (i + 1, j + 1);

		int oldVal = imgIn_d[ (i + 1) * (colsB) + j + 1 ]; // (i, j);

		int newVal = 0;
		for (int u = 0; u < 3; u ++)
			for (int v = 0; v < 3; v++)
				if (neighbors[u][v] >= oldVal)
					newVal += weights[u][v];

		imgOut_d[i * cols + j] = newVal;
		atomicAdd(&histogram_d[newVal], 1);
	}

}



// TODO insert main code in this function:
__host__ Mat localBinaryPattern(Mat &imgIn_h) {
	//output image
		unsigned char *imgOut_d;
		size_t imgOutSize = imgIn_h.step * imgIn_h.rows;
		CUDA_CHECK_RETURN(cudaMalloc((void ** )&imgOut_d, imgOutSize));
		Mat imgOut_h = Mat::zeros(imgIn_h.rows, imgIn_h.cols, CV_8UC1);


		//input image
		unsigned char *imgIn_d;
		copyMakeBorder(imgIn_h, imgIn_h, 1, BLOCK_WIDTH - 1, 1, BLOCK_WIDTH - 1, BORDER_CONSTANT, 0);
		size_t imgInSize = imgIn_h.step * imgIn_h.rows;
		CUDA_CHECK_RETURN(cudaMalloc((void ** )&imgIn_d, imgInSize));
		CUDA_CHECK_RETURN(cudaMemcpy(imgIn_d, imgIn_h.data, imgInSize, cudaMemcpyHostToDevice));

		//histogram
		int *histogram_h, *histogram_d;
		histogram_h = (int *) malloc(sizeof(int) * 256);
		CUDA_CHECK_RETURN(cudaMalloc((void ** )&histogram_d, sizeof(int) * 256));
		CUDA_CHECK_RETURN(cudaMemset(histogram_d, 0, sizeof(int) * 256 ));

		//weights
		int weights_h[3][3] = {1, 2, 4, 128, 0, 8, 64, 32, 16};
		cudaMemcpyToSymbol(weights, &weights_h, sizeof(int) * 9);


		dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
		dim3 gridDim(ceil( (float) imgOut_h.cols / blockDim.x), ceil( (float) imgOut_h.rows / blockDim.y) );
		//lbpApply<<<gridDim, blockDim>>>(imgIn_d, imgOut_d, histogram_d, imgOut_h.rows, imgOut_h.cols);
		lbpApplyS<<<gridDim, blockDim>>>(imgIn_d, imgOut_d, histogram_d, imgOut_h.rows, imgOut_h.cols);
		cudaDeviceSynchronize();

		CUDA_CHECK_RETURN(cudaMemcpy(imgOut_h.data, imgOut_d, imgOutSize, cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(histogram_h, histogram_d, sizeof(int) * 256, cudaMemcpyDeviceToHost));

		writeCsv(histogram_h);

		free(histogram_h);

		cudaFree(imgOut_d);
		cudaFree(imgIn_d);
		cudaFree(histogram_d);

		return imgOut_h;
}


int main(int argc, char **argv){
	//String imgName = argv[1];
	String imgName = "images.jpg";
	Mat imgIn_h = cv::imread("input/" + imgName, 0);


	//imshow("Image before LBP", imgIn_h);
	warm_up_gpu<<<128, 128>>>();  // avoids cold start for testing purposes
	auto start = chrono::high_resolution_clock::now();

	Mat imgOut_h = localBinaryPattern(imgIn_h);

	auto end = chrono::high_resolution_clock::now();
	auto ms_int = duration_cast<chrono::milliseconds>(end - start);

	imshow("Image after LBP", imgOut_h);
	waitKey(0);

	int time = ms_int.count();

	printf("image processed in %d milliseconds \n", time);

	return time;

}
