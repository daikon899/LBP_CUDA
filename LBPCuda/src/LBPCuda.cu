#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>
#include <cuda.h>
#include <random>
#include <unistd.h>
#include <chrono>
#include "opencv2/opencv_modules.hpp"


using namespace std

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

__device__ void lbpApply(Mat imgIn_d, Mat imgOut_d, Mat histogram_d){

	int i = blockIdx.y * blockDim.y + threadIdx.y; //rows
	int j = blockIdx.x * blockDim.x + threadIdx.x; //cols

	if (i > 0 && i < imgIn_d.rows - 1 && j > 0 && j < imgIn_d.cols - 1)

	int neighbors[8];
	neighbors[0] = imgIn.at<uchar>(i - 1, j - 1);
	neighbors[1] = imgIn.at<uchar>(i - 1, j);
	neighbors[2] = imgIn.at<uchar>(i - 1, j + 1);
	neighbors[3] = imgIn.at<uchar>(i, j + 1);
	neighbors[4] = imgIn.at<uchar>(i + 1, j + 1);
	neighbors[5] = imgIn.at<uchar>(i + 1, j);
	neighbors[6] = imgIn.at<uchar>(i + 1, j - 1);
	neighbors[7] = imgIn.at<uchar>(i, j - 1);

	int oldVal = imgIn.at<uchar>(i, j);

	int newVal = 0;
	for (int k = 0; k < 8; k++) {
		if (neighbors[k] >= oldVal)
			newVal += weights[k];
	}
	imgOut.at<uchar>(i - 1, j - 1) = newVal;
	histogram[newVal]++;
}



__host__ Mat localBinaryPattern(Mat &imgIn_h) {
    const int weights[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    int *histogram = new int[256] {0};
    Mat *imgIn_h, *imgIn_d;
    int *histogram_h, histogram_d;
    Mat imgOut_h = Mat::zeros(imgIn_h.rows, imgIn_h.cols, CV_8UC1);

    copyMakeBorder(imgIn, imgIn, 1, 1, 1, 1, BORDER_CONSTANT, 0);


    //TODO check if correct
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&imgIn_d, sizeof(Mat)));
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&imgOut_d, sizeof(Mat)));
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&histogram_d, sizeof(Mat)));
	CUDA_CHECK_RETURN(cudaMemSet(histogram, 0, sizeof(int) * 256 ));
	CUDA_CHECK_RETURN(cudaMemcpy(imgIn_d, imgIn_d, sizeof(Mat), cudaMemcpyHostToDevice));

	dim3 gridDim(ceil(imgIn.rows), ceil(imgIn.cols));
	dim3 blockDim(16,16)
	lbpApply<<<gridDim, blockDim>>>(imgIn_d, imgOut_d, histogram_d);
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpy(imgOut_h, imgOut_d, sizeof(Mat), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(histogram_h, histogram_d, sizeof(int) * 256, cudaMemcpyDeviceToHost));

    writeCsv(histogram);

    return imgOut;
}

int main(int argc, char **argv){
	String imgName = argv[1];
	Mat inputImg = cv::imread("../input/" + imgName, 0);
	//imshow("Image before LBP", inputImg);
	warm_up_gpu<<<128, 128>>>();  // avoids cold start for testing purposes
	auto start = chrono::high_resolution_clock::now();
	Mat outputImg = localBinaryPattern(inputImg);
	auto end = chrono::high_resolution_clock::now();
	auto ms_int = duration_cast<chrono::milliseconds>(end - start);

	//imshow("Image after LBP", outputImg);
	//waitKey(0);

	return ms_int.count();

}
