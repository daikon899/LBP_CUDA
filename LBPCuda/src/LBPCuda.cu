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

	int i = blockIdx.y * blockDim.y + threadIdx.y; //rows
	int j = blockIdx.x * blockDim.x + threadIdx.x; //cols
/*
	//create weights
	__shared__ const int weights[8];
	if(i < 8 && j == 1) {
		weights[i] = 2^i;
	}
	__synchthreads();
*/

	if (i > 0 && i < rows && j > 0 && j < cols){
		imgOut_d[i * cols + j] = imgIn_d[(i + 1) * cols + j + 1 ];
		/*
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
		*/
	}

}



__host__ Mat localBinaryPattern(const Mat &imgIn_h) {


    // output image
    unsigned char *imgOut_d;
    size_t imgOutSize = imgIn_h.step * imgIn_h.rows;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&imgOut_d, imgOutSize));
    Mat imgOut_h = Mat::zeros(imgIn_h.rows, imgIn_h.cols, CV_8UC1);


    //input image
    unsigned char *imgIn_d;
    // copyMakeBorder(imgIn_h, imgIn_h, 1, 1, 1, 1, BORDER_CONSTANT, 0); //FIXME error if uncomment this
    size_t imgInSize = imgIn_h.step * imgIn_h.rows;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&imgIn_d, imgInSize));
    CUDA_CHECK_RETURN(cudaMemcpy(imgIn_d, imgIn_h.data, imgInSize, cudaMemcpyHostToDevice));

    //histogram
    int *histogram_h, *histogram_d;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&histogram_d, sizeof(int) * 256));
	CUDA_CHECK_RETURN(cudaMemset(histogram_d, 0, sizeof(int) * 256 ));


	dim3 gridDim(ceil(imgIn_h.rows), ceil(imgIn_h.cols));
	dim3 blockDim(16,16);
	lbpApply<<<gridDim, blockDim>>>(imgIn_d, imgOut_d, histogram_d, imgOut_h.rows, imgOut_h.cols);
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpy(imgOut_h.data, imgOut_d, imgOutSize, cudaMemcpyDeviceToHost));
	//CUDA_CHECK_RETURN(cudaMemcpy(histogram_h, (void **)&histogram_d, sizeof(int) * 256, cudaMemcpyDeviceToHost));

    //writeCsv(histogram_h);

    return imgOut_h;
}


int main(int argc, char **argv){
	String imgName = argv[1];
	Mat inputImg = cv::imread("input/" + imgName, 0);

	//imshow("Image before LBP", inputImg);
	//warm_up_gpu<<<128, 128>>>();  // avoids cold start for testing purposes
	auto start = chrono::high_resolution_clock::now();

    copyMakeBorder(inputImg, inputImg, 1, 1, 1, 1, BORDER_CONSTANT, 0); // TODO add this to localbinarypattern function

	Mat outputImg = localBinaryPattern(inputImg);
	auto end = chrono::high_resolution_clock::now();
	auto ms_int = duration_cast<chrono::milliseconds>(end - start);

	imshow("Image after LBP", outputImg);
	waitKey(0);

	return ms_int.count();

}
