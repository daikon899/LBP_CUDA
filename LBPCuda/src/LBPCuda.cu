#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>
#include <cuda.h>
#include <random>

#include <unistd.h>
#include <chrono>
using namespace std::chrono;

using namespace std;

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

int main(int argc, char **argv){

	warm_up_gpu<<<128, 128>>>();  // avoids cold start for testing purposes
	auto start = high_resolution_clock::now();
	//kMeansCuda(data_h, MAX_ITERATIONS, 1);
	auto end = high_resolution_clock::now();

	auto ms_int = duration_cast<milliseconds>(end - start);
	cout << "duration in milliseconds: " << ms_int.count() <<"\n";


    return ms_int.count();

}
