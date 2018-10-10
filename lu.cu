#include <stdio.h>
#include <sys/time.h>
#include <iostream>
using namespace std;

/**
 * O agumento deve ser double
 */
#define GET_TIME(now) { \
	struct timespec time; \
	clock_gettime(CLOCK_MONOTONIC_RAW, &time); \
	now = time.tv_sec + time.tv_nsec/1000000000.0; \
}

/**
 * Para checar erros em chamadas Cuda
 */
#define CUDA_SAFE_CALL(call) { \
	cudaError_t err = call;     \
	if(err != cudaSuccess) {    \
		fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n",__FILE__, __LINE__,cudaGetErrorString(err)); \
		exit(EXIT_FAILURE); } \
}

void luSeq (double *A, int n) {
	for (int i=0; i<n-1; i++){
		for (int j=1; j<n; j++){
			A[j*n+i] = A[j*n+i]/A[i*n+i];
			for (int k=1; k<n; k++){
				A[j*n+k] = A[j*n+k] - A[j*n+i]*A[i*n+k];
			}
		}
	}
}

__global__ void luCalcCol(double *A , int dim, int i) {
	__shared__  double  Aii;
	if (threadIdx.x == 0) {
		Aii = A[i*(dim +1)];
	}
	__syncthreads ();
	int j = blockIdx.x * blockDim.x + threadIdx.x + i + 1;
	if ( j < dim ) {
		A[ j*dim+i ] /= Aii;
	}
}

__global__ void luCalcSub(double *A, int dim , int i) {
	__shared__  double  a_ji[32];
	__shared__  double  a_ik[32];
	int j = blockDim.x * blockIdx.x + threadIdx.x + i + 1;
	int k = blockDim.y * blockIdx.y + threadIdx.y + i + 1;
	if (( threadIdx.y == 0) && (j < dim)) {
		a_ji[threadIdx.x] = A[ j*dim + i ];
	}
	if (( threadIdx.x == 0) && (k < dim)) {
		a_ik[threadIdx.y] = A[ i*dim + k ];
	}
	__syncthreads ();
	if ((j < dim) && (k < dim)) {
		A[ j*dim + k ] -= a_ji[threadIdx.x] * a_ik[threadIdx.y];
	}
}

void  luGPU(double *A, int n, int blockSize) {
	int i, n_blocos;
	for (i = 0; i < n-1; i++) {
		n_blocos = ((n-i-1) + blockSize -1) / blockSize;
		dim3  g_blocos(n_blocos, n_blocos);
		dim3  n_threads(blockSize,blockSize);
		luCalcCol <<< n_blocos, blockSize >>>(A, n, i);
		CUDA_SAFE_CALL(cudaGetLastError());
		luCalcSub <<< g_blocos, n_threads >>>(A, n, i);
		CUDA_SAFE_CALL(cudaGetLastError());
	}
}

void fillMatrix(double* A, int n){
	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			A[i*n+j] = (i+1)*(j+1);
		}
	}
}

void checkResults(double *mat1, double *mat2, int n){
	for (int i=0; i<n; i++) {
		for (int j=0; j<n; j++) {
			if (fabs(mat1[i*n+j] - mat2[i*n+j]) > 1e-5) {
				cerr << "Resultado incorreto em " << i << " x " << j << " -> " << mat1[i*n+j] << " " <<  mat2[i*n+j] << endl;
				exit(EXIT_FAILURE);
			}
		}
	}
}

void printResults(int n, double timeSeq, double timeCpuGpu, double timeRunPar, double timeGpuCpu){
	cout << n << ";" << timeSeq << ";" << timeCpuGpu << ";" << timeRunPar << ";" << timeGpuCpu << endl;
}

int  main(int argc, char** argv) {
	int n, blockSize;
	double *Aseq, *Apar, *Adevice;
	double begin, end, timeSeq, timeCpuGpu, timeRunPar, timeGpuCpu;
	
	if(argc < 3) {
		cerr << "Digite: "<< argv[0] <<" <Dimensão da matriz> <Dimensão do bloco>" << endl;
		exit(EXIT_FAILURE);
	}
	n = atol(argv[1]);
	blockSize = atol(argv[2]);
	
	size_t  quant_mem = n*n*sizeof(double);
	Aseq = (double *) malloc(quant_mem);
	if ( Aseq == NULL   ) {
		cerr << "Memoria  insuficiente" << endl;
		exit(EXIT_FAILURE);
	}
	Apar = (double *) malloc(quant_mem);
	if ( Apar == NULL   ) {
		cerr << "Memoria  insuficiente" << endl;
		exit(EXIT_FAILURE);
	}
	fillMatrix(Aseq, n);
	
	GET_TIME(begin);
	CUDA_SAFE_CALL(cudaMalloc((void**) &Adevice, quant_mem));
	CUDA_SAFE_CALL(cudaMemcpy(Aseq, Adevice, quant_mem, cudaMemcpyDeviceToHost));
	GET_TIME(end);
	timeCpuGpu = end-begin;
	
	GET_TIME(begin);
	luGPU(Adevice, n, blockSize);
	GET_TIME(end);
	timeRunPar = end-begin;
	
	GET_TIME(begin);
	CUDA_SAFE_CALL(cudaMemcpy(Apar, Adevice, quant_mem, cudaMemcpyDeviceToHost));
	GET_TIME(end);
	timeGpuCpu = end-begin;
	
	GET_TIME(begin);
	luSeq(Aseq, n);
	GET_TIME(end);
	timeSeq = end-begin;
	
	CUDA_SAFE_CALL(cudaFree(Adevice));
	free(Aseq);
	free(Apar);
	
	checkResults(Aseq, Apar, n);
	printResults(n, timeSeq, timeCpuGpu, timeRunPar, timeGpuCpu);
	
	CUDA_SAFE_CALL(cudaDeviceReset());
	exit(EXIT_SUCCESS);
}
