#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
using namespace std;

/**
 * O argumento deve ser double
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

void setupMatrix(double *A, int n){
	for (int i=0; i<n; i++){
		A[i] = 20;
		A[n*n-i-1]=30;
	}
	double t = 20, a = 10.0/(n-1);
	for (int i=1; i<n-1; i++){
		t+=a;
		A[i*n] = t;
		for (int j=1; j<n-1; j++){
			A[i*n+j]=20;
		}
		A[i*n + n - 1] = t;
	}
}

void print(double *A, int n){	
	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			cout << A[i*n+j] << "  ";
		}
		cout << endl;
	}
}

int  main(int argc, char** argv) {
	int n=0, blockSize;
	double *Aseq, *Adevice;
	double begin, end, timeSeq, timeCpuGpu, timeRunPar, timeGpuCpu;	
	if(argc < 3) {
		cerr << "Digite: "<< argv[0] <<" <Dimensão da matriz> <Dimensão do bloco>" << endl;
		exit(EXIT_FAILURE);
	}
	n = atol(argv[1]);
	blockSize = atol(argv[2]);
	
	size_t matBytes = n*n*sizeof(double);
	Aseq = (double *) malloc(matBytes);
	if ( Aseq == NULL   ) {
		cerr << "Memoria  insuficiente" << endl;
		exit(EXIT_FAILURE);
	}
	setupMatrix(Aseq, n);
	print(Aseq, n);
// 	GET_TIME(begin);
// 	CUDA_SAFE_CALL(cudaMalloc((void**) &Adevice, matBytes));
// 	CUDA_SAFE_CALL(cudaMemcpy(Aseq, Adevice, matBytes, cudaMemcpyDeviceToHost));
// 	GET_TIME(end);
// 	timeCpuGpu = end-begin;
	
// 	GET_TIME(begin);
// 	luGPU(Adevice, n, blockSize);
// 	GET_TIME(end);
// 	timeRunPar = end-begin;
	
// 	GET_TIME(begin);
// 	CUDA_SAFE_CALL(cudaMemcpy(Apar, Adevice, matBytes, cudaMemcpyDeviceToHost));
// 	GET_TIME(end);
// 	timeGpuCpu = end-begin;
	
// 	GET_TIME(begin);
// 	luSeq(Aseq, n);
// 	GET_TIME(end);
// 	timeSeq = end-begin;
	
// 	CUDA_SAFE_CALL(cudaFree(Adevice));
	free(Aseq);
	
// 	checkResults(Aseq, Apar, n);
	
// 	printResults(n, timeSeq, timeCpuGpu, timeRunPar, timeGpuCpu);
	
	CUDA_SAFE_CALL(cudaDeviceReset());
	exit(EXIT_SUCCESS);
}
