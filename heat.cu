#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
using namespace std;

#define ALPHA 19e-5
#define DELTA_T 120
#define ROUNDS 3*60*60/DELTA_T
#define DISTANCE 0.1

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

__global__ void updateHeat(double *last, double *next , int n, int deltaT) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int pos = i*n + j;
	if (i==0 || i==n-1 || j==0 || j==n-1){
		next[pos] = last[pos];
	} else if (i < n && j < n){
		next[pos] = last[pos] + 
			(ALPHA*deltaT/(DISTANCE*DISTANCE))*(last[pos-1]+last[pos+1]+last[pos-n]+last[pos+n]-4*last[pos]);	
	}
}

void print(double *A, int n){	
	cout << "#############################################################" << endl;
	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			cout << A[i*n+j] << "  ";
		}
		cout << endl;
	}
	cout << "#############################################################" << endl;
}

void playRounds(double **AdevicePointer, int n, int blockSize, int rounds, int deltaT) {

	double *Atemp, *aux, *Adevice = AdevicePointer[0], *A;
	size_t matBytes = n*n*sizeof(double);
	A = (double *) malloc(matBytes);
	CUDA_SAFE_CALL(cudaMalloc((void**) &Atemp, matBytes));
	
	int nBlocks = (n + blockSize -1) / blockSize;
	dim3  gBlocks(nBlocks, nBlocks);
	dim3 nThreads(blockSize,blockSize);
	
	for(int i=0; i<rounds; i++){
		updateHeat <<< gBlocks, nThreads >>>(Adevice, Atemp, n, deltaT);
		CUDA_SAFE_CALL(cudaGetLastError());
		aux = Adevice;
		Adevice = Atemp;
		Atemp = aux;
		CUDA_SAFE_CALL(cudaMemcpy(A, Adevice, matBytes, cudaMemcpyDeviceToHost));
		print(A, n);
	}
	CUDA_SAFE_CALL(cudaFree(Atemp));
	AdevicePointer[0] =Adevice;
}

void playRoundsSeq(double **APointer, int n, int rounds, int deltaT) {
		cout << "OwO" << endl;
	double *A = APointer[0], *Temp, *aux;
		cout << "OwO" << endl;
	size_t matBytes = n*n*sizeof(double);
		cout << "OwO" << endl;
	Temp = (double *) malloc(matBytes);
		cout << "OwO" << endl;
	for(int r=0; r<rounds; r++){
		cout << "OwO" << endl;
		for (int i = 0; i < n; ++i) {
		cout << "OwO" << endl;
			Temp[i*n] = A[i*n];
		cout << "OwO" << endl;
			for (int j = 1; j < n-1; ++j)	{
		cout << "OwO" << endl;
				int pos = i*n + j;
		cout << "OwO" << endl;
				if (i==0 || i==n-1){
					Temp[pos] = A[pos];
				} else if (i < n && j < n){
					Temp[pos] = A[pos] + 
						(ALPHA*deltaT/(DISTANCE*DISTANCE))*(A[pos-1]+A[pos+1]+A[pos-n]+A[pos+n]-4*A[pos]);	
				}
		cout << "OwO" << endl;
			}
		cout << "OwO" << endl;
			Temp[(i+1)*n-1] = A[(i+1)*n-1];
		}
		cout << "OwO" << endl;
		aux = A;
		A = Temp;
		Temp = aux;
		print(A, n);
	}
	free(Temp);
	APointer[0] = A;
}


void printResults(int n, double timeCpuGpu, double timeRunPar, double timeGpuCpu){
	cout << n << ";" << timeCpuGpu << ";" << timeRunPar << ";" << timeGpuCpu << endl;
}

int  main(int argc, char** argv) {
	int n=0, blockSize;
	double *A, *Adevice;
	double begin, end, timeCpuGpu = 0, timeRun = 0, timeGpuCpu = 0;	
	if(argc < 3) {
		cerr << "Digite: "<< argv[0] <<" <Dimensão da matriz> <Tempo total> <Delta T> [Dimensão do bloco]" << endl;
		exit(EXIT_FAILURE);
	}
	n = atol(argv[1]);
	int deltaT = atol(argv[3]);
	int rounds = atol(argv[2])/deltaT;

	size_t matBytes = n*n*sizeof(double);
	A = (double *) malloc(matBytes);
	if ( A == NULL   ) {
		cerr << "Memoria  insuficiente" << endl;
		exit(EXIT_FAILURE);
	}
	setupMatrix(A, n);
	print(A, n);
	if (argc > 3){
		blockSize = atol(argv[4]);
		GET_TIME(begin);
		CUDA_SAFE_CALL(cudaMalloc((void**) &Adevice, matBytes));
		CUDA_SAFE_CALL(cudaMemcpy(Adevice, A, matBytes, cudaMemcpyHostToDevice));
		GET_TIME(end);
		timeCpuGpu = end-begin;
		
		GET_TIME(begin);
		playRounds(&Adevice, n, blockSize, rounds, deltaT);
		GET_TIME(end);
		timeRun = end-begin;
		
		GET_TIME(begin);
		CUDA_SAFE_CALL(cudaMemcpy(A, Adevice, matBytes, cudaMemcpyDeviceToHost));
		GET_TIME(end);
		timeGpuCpu = end-begin;
		// print(A, n);
		
		CUDA_SAFE_CALL(cudaFree(Adevice));
	} else {
		cout << "OwO" << endl;
		GET_TIME(begin);
		cout << "OwO" << endl;
		playRoundsSeq(&A, n, rounds, deltaT);
		GET_TIME(end);
		timeRun = end-begin;
	}
	printResults(n, timeCpuGpu, timeRun, timeGpuCpu);
	free(A);
	CUDA_SAFE_CALL(cudaDeviceReset());
	exit(EXIT_SUCCESS);
}
