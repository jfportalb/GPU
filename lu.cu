#include <stdio.h>
#include <sys/time.h>
#include <iostream>
using namespace std;

/**
 * Para checar erros em chamadas Cuda
 */
#define CUDA_SAFE_CALL(call) { \
	cudaError_t err = call;     \
	if(err != cudaSuccess) {    \
		fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n",__FILE__, __LINE__,cudaGetErrorString(err)); \
		exit(EXIT_FAILURE); } \
}

__global__  void  lu_calc_col( double* d_m , int dim , int i ) {
	__shared__  double  a_ii;
	if (threadIdx.x == 0) {
		a_ii = d_m[i*(dim +1)];
	}
	__syncthreads ();
	int j   = blockIdx.x * blockDim.x + threadIdx.x + i + 1;
	if ( j < dim ) {
		d_m[ j*dim+i ] /= a_ii;
	}
}

__global__  void  lu_calc_subm( double* d_m , int dim , int i) {
	__shared__  double  a_ji[TAM_BLOCO ];
	__shared__  double  a_ik[TAM_BLOCO ];
	int j = blockDim.x * blockIdx.x + threadIdx.x + i + 1;
	int k = blockDim.y * blockIdx.y + threadIdx.y + i + 1;
	if (( threadIdx.y == 0) && (j < dim)) {
		a_ji[threadIdx.x] = d_m[ j*dim + i ];
	}
	if (( threadIdx.x == 0) && (k < dim)) {
		a_ik[threadIdx.y] = d_m[ i*dim + k ];
	}
	__syncthreads ();
	if ((j < dim) && (k < dim)) {
		d_m[ j*dim + k ] -= a_ji[threadIdx.x] * a_ik[threadIdx.y];
	}
}

void  alg_lu_gpu( double* d_m , int  dim) {
	int i, n_blocos , TAM_BLOCO = 32;
	for (i = 0; i < dim -1; i++) {
		n_blocos = ((dim -i-1) + TAM_BLOCO -1) / TAM_BLOCO;
		dim3  g_blocos(n_blocos , n_blocos);
		dim3  n_threads(TAM_BLOCO ,TAM_BLOCO);
		lu_calc_col <<< n_blocos , TAM_BLOCO  >>>(d_m , dim , i);
		CUDA_SAFE_CALL(cudaGetLastError ());
		lu_calc_subm <<< g_blocos , n_threads  >>>(d_m , dim , i);
		CUDA_SAFE_CALL(cudaGetLastError ());
	}
}

int  main() {
	int  dim_mat;
	double* m;
	
	if(argc < 2) {
		cerr << "Digite: "<< argv[0] <<" <Dimensão da matriz>" << endl;
		exit(EXIT_FAILURE);
	}
	dim_mat = atol(argv[1]);
	
	size_t  quant_mem = dim_mat*dim_mat*sizeof(double);
	m = (double *) malloc(quant_mem);
	if ( m == NULL   ) {
		cerr << "Memoria  insuficiente" << endl;
		exit(EXIT_FAILURE);
	}
	// TODO Adicionar código para preencher a matriz e criar outros dados necessários para seu problema
	
	// Alocar  memória na GPU  para  copiar a matriz
	double* d_m;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_m, quant_mem));
	// copiar a matriz  para a GPU
	CUDA_SAFE_CALL(cudaMemcpy(m, d_m, quant_mem, cudaMemcpyDeviceToHost));
	// executar a fatora ̧cão  na GPU
	alg_lu_gpu(d_m, dim_mat);
	// copiar o resultado  da GPU  para a CPU
	CUDA_SAFE_CALL(cudaMemcpy(m, d_m, quant_mem, cudaMemcpyDeviceToHost));
	// limpar a mem ́oria  da GPU
	CUDA_SAFE_CALL(cudaFree(d_m));
	// adicionar  código  para  usar as  matrizes L e U (contidas  em m) e os  outros  dados
	free(m);
	CUDA_SAFE_CALL(cudaDeviceReset());
	exit(EXIT_SUCCESS);
}
