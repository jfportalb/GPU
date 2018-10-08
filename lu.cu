#include <stdio.h>
#include <sys/time.h>
#include <iostream>
using namespace std;

#define TAM_BLOCO 32

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

int  main() {
	int  dim_mat;
	double* m;
	// adicionar   c ́odigo  para  inicializar a vari ́avel
	// dim_mat (dimens~ao  da  matriz)
	size_t  quant_mem = dim_mat*dim_mat*sizeof(double);
	m = (double *) malloc(quant_mem);
	if ( m == NULL   ) {
		fprintf(stderr , "Memoria  insuficiente\n");
		exit(EXIT_FAILURE);
	}
	// adicionar   c ́odigo  para  preencher a matriz
	// e criar  outros  dados  necess ́arios  para  seu  problema
	// alocar  mem ́oria   na GPU  para  copiar a matriz
	double* d_m;
	CUDA_SAFE_CALL( cudaMalloc( (void **) &d_m , quant_mem ));
	// copiar a matriz  para a GPU
	CUDA_SAFE_CALL( cudaMemcpy(m, d_m , quant_mem ,
	cudaMemcpyDeviceToHost));
	// executar a fatora ̧c~ao  na GPU
	alg_lu_gpu(d_m , dim_mat);
	// copiar o resultado  da GPU  para a CPU
	CUDA_SAFE_CALL( cudaMemcpy(m, d_m , quant_mem ,
	cudaMemcpyDeviceToHost));
	// limpar a mem ́oria  da GPU
	CUDA_SAFE_CALL(cudaFree(d_m));
	// adicionar  c ́odigo  para  usar
	//as  matrizes L e U (contidas  em m)
	//e os  outros  dados
	free(m);
	CUDA_SAFE_CALL( cudaDeviceReset () );
	exit(EXIT_SUCCESS);
}
