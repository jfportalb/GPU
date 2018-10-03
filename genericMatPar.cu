/**
 * Descrição: Multiplicação de matrizes em paralelo usando GPU
 * Entrada: Dimensão das matrizes e dos blocos de threads
 * Saída: Tempos de execução 
 */
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
/**
 * Funcao para execucao sequencial
 */
void multMatSeq(float *a, float *b, float *c, int mA, int nAmB, int nB) {
   int i, j, k;
   float soma;
   for(i=0; i<mA; i++)
      for(j=0; j<nB; j++) {
         soma = 0;
         for(k=0; k<nAmB; k++) {
            soma += a[i*nAmB+k] * b[k*nB+j];
         }
         c[i*nB+j] = soma;
      }
}

/**
 * Kernel para execucao paralela em CUDA
 */
__global__ void multMatPar(float *a, float *b, float *c, int mA, int nAmB, int nB) {
	// Coordenadas globais da thread
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	// Coordenadas locais da thread
	int i_bloco = threadIdx.x;
	int j_bloco = threadIdx.y;
	extern __shared__ float mat_sub[];
	// Memória compartilhada para a submatriz de A
	float* Asub = (float*) mat_sub;
	// Memória compartilhada para a submatriz de B
	float* Bsub= (float*) &Asub[blockDim.x*blockDim.y];
	float valor = 0;
	for(int passo=0; passo<nAmB; passo+=blockDim.y) {
		if (i < mA && (passo+j_bloco) < nAmB)
			Asub[i_bloco*blockDim.y+j_bloco] = a[i*nAmB+passo+j_bloco];
		if ((passo+i_bloco) < nAmB && j < nB)
			Bsub[i_bloco*blockDim.y+j_bloco] = b[(passo+i_bloco)*nAmB+j];
		__syncthreads();
		if (i < mA && j < nB)
			for (int k = 0; k < blockDim.y; k++) {
				valor += Asub[i_bloco*blockDim.y+k] *	Bsub[k*blockDim.y+j_bloco];
			}
		__syncthreads();
	}
	if (i < mA && j < nB)
		c[i*nB+j] = valor;
}

/**
 * Função que aloca espaco para uma matriz e preenche seus valores
 * Entrada: matriz de entrada, dimensoes da matriz
 * Saída: retorna 1 se a matriz foi preenchida com sucesso e 0 caso contrario
 */
int preencheMatriz(float **mat, int linhas, int colunas) {
	int i, j;
	//aloca espaco de memoria para a matriz
	*mat = (float*) malloc(sizeof(float) * linhas * colunas);
	if (mat == NULL) return 0;
	//preenche o vetor
	for (i=0; i<linhas; i++) {
		for (j=0; j<colunas; j++) {
			*((*mat) + (i*colunas+j)) = 1.5;
		}
	}
	return 1;
}

void checkResults(float *mat1, float *mat2, int m, int n){
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			if (fabs(mat1[i*n+j] - mat2[i*n+j]) > 1e-5) {
				fprintf(stderr, "resultado incorreto\n");
				exit(EXIT_FAILURE);
			}
		}
	}
}

/**
 * Imprime os resultados do programa
 */
void printResults(unsigned int mA, unsigned int nA, unsigned int mB, unsigned int nB, unsigned int blockLines, unsigned int blockColumns, double tempoSeq, double delta_eventos, double initialParTime, double finalParTime, bool csv = true){
	if (csv) {
		cout << mA << ";" << nA << ";" << mB << ";" << nB << ";" << blockLines << ";" << blockColumns << ";" << tempoSeq << ";" << delta_eventos/1000 << ";" << initialParTime << ";" << finalParTime << ";" << endl;
	} else {
		cout << "Dimensões da matriz A = " << mA << " x " << nA << endl
			 << "Dimensões da matriz B = " << mB << " x " << nB << endl
			 << "Dimensões dos blocos = " << blockLines << " x " << blockColumns << endl
			 << "Tempo sequencial = "<< tempoSeq <<" seg" << endl
			 << "Tempo paralelo kernel = "<< delta_eventos/1000 << " seg" << endl
			 << "Tempo paralelo begin = "<< initialParTime <<" seg" << endl
			 << "Tempo paralelo end    = "<< finalParTime <<" seg" << endl
			 << "Tempo paralelo total  = "<< initialParTime+(delta_eventos/1000)+finalParTime <<" seg" << endl;
	}
}

//funcao principal
int main(int argc, char** argv) {
	float *h_a, *h_b, *h_c, *h_c_seq; //matrizes host
	float *d_a, *d_b, *d_c; //matrizes device
	//para medidas de tempo
	double begin, end, initialParTime, finalParTime, tempoSeq;
	cudaEvent_t start, stop;
	//entrada de dados
	unsigned int mA, nA, mB, nB; // Dimensão das matrizes de entrada
	long int bytesA, bytesB, bytesC; //qtde bytes por matriz
	
	//tamanho dos blocos de threads
	unsigned int blockLines, blockColumns;

	//le e valida os parametros de entrada
	if(argc < 6) {
		cerr << "Digite: "<< argv[0] <<" <nº de linhas da matriz A> <nº de colunas da matriz A> <nº de linhas da matriz B> <nº de colunas da matriz B> <nº de linhas e colunas dos blocos>" << endl;
		exit(EXIT_FAILURE);
	}
	//dimensao das matrizes e tamanho dos blocos
	mA = atol(argv[1]);
	nA = atol(argv[2]);
	mB = atol(argv[3]);
	nB = atol(argv[4]);
	blockLines = atol(argv[5]);
	blockColumns = atol(argv[5]);

	if (nA != mB) {
		cerr << "Impossível executar multiplicação das matrizes. Número de colunas da matriz A ("<< nA <<") não bate com o número de colunas da matriz B ("<< mB <<")" << endl;
		exit(EXIT_FAILURE);
	}

	//calcula o tamanho em bytes das matrizes
	bytesA = mA*nA*sizeof(float);
	bytesB = mB*nB*sizeof(float);
	bytesC = mA*nB*sizeof(float);

	// Aloca e preenche a matriz de entrada A
	if (preencheMatriz(&h_a, mA, nA) == 0) {
		cerr << "Erro de preenchimento da matriz de entrada A" << endl;
		exit(EXIT_FAILURE);
	}
	// Aloca e preenche a matriz de entrada B
	if (preencheMatriz(&h_b, mB, nB) == 0) {
		cerr << "Erro de preenchimento da matriz de entrada B" << endl;
		exit(EXIT_FAILURE);
	}
	// Aloca a matriz de saída paralelo
	h_c = (float*) malloc(bytesC);
	if (h_c == NULL) {
		cerr << "Erro de alocacao da matriz de saida" << endl;
		exit(EXIT_FAILURE);
	}
	// Aloca a matriz de saída sequencial
	h_c_seq = (float*) malloc(bytesC);
	if (h_c_seq == NULL) {
		cerr << "Erro de alocacao da matriz de saida" << endl;
		exit(EXIT_FAILURE);
	}
	
	//!!! ------------------------ executa sequencial ---------------------------------- !!!//
	GET_TIME(begin);
	multMatSeq(h_a, h_b, h_c_seq, mA, nA, nB);
	GET_TIME(end);

	tempoSeq = end-begin; // calcula o tempo sequencial em segundos
	
	//!!! ------------------------ executa em paralelo em CUDA -------------------------- !!!//
	GET_TIME(begin);
	// Aloca espaço para as matrizes na GPU
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_a, bytesA));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_b, bytesB));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_c, bytesC));
	
	// Copia as matrizes de entrada da CPU para a GPU (host para device)
	CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, bytesA, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, bytesB, cudaMemcpyHostToDevice));

	// Invoca o kernel com blocos de tamanhos fixos
	dim3 threadsBloco = {blockLines, blockColumns, 1};
	dim3 blocosGrade = {(nB + threadsBloco.x - 1)/threadsBloco.x, (mA + threadsBloco.y - 1)/threadsBloco.y, 1};
	int tamMemCompartilhada = blockLines*blockColumns*4*2;
	GET_TIME(end);
	initialParTime = end-begin; // Calcula o tempo das inicializações paralelo em segundos
	
	//dispara o kernel
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));
	CUDA_SAFE_CALL(cudaEventRecord(start));
	multMatPar<<<blocosGrade, threadsBloco, tamMemCompartilhada>>>(d_a, d_b, d_c, mA, nA, nB);
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaEventRecord(stop));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));
	float delta_eventos = 0;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&delta_eventos, start, stop));

	//copia resultado da GPU para a CPU (device para host)
	GET_TIME(begin);
	CUDA_SAFE_CALL(cudaMemcpy(h_c, d_c, bytesC, cudaMemcpyDeviceToHost))
	GET_TIME(end);
	finalParTime = end-begin; // calcula o tempo das finalizacoes paralelo em segundos
	
	// Libera a memória na GPU
	CUDA_SAFE_CALL(cudaFree(d_a));
	CUDA_SAFE_CALL(cudaFree(d_b));
	CUDA_SAFE_CALL(cudaFree(d_c));

	// Libera a memória na CPU
	free(h_a);
	free(h_b);
	free(h_c);

	//------------------------------- imprime dos tempos de execucao ----------------------//
	printResults(mA, nA, mB, nB, blockLines, blockColumns, tempoSeq, delta_eventos, initialParTime, finalParTime);

	return 0;
}
