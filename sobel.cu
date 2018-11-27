#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <cstdint>
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

/**
 * Kernel para execucao paralela em CUDA
 */
__global__ void applyMask(uint8_t *image, uint8_t *ret, int width, int heigth, int colors) {
	int i = blockDim.x * blockIdx.x + threadIdx.x +1;
	int j = blockDim.y * blockIdx.y + threadIdx.y +1;
	int c = threadIdx.z;
	if (i<heigth && j<width){
		int gx = image[c+colors*((i-1)*width + j-1)] + 2*image[c+colors*((i-1)*width+j)] + image[c+colors*((i-1)*width+j+1)] - image[c+colors*((i+1)*width+j-1)] - 2*image[c+colors*((i+1)*width+j)] - image[c+colors*((i+1)*width+j+1)];
		int gy = image[c+colors*((i-1)*width+j-1)] + 2*image[c+colors*(i*width+j-1)] + image[c+colors*((i+1)*width+j-1)] - image[c+colors*((i-1)*width+j+1)] - 2*image[c+colors*(i*width+j+1)] - image[c+colors*((i+1)*width+j+1)];
		double g = sqrtf(gx*gx + gy*gy)/4;
		ret[c+colors*(i*width+j)] = (uint8_t) g;
	}
}

/**
 * Imprime os resultados do programa
 */
void printResults(unsigned int width, unsigned int heigth, unsigned int colors, unsigned int blockLines, unsigned int blockColumns, float delta_eventos, double initialParTime, double finalParTime, bool csv = true){
	if (csv) {
		cout << width << ";" << heigth << ";" << colors << ";" << blockLines << ";" << blockColumns << ";" << delta_eventos/1000 << ";" << initialParTime << ";" << finalParTime << ";" << endl;
	} else {
		cout << "Dimensões da imagem = " << width << " x " << heigth << " (x"<<colors<<")" << endl
			 << "Dimensões dos blocos = " << blockLines << " x " << blockColumns << endl
			 << "Tempo paralelo kernel = "<< delta_eventos/1000 << " seg" << endl
			 << "Tempo paralelo begin = "<< initialParTime <<" seg" << endl
			 << "Tempo paralelo end    = "<< finalParTime <<" seg" << endl
			 << "Tempo paralelo total  = "<< initialParTime+(delta_eventos/1000)+finalParTime <<" seg" << endl;
	}
}

int main(int argc, char** argv) {
	// TIME
		double begin, end, initialParTime, finalParTime;
		float delta_eventos = 0;
		cudaEvent_t start, stop;

	// INPUT
		unsigned int width,heigth,colors, blockLines, blockColumns;
		long int imageBytes; //qtde bytes por matriz
		uint8_t *image; //matrizes host
		uint8_t *original, *result; //matrizes device

	// GET INPUT
		if(argc < 5) {
			cerr << "Digite: "<< argv[0] <<" <largura da imagem> <altura da imagem> <cores na imagem (1 para escala de cinza ou 3 para rgb)> <nº de linhas e colunas dos blocos>" << endl;
			exit(EXIT_FAILURE);
		}
		//dimensao das matrizes e tamanho dos blocos
		width = atol(argv[1]);
		heigth = atol(argv[2]);
		colors = atol(argv[3]);
		blockLines = atol(argv[4]);
		blockColumns = atol(argv[4]);

	// LOAD IMAGE
		imageBytes = width*heigth*colors*sizeof(uint8_t);
		image = (uint8_t *) malloc(imageBytes);
		if ( image == NULL   ) {
			cerr << "Memoria  insuficiente" << endl;
			exit(EXIT_FAILURE);
		}
		ifstream infile ("image.bin", ios::binary);
		infile.read(reinterpret_cast<char *>(image), imageBytes);
		infile.close();

	// ALLOCATE SPACE AND COPY IMAGE TO DEVICE (CPU → GPU)
		GET_TIME(begin);
		// Aloca espaço para as matrizes na GPU
		CUDA_SAFE_CALL(cudaMalloc((void**) &original, imageBytes));
		CUDA_SAFE_CALL(cudaMalloc((void**) &result, imageBytes));
		
		// Copia as matrizes de entrada da CPU para a GPU (host para device)
		CUDA_SAFE_CALL(cudaMemcpy(original, image, imageBytes, cudaMemcpyHostToDevice));

		// Invoca o kernel com blocos de tamanhos fixos
		dim3 threadsBlock = {blockLines, blockColumns, colors};
		dim3 blocksGrid = {(heigth + threadsBlock.x - 3)/threadsBlock.x, (width + threadsBlock.y - 3)/threadsBlock.y, 1};
		int tamMemCompartilhada = threadsBlock.x*threadsBlock.y*8*2;
		GET_TIME(end);
		initialParTime = end-begin; // Calcula o tempo das inicializações paralelo em segundos

	// KERNEL RUN
		CUDA_SAFE_CALL(cudaEventCreate(&start));
		CUDA_SAFE_CALL(cudaEventCreate(&stop));
		CUDA_SAFE_CALL(cudaEventRecord(start));
		applyMask<<<blocksGrid, threadsBlock, tamMemCompartilhada>>>(original, result, width, heigth, colors);
		CUDA_SAFE_CALL(cudaGetLastError());
		CUDA_SAFE_CALL(cudaEventRecord(stop));
		CUDA_SAFE_CALL(cudaEventSynchronize(stop));
		CUDA_SAFE_CALL(cudaEventElapsedTime(&delta_eventos, start, stop));

	// GET RESULT FROM DEVICE (GPU → CPU)
		GET_TIME(begin);
		CUDA_SAFE_CALL(cudaMemcpy(image, result, imageBytes, cudaMemcpyDeviceToHost))
		GET_TIME(end);
		finalParTime = end-begin; // calcula o tempo das finalizacoes paralelo em segundos

	// SAVE RESULT IN FILE
		ofstream outfile ("imageOut.bin", ios::binary);
		outfile.write(reinterpret_cast<char *>(image), imageBytes);
		outfile.close();

	// FREE MEMORY
		CUDA_SAFE_CALL(cudaFree(original));
		CUDA_SAFE_CALL(cudaFree(result));
		free(image);

	// PRINT TIMES
		printResults(width, heigth, colors, blockLines, blockColumns,
			delta_eventos, initialParTime, finalParTime);

	return 0;
}
