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

__global__ void applyMaskWithSharedMemory(uint8_t *image, uint8_t *ret, int width, int heigth, int colors) {
	__shared__ uint8_t *subimage;
	int i = blockDim.x * blockIdx.x + threadIdx.x +1;
	int j = blockDim.y * blockIdx.y + threadIdx.y +1;
	int c = threadIdx.z;
	if (i<heigth && j<width){
		subimage[c+colors*((threadIdx.x+1)*width + threadIdx.y+1)] = image[c+colors*((i)*width+j)];
		if (!threadIdx.x){
			subimage[c+colors*((threadIdx.x)*width + threadIdx.y+1)] = image[c+colors*((i-1)*width+j)];
			if (!threadIdx.y){
				subimage[c+colors*((threadIdx.x)*width + threadIdx.y)] = image[c+colors*((i-1)*width+j-1)];
			}
		}
		if (!threadIdx.y){
			subimage[c+colors*((threadIdx.x+1)*width + threadIdx.y)] = image[c+colors*((i)*width+j-1)];
		}
		int gx = image[c+colors*((i-1)*width + j-1)] + 2*image[c+colors*((i-1)*width+j)] + image[c+colors*((i-1)*width+j+1)] - image[c+colors*((i+1)*width+j-1)] - 2*image[c+colors*((i+1)*width+j)] - image[c+colors*((i+1)*width+j+1)];
		int gy = image[c+colors*((i-1)*width+j-1)] + 2*image[c+colors*(i*width+j-1)] + image[c+colors*((i+1)*width+j-1)] - image[c+colors*((i-1)*width+j+1)] - 2*image[c+colors*(i*width+j+1)] - image[c+colors*((i+1)*width+j+1)];
		double g = sqrtf(gx*gx + gy*gy)/4;
		ret[c+colors*(i*width+j)] = (uint8_t) g;
	}
}

/**
 * Imprime os resultados do programa sequencial
 */
void printResultsSeq(unsigned int width, unsigned int heigth, unsigned int colors, double tempoSeq, bool csv = true){
	if (csv) {
		cout << width << ";" << heigth << ";" << colors << ";" << tempoSeq<< endl;
	} else {
		cout << "Dimensões da imagem = " << width << " x " << heigth << " (x"<<colors<<")" << endl
			 << "Tempo sequencial = "<< tempoSeq << " seg" << endl;
	}
}

/**
 * Imprime os resultados do programa paralelo
 */
void printResultsPar(unsigned int width, unsigned int heigth, unsigned int colors, unsigned int blockDim, float delta_eventos, double initialParTime, double finalParTime, bool csv = true){
	if (csv) {
		cout << width << ";" << heigth << ";" << colors << ";" << blockDim << ";" << delta_eventos/1000 << ";" << initialParTime << ";" << finalParTime << endl;
	} else {
		cout << "Dimensões da imagem = " << width << " x " << heigth << " (x"<<colors<<")" << endl
			 << "Dimensões dos blocos = " << blockDim << " x " << blockDim << endl
			 << "Tempo paralelo kernel = "<< delta_eventos/1000 << " seg" << endl
			 << "Tempo paralelo begin = "<< initialParTime <<" seg" << endl
			 << "Tempo paralelo end    = "<< finalParTime <<" seg" << endl
			 << "Tempo paralelo total  = "<< initialParTime+(delta_eventos/1000)+finalParTime <<" seg" << endl;
	}
}

void applyMaskPar(uint8_t **imagePointer, int width, int heigth, int colors, int blockDim, bool shared){
		long int imageBytes = width*heigth*colors*sizeof(uint8_t);
		double begin, end;
		uint8_t *image = imagePointer[0];
		double initialParTime, finalParTime;
		float delta_eventos = 0;
		cudaEvent_t start, stop;
		uint8_t *original, *result; //matrizes device
		// ALLOCATE SPACE AND COPY IMAGE TO DEVICE (CPU → GPU)
			GET_TIME(begin);
			// Aloca espaço para as matrizes na GPU
			CUDA_SAFE_CALL(cudaMalloc((void**) &original, imageBytes));
			CUDA_SAFE_CALL(cudaMalloc((void**) &result, imageBytes));
			
			// Copia as matrizes de entrada da CPU para a GPU (host para device)
			CUDA_SAFE_CALL(cudaMemcpy(original, image, imageBytes, cudaMemcpyHostToDevice));

			// Invoca o kernel com blocos de tamanhos fixos
			dim3 threadsBlock = {blockDim, blockDim, colors};
			dim3 blocksGrid = {(heigth + threadsBlock.x - 3)/threadsBlock.x, (width + threadsBlock.y - 3)/threadsBlock.y, 1};
			GET_TIME(end);
			initialParTime = end-begin; // Calcula o tempo das inicializações paralelo em segundos

		// KERNEL RUN
			CUDA_SAFE_CALL(cudaEventCreate(&start));
			CUDA_SAFE_CALL(cudaEventCreate(&stop));
			CUDA_SAFE_CALL(cudaEventRecord(start));
			if (shared){
				int tamMemCompartilhada = threadsBlock.x*threadsBlock.y*8*2;
				applyMaskWithSharedMemory<<<blocksGrid, threadsBlock, tamMemCompartilhada>>>(original, result, width, heigth, colors);
			} else {
				applyMask<<<blocksGrid, threadsBlock>>>(original, result, width, heigth, colors);
			}
			CUDA_SAFE_CALL(cudaGetLastError());
			CUDA_SAFE_CALL(cudaEventRecord(stop));
			CUDA_SAFE_CALL(cudaEventSynchronize(stop));
			CUDA_SAFE_CALL(cudaEventElapsedTime(&delta_eventos, start, stop));

		// GET RESULT FROM DEVICE (GPU → CPU)
			GET_TIME(begin);
			CUDA_SAFE_CALL(cudaMemcpy(image, result, imageBytes, cudaMemcpyDeviceToHost))
			GET_TIME(end);
			finalParTime = end-begin; // calcula o tempo das finalizacoes paralelo em segundos

		// FREE MEMORY
			CUDA_SAFE_CALL(cudaFree(original));
			CUDA_SAFE_CALL(cudaFree(result));

			printResultsPar(width, heigth, colors, blockDim, delta_eventos, initialParTime, finalParTime);
}

void applyMaskSeq(uint8_t **imagePointer, int width, int heigth, int colors){
	double begin, end;
	uint8_t *image = imagePointer[0], *ret;
	long int imageBytes = width*heigth*colors*sizeof(uint8_t);
	ret = (uint8_t *) malloc(imageBytes);
	if ( ret == NULL   ) {
		cerr << "Memoria  insuficiente" << endl;
		exit(EXIT_FAILURE);
	}
	double tempoSeq;
	GET_TIME(begin);
	for (int i = 1; i < heigth -1; ++i) {
		for (int j = 0; j < width -1; ++j) {
			for (int c = 0; c < colors; ++c) {
				int gx = image[c+colors*((i-1)*width + j-1)] + 2*image[c+colors*((i-1)*width+j)] + image[c+colors*((i-1)*width+j+1)] - image[c+colors*((i+1)*width+j-1)] - 2*image[c+colors*((i+1)*width+j)] - image[c+colors*((i+1)*width+j+1)];
				int gy = image[c+colors*((i-1)*width+j-1)] + 2*image[c+colors*(i*width+j-1)] + image[c+colors*((i+1)*width+j-1)] - image[c+colors*((i-1)*width+j+1)] - 2*image[c+colors*(i*width+j+1)] - image[c+colors*((i+1)*width+j+1)];
				double g = sqrt(gx*gx + gy*gy)/4;
				ret[c+colors*(i*width+j)] = (uint8_t) g;
			}
		}
	}
	GET_TIME(end);
	tempoSeq = end-begin;
	printResultsSeq(width, heigth, colors, tempoSeq);
	imagePointer[0] = ret;
	free(image);
}

int main(int argc, char** argv) {

	// INPUT
		int width, heigth, colors, blockDim;
		long int imageBytes; //qtde bytes por matriz
		uint8_t *image; //matrizes host
		char *inputFileName, *outputFileName;

	// GET INPUT
		if(argc < 6) {
			cerr << "Digite: "<< argv[0] <<" <largura da imagem> <altura da imagem> <cores na imagem (1 para escala de cinza ou 3 para rgb)> <arquivo de entrada> <arquivo de saída> [nº de linhas e colunas dos blocos]" << endl;
			exit(EXIT_FAILURE);
		}
		//dimensao das matrizes e tamanho dos blocos
		width = atol(argv[1]);
		heigth = atol(argv[2]);
		colors = atol(argv[3]);
		inputFileName = argv[4];
		outputFileName = argv[5];

	// LOAD IMAGE
		imageBytes = width*heigth*colors*sizeof(uint8_t);
		image = (uint8_t *) malloc(imageBytes);
		if ( image == NULL   ) {
			cerr << "Memoria  insuficiente" << endl;
			exit(EXIT_FAILURE);
		}
		ifstream infile (inputFileName, ios::binary);
		infile.read(reinterpret_cast<char *>(image), imageBytes);
		infile.close();

	if (argc > 6){
		blockDim = atol(argv[6]);
		applyMaskPar(&image, width, heigth, colors, blockDim);
	} else {
		applyMaskSeq(&image, width, heigth, colors);
	}
	// SAVE RESULT IN FILE
		ofstream outfile (outputFileName, ios::binary);
		outfile.write(reinterpret_cast<char *>(image), imageBytes);
		outfile.close();

	free(image);

	return 0;
}
