#include "cuhelper.cuh"
#include <stdint.h>

#ifndef MIN
#   define MIN(x,y) ((x)<(y)?(x):(y))
#endif

#ifndef MAX
#   define MAX(x,y) ((x)>(y)?(x):(y))
#endif

#define hk 8
#define hcol 160
#define offset2 2
#define BlockDimCalcGray 40

// dimension one
//texture<uint32_t, 1, cudaReadModeElementType> texRef;
__device__ uint32_t texRef2[(1 << hk) * hcol];


// src= https://www.yumpu.com/de/document/read/10824881/effizientes-losen-linearer-gleichungssysteme-uber-gf2-cdc-
__global__ void
kernel_gray(uint32_t *d_g, uint32_t *matr, int start) {
	int trow = threadIdx.x;
	int tcol = threadIdx.y * offset2;
	int blockid = blockIdx.y;
	__shared__ uint32_t refForm[hk][hcol / BlockDimCalcGray];
	uint32_t value[offset2] = {0};
	int modrow = trow % hk;

	for (int l = 0; l < offset2; l++) {
		refForm[modrow][tcol + l] = matr[(start + modrow) * hcol +
		                                 (blockid * (hcol / BlockDimCalcGray)) +
		                                 tcol + l];
		value[l] = 0;
	}

	__syncthreads();

	for (int m = 0; m < hk; m++) {
		if (((trow >> m) & 1) == 1) {
			for (int l = 0; l < offset2; l++) {
				value[l] = value[l] ^ refForm[hk - m - 1][tcol + l];
			}
		}
	}

	for (int l = 0; l < offset2; l++) {
		d_g[trow * hcol + (blockid * (hcol / BlockDimCalcGray)) + tcol + l] = value[l];
	}

}

__global__ void
kernel_gray_xor(uint32_t *matr, int e, int off, int i, int offset) {
	// add block offset for row x-dim
	matr = matr + blockDim.x * blockIdx.x * hcol;
	int result = (matr[threadIdx.x * hcol + e] >> off) & ((1 << hk) - 1);
	__syncthreads();

	int num = hcol / offset;
	int start = e / num;

	if ((((threadIdx.x + blockDim.x * blockIdx.x) > (3 * hk + i - 2))
	     or ((threadIdx.x + blockDim.x * blockIdx.x) < i))
	    and (result)) {
		// calc pos in matr
		unsigned int k = threadIdx.x * hcol;
		// calc position in row
		unsigned int j = threadIdx.y;

		for (int l = start; l < offset; l++) {
			//matr[i + l*num + j] = matr[i + l*num + j] ^ tex1Dfetch(texRef, (result)*hcol + j + l*num);
			matr[k + l * num + j] = matr[k + l * num + j] ^ texRef2[result * hcol + j + l * num];

		}
	}
}

const uint32_t pieces = 32;
const uint32_t row = hcol * 2; // 9984
const uint32_t offset = 10; // number of words on thread execute
const uint32_t knumrow = 3 * hk - 1;
const uint32_t dimgridxgray = row;//9984;

void print_data(uint32_t **data, int nrows, int ncols) {
	for (int p = 0; p < pieces; p++) {
		for (int i = 0; i < (row / pieces); i++) {
			printf(" %u . row %u", i + p * (row / pieces), data[i][0]);
			for (int l = 0; l < ncols; l++) {
				printf("%u", data[i][l]);
			}
			printf("\n");
		}
	}
}

void fourrussians(int nrows, int ncols) {
	uint32_t sizeelem = 32;
	int e = 0;
	uint32_t gauss[knumrow][hcol];
	uint32_t mem_size_gaus = sizeof(uint32_t) * knumrow * ncols;

	dim3 dimBlock2((int) (nrows / dimgridxgray), (int) (ncols / offset));
	dim3 dimBlockGrayCalc((int) 1 << hk, MAX(1, (int) (hcol / offset2) / BlockDimCalcGray));

	// add an empty z dimension
	dim3 dimGridGray(dimgridxgray, 1);
	dim3 dimGridGrayCalc(1, BlockDimCalcGray);

	// allocate mem on device
	uint32_t size_M = nrows * ncols;
	uint32_t mem_size_M = sizeof(uint32_t) * size_M;

	uint32_t size_Gray = (1 << hk) * ncols;
	uint32_t mem_size_gray = sizeof(uint32_t) * size_Gray;

	uint32_t *d_g;
	checkCudaErrors(cudaMalloc(&d_g, mem_size_gray));

	uint32_t *d_m;
	checkCudaErrors(cudaMalloc(&d_m, mem_size_M));
	uint32_t dataorg[row / pieces][hcol];

	std::cout << "starting rng\n";


	// TODO init matrix? maybe directly copy it from m4ri
	for (int l = 0; l < pieces; l++) {
		for (int i = 0; i < (row / pieces); i++) {
			for (int j = 0; j < hcol - 1; j++) {
				// random 32 bit number
				//dataorg[i][j] = random()*((l+1) * l) * pow(double(-1), double(random()%32));
				dataorg[i][j] = rand() & 1;
			}
		}
		checkCudaErrors(
				cudaMemcpy(d_m + l * (hcol * (nrows / pieces)), dataorg, mem_size_M / pieces, cudaMemcpyHostToDevice));
	}

	std::cout << "starting m4ri\n";


	// calc row echolon form of size k x k
	for (int i = 0; i < MIN(nrows - knumrow, (ncols - 1) * sizeelem); i = i + hk) {
		std::cout << "row: " << i << "\n";
		std::cout << "gauss: " << gauss << " " << d_m + (i * ncols) << " " << mem_size_gaus << "\n";

		checkCudaErrors(cudaMemcpy(gauss, d_m + (i * ncols), mem_size_gaus, cudaMemcpyDeviceToHost));

		for (uint32_t m = 0; m << hk; m++) {
			// master column
			e = (i + m) / sizeelem;
			// master bit
			int n = ((e + 1) * sizeelem) - (i + m);


			int notchanged = 1;
			uint32_t count = m + 1;
			uint32_t pos = 1u << (n - 1); // 2^i;

			if ((gauss[m][e] & pos) != pos) {
				do {
					if ((gauss[count][e] & pos) == pos) {
						notchanged = 0;
						for (int o = 0; o < hcol; o++) {
							gauss[m][o] ^= gauss[count][o];
						}
					}

					count += 1;
				} while (notchanged and (count < knumrow));

				if (notchanged) {
					count = 0;
				}
			}

			for (int o = 0; o < knumrow; o++) {
				if (((gauss[o][e] & (1u << (n - 1))) >> (n - 1))
				    and (o != m)) {
					for (int p = 0; p < ncols; p++) {
						gauss[o][p] ^= gauss[m][p];
					}
				}
			}
		}


		e = i / sizeelem;
		checkCudaErrors(cudaMemcpy(d_m + (i * ncols), gauss, mem_size_gaus, cudaMemcpyHostToDevice));

		kernel_gray<<<dimGridGrayCalc, dimBlockGrayCalc>>>(d_g, d_m, i);
		checkCudaErrors(cudaGetLastError());

		//cudaBindTexture(0, texRef, d_g, sizeof(uint32_t) * ((1 << hk) * hcol));
		cudaMemcpyToSymbol(texRef2, d_g, sizeof(uint32_t) * ((1 << hk) * hcol));

		// offest in one word
		int offsetgray = sizeelem - hk - (i % sizeelem);
		kernel_gray_xor<<<dimGridGray, dimBlock2>>>(d_m, e, offsetgray, i, offset);
		checkCudaErrors(cudaGetLastError());
	}

	// deallocate
	checkCudaErrors(cudaFree(d_g));

	// print matrix
	for (int p = 0; p < pieces; p++) {
		checkCudaErrors(
				cudaMemcpy(dataorg, d_m + (p * (row / pieces * hcol)), mem_size_M / pieces, cudaMemcpyDeviceToHost));

		for (int i = 0; i < (row / pieces); i++) {
			printf(" %u . row %u", i + p * (row / pieces), dataorg[i][0]);
			for (int l = 1; l < ncols; l++) {
				printf("%u", dataorg[i][l]);
			}
			printf("\n");
		}
	}
	checkCudaErrors(cudaFree(d_m));

}

int test4rus() {
	fourrussians(row, hcol);
	return 0;
}