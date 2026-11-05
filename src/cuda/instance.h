#ifndef INCLUDE_INSTANCE
#define INCLUDE_INSTANCE

#include <iostream>
#include <cstdint>
#include "m4ri/m4ri.h"

///
/// \param devProp
/// \return
int getSPcores(cudaDeviceProp devProp) {
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major){
		case 2: // Fermi
			if (devProp.minor == 1) cores = mp * 48;
			else cores = mp * 32;
			break;
		case 3: // Kepler
			cores = mp * 192;
			break;
		case 5: // Maxwell
			cores = mp * 128;
			break;
		case 6: // Pascal
			if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
			else if (devProp.minor == 0) cores = mp * 64;
			else printf("Unknown device type\n");
			break;
		case 7: // Volta and Turing
			if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
			else printf("Unknown device type\n");
			break;
		case 8: // Ampere
			if (devProp.minor == 0) cores = mp * 64;
			else if (devProp.minor == 6) cores = mp * 128;
			else printf("Unknown device type\n");
			break;
		default:
			printf("Unknown device type\n");
			break;
	}

	return cores;
}


// problem instance for the algorithm.
struct DecodingInstance {
	// some cuda kernel information
	size_t blocks;
	size_t tpb;

	// CPU Parameters
	mzd_t *AT;          // parity check matrix transposed: n \times n-k
	mzd_t *A;           // parity check matrix: n-k \times n
	mzd_t *syndrom;     // syndrome 1 \times n-k
	mzd_t *syndromT;    // syndrome: n-k \times 1

	/// print some information
	void print() {
		std::cout << "Blocks: " << blocks 
				  << ", threads per block: " << tpb 
				  << ", total threads: " << blocks*tpb << "\n";
	}

	DecodingInstance(
		const char *H,
		const char *S,
		const uint32_t n, const uint32_t k,
		const uint32_t set_tpb) {

		cudaSetDevice(0);
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		const uint32_t ncores = getSPcores(prop);
	
		this->tpb      = MIN(1024, set_tpb);
		this->blocks   = MAX(1, set_tpb / tpb);
		this->AT = mzd_from_str(n, n - k, H),
		this->syndrom = mzd_from_str(1, n - k, S),
	
		this->A = mzd_transpose(nullptr, this->AT);
		this->syndromT = mzd_transpose(nullptr, this->syndrom);
	}

	~DecodingInstance() {
		mzd_free(this->syndrom);
		mzd_free(this->syndromT);
		mzd_free(this->A);
		mzd_free(this->AT);
	}
};

#endif
