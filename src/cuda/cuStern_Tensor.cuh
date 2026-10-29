#pragma once

#include "cuhelper.cuh"
#include "cuMisc.cuh"

#include <stdint.h>
#include <cuda/atomic>
#include <mma.h>

using namespace nvcuda;
using namespace nvcuda::wmma; //::experimental;


// compile constants
constexpr int WMMA_M = 8;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 128;



template<typename T>
class ConfigDumer {
	/// low level visualisation of the matching procedure used in the algorithm if stern
	///
	///				  check on full length
	///						   ▲
	/// Ts      T       S      │
	/// ┌┐     ┌─┐     ┌─┐     │
	/// │┼────►│ ├────►│ │     │
	/// └┘     │ │     │ │     │l bits
	///     ┌─►│ ├───┐ │ ├─────┴──────┐
	/// ┌┐  │  │ │   └►│ │            │
	/// │┼──┘  │ │     │ │            │
	/// └┘     └─┘     └─┘            │
	/// Te              ▲             │
	///                 │ hashing     │
	///              ┌──┴──┐       ┌──┴──┐
	///              │     │       │     │
	///              │     │       │     │
	///              │ L1  │       │ L2  │
	///              └─────┘       └─────┘
private:
	// disable the default constructor
	constexpr ConfigDumer() : n(0), k(0), w(0), p(0), l(0) {}
public:
	// instance parameters
	const uint32_t n; // code length
	const uint32_t k; // code dimension
	const uint32_t w; // code weight
	const uint32_t p; // weight o enumerate in the baselists
	const uint32_t l; // number of bits to match on the first level.

	// Additional parameters
	// multiplicative factor between the number of limbs needed to represent a column
	// (or row in the transposed matrix) on the cpu and gpu.
	const uint32_t cpu_gpu_width_factor = sizeof(uint64_t) / sizeof(T);
	const uint32_t cpu_matrix_row   = k+l+1;        // CPU Matrix rows of HT, +1 for the syndrome
	const uint32_t cpu_matrix_width = (n-k+63)/64;  // CPU Matrix width in limbs of HT

	// number of limbs per column (or row in the transposed matrix)
	const uint32_t gpu_matrix_width = cpu_gpu_width_factor * cpu_matrix_width;

	const uint32_t llimbpos         = (n-k-1)/(sizeof(T)*8); //IMPORTANT: must be 0 aligned

	const bool stern = false;      // todo implement.

	//
	const uint32_t kl_size          = k + l;
	const uint32_t klpart_size      = cceil(float(kl_size) / 2.);
	const uint64_t numb_p_out_of_k  = bc(uint64_t(klpart_size), uint64_t(p));
	const uint64_t l_table_size     = uint32_t(1) << l;

	// Calculate the optimal `r` value for M4RI
	const uint32_t m4ri_k = matrix_opt_k(n - k, MATRIX_AVX_PADDING(n));

	// if this value is set to a different value then -1 the program will end the
	// computation after `loops` permutation, regardless if it found the solution or not
	const uint64_t loops        = uint64_t(-1);

	// print every `print_loops` a status update of the program
	const uint64_t print_loops  = 10000;

	// check every `exit_loops` if another thread, which is working on a different permutation
	// found the solution. If so exit the computation.
	// NOTE: the check if something is found includes atomic operations, which are quite expensive
	//  so dont check every loop if something is found.
	const uint64_t exit_loops   = 1000;

	// Nr of stream per thread.
	// currently unused
	const uint32_t nr_streams = 1;
	// number of different cpu threads working on different permutations.
	// currently unused
	const uint32_t nr_threads = 1;

	/// constructor
	constexpr ConfigDumer(const uint32_t n, // code length
	                      const uint32_t k, // code dimension
	                      const uint32_t l, // number of bits to match on
	                      const uint32_t w, // code weight
	                      const uint32_t p  // weight to enumerate
	                      ) :
			n(n), k(k), l(l), w(w), p(p) {};

	/// default copy constructor
	constexpr ConfigDumer(ConfigDumer &c) = default;

	// return the number of loops needed in expectation
	constexpr uint64_t Loops() const {
		return bc(n, w) / (bc(n-kl_size, w-p) * bc(kl_size, p));
	}

	/// print information about the configuration
	void print() const {
		std::cout << "n: " << n
		          << ", k: " << k
		          << ", l: " << l
		          <<", (k+l)/2: " << klpart_size
		          << ", n-k:" << n-k
		          << ", p: " << p
				  << ", cpu_gpu_width_factor: " << cpu_gpu_width_factor
		          << ", cpu_matrix_row" << cpu_matrix_row
		          << ", gpu_matrix_width: " << gpu_matrix_width
				  << ", gpu_matrix_width: " << gpu_matrix_width
		          << ", cpu_matrix_width: " << cpu_matrix_width
				  << ", kl_size: " << kl_size
				  << ", klpart_size: " << klpart_size
				  << ", numb_p_out_of_k: " << numb_p_out_of_k
				  << ", l_table_size: " << l_table_size
				  << ", llimbpos: " << llimbpos
				  << ", n-k-l: " << n-k-l << "\n";
	}
};

//template<const ParameterDumer &param, const ConfigDumer<param> &config>
class ParamsDumer {
public:
	// Cuda parameters
	thrust::host_vector<uint16_t> *combinations;
	uint64_t blocks = 0;
	uint64_t tpb = 0;

	uint64_t tableSize = 0;
	uint64_t tableReps = 0;

	// CPU Parameters
	mzd_t *AT = nullptr;
	mzd_t *A = nullptr;
	mzd_t *syndrom = nullptr;
	mzd_t *syndromT = nullptr;

	ParamsDumer (thrust::host_vector<uint16_t> *b,
	            uint64_t blocks,
	            uint64_t tpb,
	            uint64_t tableSize,
	            uint64_t tableReps,
	            mzd_t *AT,
	            mzd_t *syndrome) :
	                               combinations(b),
	                               blocks(blocks),
	                               tpb(tpb),
	                               tableSize(tableSize),
	                               tableReps(tableReps),
	                               AT(AT),
	                               syndrom(syndrome) {};
};



///
template<typename T>
__device__ void tensor_8x8x128(T *a, T *b, T *c) {
	int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

	// Declare the fragments
	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

	wmma::fill_fragment(acc_frag, 0);
	wmma::fill_fragment(c_frag, 0);
	wmma::load_matrix_sync(a_frag, a, 128);
	wmma::load_matrix_sync(b_frag, b, 128);

	// Perform the matrix multiplication
	wmma::bmma_sync(c_frag, a_frag, b_frag, c_frag);

	// Store the output
	wmma::store_matrix_sync(c, c_frag, 128, wmma::mem_row_major);
}

template<typename T,
         typename BucketType,
		 const uint32_t n,
         const uint32_t k,
         const uint32_t l,
         const uint32_t w,
         const uint32_t p,
		 const uint32_t gpu_matrix_width,
         const uint32_t gpu_l_limbpos>
__global__ void KernelDumer(T *__restrict__ pde,                // error
                            const T *__restrict__ pdH,          // parity check matrix
                            const T *__restrict__ piHT,         // only l coordinates of the parity check matrix
                            const T *__restrict__ pds,          // syndrome full length
                            const T s,                          // syndrome
                            const uint16_t *__restrict__ pdC,   //
                            const uint64_t nr_combinations,     //
                            uint32_t *compare)                  //
{
	// Work here

	__shared__ T data[32];

}

/// main function to call to execute the stern/dumer algorithm
template<typename T, const ConfigDumer<T> &config>
void* Dumer_Thread(void *par) {
	// const values
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t k = config.k;
	constexpr static uint32_t w = config.w;
	constexpr static uint32_t l = config.l;
	constexpr static uint32_t p = config.p;

	constexpr static uint32_t width = config.gpu_matrix_width;         //
	constexpr static uint32_t llimbpos = config.llimbpos;   //

	// TODO generalize
	using BucketType            = LogTypeTemplate<MAX(l, 32)>;
	using CombinationsType      = uint16_t;

	// TODO describe
	ParamsDumer *params = (ParamsDumer *)par;

	// CPU stuff
	mzd_t *e            = mzd_init(1, config.n);
	mzd_t *wH           = matrix_init(config.n-config.k, config.n+1);
	mzd_t *wHT          = mzd_init(wH->ncols, wH->nrows);
	mzd_t *H           =  mzd_init(config.n-config.k, config.k+config.l+1);
	mzd_t *HT           = mzd_init(H->ncols, H->nrows);
	mzp_t *permutation  = mzp_init(config.n);

	// TODO swap matrix rows
	mzd_t *tmp = matrix_concat(nullptr, params->A, params->syndromT);
	mzd_copy(wH, tmp);
	mzd_free(tmp);

	// GPU STUFF
	thrust::device_vector<T> de(config.cpu_gpu_width_factor *params->A->width);
	checkCudaErrors(cudaPeekAtLastError());

	// Copy as much as possible on the cuda device. If it is not possible to pack everything on the device
	// it will be chunked. Also, every CPU thread needs its own `dCombination` vector. Otherwise, they could not concurrently work.
	thrust::device_vector<CombinationsType> dCombinations(p*params->tableSize);
	thrust::copy(params->combinations->begin(), params->combinations->begin()+(p*params->tableSize), dCombinations.begin());

	uint32_t *dCompare  = nullptr, Compare = 0;

	checkCudaErrors(cudaMalloc(&dCompare, sizeof(uint32_t)));
	checkCudaErrors(cudaMemcpy(dCompare, &Compare,    sizeof(uint32_t), cudaMemcpyHostToDevice));

	// GPU stuff + Pointer abstraction
	T* pdHT;
	checkCudaErrors(cudaMalloc(&pdHT, config.cpu_gpu_width_factor * config.cpu_matrix_row * config.cpu_matrix_width * sizeof(T)));
	T* pdsT                 = &pdHT[config.cpu_gpu_width_factor * (config.cpu_matrix_row - 1) * config.cpu_matrix_width];
	T* pde                  = thrust::raw_pointer_cast(de.data());
	CombinationsType* pdC   = thrust::raw_pointer_cast(dCombinations.data());

	// timing stuff
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float kernel_runtime = 0., total_kernel_runtime = 0.;

	double time = (double)clock();
	uint64_t loops = 0;
	customMatrixData *matrix_data = init_matrix_data(wH->ncols);

	while (!Compare) {
		loops += 1;

		// create for each stream its own gaus elimination
		matrix_create_random_permutation(wH, wHT, permutation);
		matrix_echelonize_partial_plusfix(wH, config.m4ri_k, n-k-l, matrix_data, 0, n - k - l, 0, permutation);
		mzd_submatrix(H, wH, 0, n-k-l, n-k, n+1);
		matrix_transpose(HT, H);
		// end of gaussian elimination phase
		cudaEventRecord(start);

		// TODO opt memcopy in this functions
		MatrixToCuda<T>(pdHT, HT);
		DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError());)
		DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize());)

		kernel_runtime = 0.;
		for (uint32_t tableRet = 0; tableRet < params->tableReps; tableRet++) {
			// start the actual kernel
			KernelDumer<T, BucketType, n, k, l, w, p, width, llimbpos>
				<<<params->blocks, params->tpb>>>
					(pde, pdHT, pdsT,
					 pdC, pdS,
					 pdT, pdTs, pdTe,
					 dT.begin(), dT.end(), dTs.begin(), dTs.end(), dTe.begin(), dTe.end(),
			         params->tableSize, dCompare);

			DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError());)
			DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize());)

			// fetch new set of permutations.
			if (params->tableReps != 1) {
				if (tableRet != params->tableReps-1) {
					const uint64_t from = p*tableRet*params->tableSize;
					const uint64_t to   = p*(tableRet+1)*params->tableSize;
					std::cout << "mem copy from: " << from << " to " << to << "\n";
					thrust::copy(params->combinations->begin()+from,
					             params->combinations->begin()+to,
					             dCombinations.begin());
				}
			}
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&kernel_runtime, start, stop);
		total_kernel_runtime += kernel_runtime;

		checkCudaErrors(cudaMemcpy(&Compare, dCompare, sizeof(bool), cudaMemcpyDeviceToHost));
		DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize());)

		if (Compare) {
			// apply back perm
			mzd_t *tmpe = mzd_init(1, n);
			CudaToMatrix<T>(tmpe, de);
			for (uint32_t i = 0; i < n; i++) {
				mzd_write_bit(e, 0, permutation->values[i], mzd_read_bit(tmpe, 0, i));
			}

			mzd_print(tmpe);
			mzd_print(e);
			mzd_free(tmpe);
		}

		if ((loops%config.print_loops) == 0) {
			const uint64_t cloops = loops;
			const double time2 = ((double)clock() -  time)/CLOCKS_PER_SEC;
			const double ktime = total_kernel_runtime/1e3;
			std::cout << "\rloops: " << cloops << "(" << double(cloops)/double(config.Loops())*100. << "), loops/s:" << cloops/time2 << ", kernel runtime: " << ktime << "s (" << ktime/time2*100. << "%) , runtime: " << time2 << "s" << std::flush;
		}
	}

	time = ((double)clock() - time)/CLOCKS_PER_SEC;
	total_kernel_runtime /= 1e3;
	std::cout << "runtime: "  << time << "s, loops: " << loops << ", E[loops]: " << config.Loops() << " (" << double(loops)/double(config.Loops())*100. << "%), total kernel runtime: " << total_kernel_runtime << "s (" << total_kernel_runtime/time*100. << "%), loops/kernel s: " << loops/total_kernel_runtime << ", loops/s: " << loops/time << "\n";
	bool ret = checkCorrectness(e, params->A, params->syndromT, w);

	// CPU
	mzd_free(wH);
	mzd_free(wHT);
	mzd_free(HT);
	mzd_free(H);
	mzp_free(permutation);
	mzd_free(e);

	checkCudaErrors(cudaFree(dCompare));
	checkCudaErrors(cudaFree(pdHT));

	return nullptr;
}


/// calculates the mem consumptions in bytes
/// \tparam param	parameter set of the instance
/// \tparam config	additional parameters needed to run the code
/// \param sizelist	size of the list to enumerate. Allows for Stern or Dumer
/// \return	approx bytes the algorithm needs.
template<typename T, const ConfigDumer<T> &config>
uint64_t calc_stern_mem_consumption(const size_t sizelist) {
	// TODO generalize
	using BucketType            = LogTypeTemplate<MAX(config.l, 32)>;
	const uint64_t mem = (sizelist*sizeof(uint16_t)) + (sizelist*sizeof(T)) +   // T
						 (sizelist*config.gpu_matrix_width *sizeof(BucketType)) +           // S
						 ((uint32_t(1) << config.l)*sizeof(BucketType)*2);      // Ts+Te
	return mem;
}

template<typename T, const ConfigDumer<T> &config>
int CudaDumer() {
	srand(time(nullptr));
	random_seed(rand());
	config.print();

	// Try to allocate as much as possible in one thread and pass it to the other threads
	thrust::host_vector<uint16_t> combinations;
	double time = (double)clock();
	auto numb_p_out_of_k_calc = calc_p_combinations<uint16_t>(combinations, config.klpart_size, config.p);
	time = ((double)clock() - time)/CLOCKS_PER_SEC;
	if(config.numb_p_out_of_k != numb_p_out_of_k_calc) {
		std::cout << "ERROR: not correct combinations Dumer\n";
		std::cout << config.numb_p_out_of_k << " vs " << numb_p_out_of_k_calc << "\n";
		return -1;
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0);

	const uint64_t max_mem        = prop.totalGlobalMem;
	const uint64_t max_needed_mem = calc_stern_mem_consumption<T, config>(config.numb_p_out_of_k);
	const uint64_t tableReps      = ceil(double(max_needed_mem)/double(max_mem));
	const uint64_t tableSize      = config.numb_p_out_of_k/tableReps;   // size per repetition => Actually number of elements on the device
	const uint64_t needed_mem     = calc_stern_mem_consumption<T, config>(tableSize);
	assert(tableReps > 0);

	//const uint32_t number_sm    = prop.multiProcessorCount;
	constexpr uint64_t tpb        = MIN(256, MAX(config.k+config.l, config.numb_p_out_of_k));
	const uint64_t blocks         = MAX(1, tableSize/tpb); // take one extra block, s.t. we always have enough threads.
	const uint64_t nr_gpu_threads = blocks * tpb;
	const double rounding         = 1024.*1024.;

	// Some logging
	std::cout << "Precomputation took:  " << time << "s\n";
	std::cout << "max_mem_avail:        " << max_mem << "B = " << max_mem/rounding << "MB\n";
	std::cout << "max_needed_mem:       " << max_needed_mem << "B = " << max_needed_mem/rounding << "MB\n";
	std::cout << "needed mem:           " << needed_mem << "B = " << (needed_mem/rounding) << "MB\n";
	std::cout << "numb_p_out_of_k_calc: " << numb_p_out_of_k_calc << "\n";
	std::cout << "tableSize:            " << tableSize << "\n";
	std::cout << "tableReps:            " << tableReps << "\n";
	std::cout << "starting:             " << config.nr_threads << " threads with each: " << blocks << " blocks with each " << tpb << ", in total: : " << nr_gpu_threads << "\n" << std::flush;

	ParamsDumer params (&combinations,blocks, tpb,tableSize, tableReps, mzd_from_str(n, n-k, h), mzd_from_str(1, n-k, s));
	params.A        = mzd_transpose(nullptr, params.AT);
	params.syndromT = mzd_transpose(nullptr, params.syndrom);

	// start the cpu thread
	Dumer_Thread<T, config>((void *)&params);
	cudaDeviceReset();

	// free everything we allocated
	mzd_free(params.syndrom);
	mzd_free(params.syndromT);
	mzd_free(params.A);
	mzd_free(params.AT);
	return 1;
}
