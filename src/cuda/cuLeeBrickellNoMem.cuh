#ifndef DECODING_CUDALEEBRICKELLNOMEM
#define DECODING_CUDALEEBRICKELLNOMEM

#include "cuhelper.cuh"
#include "cuMisc.cuh"
#include "cuLeeBrickell.cuh"

/// same as `ConfigLeeBrickell` but with a few additional configuration flags
/// \tparam T
template<typename T>
struct ConfigLeeBrickellNoMem : public ConfigLeeBrickell<T> {
private:
	/// Construction of the algorithm
	///        n-k-l
	///  ◄──────       ─────►   ◄───         k       ───────────►
	///
	/// ┌─────────────┬──────┬──────────────────────────────────┐  ┌─────┐  ▲
	/// │                    │                                  │  │     │  │
	/// │     I              │           H1                     │  │     │  │
	/// │                    │                                  │  │ s1  │  │n-k
	/// │                    │                                  │  │     │  ▼
	/// ├                    ┼                                  ┤  ├     ┤
	/// │                    │                                  │  │     │
	/// │                    │                                  │  │    │
	/// └─────────────┴──────┴──────────────────────────────────┘  └─────┘
	///
	/// ┌─────────────┬──────┬──────────────────────────────────┐
	/// │                    │           e2                     │
	/// │     e1             │                                  │
	/// └─────────────┴──────┴──────────────────────────────────┘
	///                                        ▲
	///                                        │
	///                                        │
	///                                        │
	///                                        │
	///                        ┌─────────┐     │
	///                        │         │     │
	///                  t1 -> │ f_1(tid)│ ◄───┴
	///                        │         │
	///                  t2 -> │         │
	///                        │         │
	///                  t3 -> │         │
	///                        │         │
	///                  tx -> │ f_1(tid)│
	///                        │         │
	///                        └─────────┘
	// disable the standard constructor
	constexpr ConfigLeeBrickellNoMem() : ConfigLeeBrickell<T>(0, 0, 0, 0) {}

public:
	// NOTE: Only needed if p == 2
	// logarithmic size of `k`, e.g. the number of bits needed to represent `k`.
	// So in the case of p == 2 we need 2*logk bits to select two uniform independent
	// bit positions from the thread id.
	const uint32_t logk = constexpr_log2(this->k);

	// NOTE: Only needed if p ==1 || p == 3.
	// total number of bits needed to extract `p` bit positions (column selection) form the thread index
	const uint32_t f = this->p * logk;

	// constructor
	constexpr ConfigLeeBrickellNoMem(const uint32_t n,// code length
	                                 const uint32_t k,// code dimension
	                                 const uint32_t w,// code weight
	                                 const uint32_t p,// number of bit positions (columns selection) the hash function needs to output.
	                                 const uint32_t max_element = 2000000) : ConfigLeeBrickell<T>(n, k, w, p, max_element) {}

	// return the number of loops needed in expectation
	constexpr uint64_t Loops() const {
		return bc(ConfigLeeBrickell<T>::n, ConfigLeeBrickell<T>::w) /
		       (bc(ConfigLeeBrickell<T>::k, ConfigLeeBrickell<T>::p) *
		        bc(ConfigLeeBrickell<T>::n-ConfigLeeBrickell<T>::k, ConfigLeeBrickell<T>::w-ConfigLeeBrickell<T>::p));
	}

	/// print information about the configuration
	void print() const {
		ConfigLeeBrickell<T>::print();
		std::cout << "logk: " << logk
		          << "f: " << f << std::flush;
	}
};

/// actual hashing function. Depending on p a different weight function is choosen.
/// \tparam T		limb type. Most likely `uint32_t`
/// \tparam k		code dimension
/// \tparam p		number of bit positions (or column selection) this function is going to output
/// \tparam logk	(only needed if p==2), logarithmic size of `k`
/// \tparam f
/// \param set		output:
/// \param value	input: thread index we gonna use as the
template<typename T, const uint32_t k, const uint32_t p, const uint32_t logk, const uint32_t f>
__device__ __forceinline__ void hash(uint16_t set[p], const T value) {
	if constexpr(p==1) {
		// easy case. Simply interpret the input `value`, which is just the thread id,
		// as the bit position (or the column selection). Just make sure everything
		// is in the correct boundaries.
		constexpr T mask1 = ((T(1) << f) - T(1));

		// NOTE: Point of optimisation: somehow remove the modulus operation.
		set[0] = (value & mask1) % k;
	} else if constexpr (p==2) {
		// old and slower code.
		//constexpr uint32_t fprime   = f/2;
		//constexpr T mask1 = (T(1) << (fprime - 1)) - 1;
		////constexpr T mod = k;
		//constexpr T off = k - (1 << (fprime - 1));

		//set[0] = (value & mask1);
		//set[1] = ((value >> (f-fprime))&mask1) + off;
		//const bool t = set[0] == set[1];
		//set[1] += t;

		// this code is faster
		constexpr T mod = k;

		set[1] = ( value         ) % mod;
		set[0] = ((value >> logk)) % mod;

		// check if the two values are the same without using any `if` clause.
		// This is needed because it can happen that we choose two times the same
		// bit position, which is an invalid output. We must choose/calculate two different
		// positions.
		const bool t = set[0] == set[1];
		set[1] += t;
	} else if constexpr(p==3){
		constexpr uint32_t ws = logk;
		// constexpr T mask1 = (T(1) << ws) - 1;
		// constexpr T mask2 = (T(1) << (2*ws)) - 1;

		constexpr T mod = k;
		set[2] = ( value          ) % mod;
		set[1] = ((value >> ws   )) % mod;
		set[0] = ( value >> (f-ws)) % mod;

		const bool t = set[0] == set[1];
		set[1] += t;

		const bool u = set[1] == set[2];
		set[2] += u;
	} else if constexpr(p==4){
		constexpr uint32_t ws = logk;
		// constexpr T mask1 = (T(1) << ws) - 1;
		// constexpr T mask2 = (T(1) << (2*ws)) - 1;

		constexpr T mod = k;
		set[0] = ( value         ) % mod;
		set[1] = ((value >>   ws)) % mod;
		set[2] = ((value >> 2*ws)) % mod;
		set[3] = ( value >>(f-ws)) % mod;

		const bool t = set[0] == set[1];
		set[1] += t;

		const bool u = set[1] == set[2];
		set[2] += u;

		const bool v = set[2] == set[3];
		set[3] += v;
	}
}

/// IMPORTANT: must be called with at least k threads per block.
/// \tparam T			limp type
/// \tparam k			code dimension, needed for the hash function
/// \tparam w			code weight, needed for the hash function
/// \tparam p			number of bit positions (or column selection) to map on
/// \tparam logk		helper for the hash function, see `ConfigLeeBrickellNoMem` for explanation
/// \tparam f			same as `logk`
/// \tparam width		number of limbs needed to store one row in `dHH` or `dss`
/// \param de			output: error vector
/// \param dH			input: parity check matrix
/// \param ds			input: syndrome
/// \param nr_combinations
/// \param compare		output: set to 1 if the solution was found, zero otherwise.
template<typename T, const uint32_t k, const uint32_t w, const uint32_t p, const uint32_t logk, const uint32_t f,const uint32_t width>
__global__ void lee_brickellnomem_step(T  *__restrict__ de,  // error vector
                                  const T *__restrict__ dHH, // parity check matrix
                                  const T *__restrict__ dss, // syndrome
                                  const uint32_t nr_combinations, // total number of combinations we try // todo move to template
                                  unsigned int *__restrict__ compare) {
	constexpr uint32_t T_bitwidth = sizeof(T) * 8;
	const uint32_t block_number =  blockDim.x * gridDim.x;
	uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	// security checks
	assert(blockDim.x >= k);

	// copy the parity check matrix and the syndrome into the shared mem.
	__shared__ T ds[width], dH[width*k];
	if (threadIdx.x < k) {
		// NOTE: this is the smartest I could think of to make sure, that the
		// parity check matrix is really copied into the shared memory.
		// Maybe there are more clever ways.
		for (uint32_t i = 0; i < width; i++) {
			dH[threadIdx.x + threadIdx.x*i] = dHH[threadIdx.x + threadIdx.x*i];
		}
	}

	if (threadIdx.x < width) {
		ds[threadIdx.x] = dss[threadIdx.x];
	}
	__syncthreads();

	uint16_t dC[p] = {0};
	while(index < nr_combinations) {
		// TODO maybe flavor the run?
		hash<T, k, p, logk, f>(dC, index);
		uint32_t weight = 0;
		T x;

		// sum up the columns (selected by the hash function) and the syndrome.
		// NOTE: that we are not fully calculating the sum of the columns (or rows in HT), but
		// rather only the weight
		#pragma unroll
		for (uint32_t j = 0; j < width; j++) {
			x = ds[j];

			#pragma unroll
			for (uint32_t i = 0; i < p; i++) {
				x ^= dH[dC[i]*width + j];
			}

			weight += __popc(x);
		}

		constexpr uint32_t weight_threshold = w-p;

		// check if we found the solution
		if(unlikely(weight <= weight_threshold)) {
			// solution found
			if(0 != atomicCAS(compare, 0, 1))
				return;

			// TODO remove from this function and rebuild everything on cpu
			for (uint16_t j = 0; j < width; j++) {
				de[j] = ds[j];
				for (uint16_t i = 0; i < p; i++) {
					de[j] ^= dH[dC[i]*width + j];
				}
			}

			for (uint16_t i = 0; i < p; i++) {
				const uint32_t pos1   = ((n - k) + dC[i]) / T_bitwidth;
				const uint32_t shift1 = ((n - k) + dC[i]) % T_bitwidth;
				de[pos1] ^= (T(1) << shift1);
			}
		}

		index += block_number;
	}
}

/// base function to call to start the algorithm
/// \tparam T
/// \tparam config
template<typename T, const ConfigLeeBrickellNoMem<T> &config>
bool CudaLeeBrickellNoMem() {
	// Init the rngs.
	srand(time(nullptr));
	random_seed(rand());

	// print information about the current configuratoion
	config.print();

	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0);

	// CPU stuff
	mzd_t *AT           = mzd_from_str(config.n, config.n-config.k, h);
	mzd_t *A            = mzd_transpose(NULL, AT);
	mzd_t *syndrom      = mzd_from_str(1, config.n-config.k, s);
	mzd_t *syndromT     = mzd_transpose(NULL, syndrom);
	mzd_t *syndrom_tmp  = mzd_init(syndrom->ncols, syndrom->nrows);
	mzd_t *syndromT_tmp = mzd_init(syndrom->nrows, syndrom->ncols);
	mzd_t *e            = mzd_init(1, config.n);
	mzd_t *eT           = mzd_init(config.n, 1);
	mzd_t *wH [config.nr_streams];
	mzd_t *wHT[config.nr_streams];
	mzd_t  *H [config.nr_streams];
	mzd_t  *HT[config.nr_streams];
	mzp_t *permutation[config.nr_streams];

	for (uint32_t i = 0; i < config.nr_streams; i++) {
		wH[i]   = mzd_init(config.n-config.k, config.n+1);
		wHT[i]  = mzd_init(config.n+1, config.n-config.k);
		H[i]    = mzd_init(config.n-config.k, config.k+1);
		HT[i]   = mzd_init(H[i]->ncols, H[i]->nrows);
		permutation[i] = mzp_init(config.n);
		mzd_concat(wH[i], A, syndromT);

		// register memory
		cudaHostRegister(HT[i]->rows[0], HT[0]->width * HT[0]->nrows * sizeof(uint64_t), 0);
	}

	// timing stuff
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float kernel_runtime = 0., total_kernel_runtime = 0.;

	uint32_t Compare = 0, *dCompare  = nullptr;
	checkCudaErrors(cudaMalloc(&dCompare, sizeof(uint32_t)));
	checkCudaErrors(cudaMemcpy(dCompare, &Compare, sizeof(uint32_t), cudaMemcpyHostToDevice));

	// allocate the error vector on the gpu
	thrust::device_vector<T> de(config.cpu_gpu_width_factor*A->width);
	T* pde = thrust::raw_pointer_cast(de.data());

	// GPU stuff + Pointer abstraction
	cudaStream_t stream[config.nr_streams];
	T* pdHT[config.nr_streams];
	T* pdsT[config.nr_streams];

	// Create a certain number of stream, which are working independently on different permutations of
	// the parity check matrix.
	for(uint32_t i = 0; i < config.nr_streams; i++) {
		cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
		checkCudaErrors(cudaMalloc(&pdHT[i], config.cpu_gpu_width_factor * config.cpu_matrix_rows * config.cpu_matrix_width * sizeof(T)));
		// let the syndrome pointer simply point to the last row (column in the not transposed matrix H) of HT.
		pdsT[i] = &pdHT[i][config.cpu_gpu_width_factor * (config.cpu_matrix_rows - 1) * config.cpu_matrix_width];
	}

	double time = (double)clock();

	// allocate as many threads and blocks needed st. tpb*block >= (k over p)
	const uint64_t tpb      = MIN(k, config.numb_p_out_of_k);
	const uint64_t blocks   = MAX(1, 1 + config.numb_p_out_of_k / tpb);
	static_assert(tpb >= config.k, "implementation limitation");
	uint64_t loops = 0;
	std::cout << "tpb: " << tpb << ", blocks " << blocks << ", total: " << tpb*blocks << "\n";

	// This data is needed to perform a fast gaussian elimination
	// TODO allocate for every thread such a structure.
	customMatrixData *matrix_data = init_matrix_data(wH[0]->ncols);

	// start of the actual algorithm
	// #pragma omp parallel num_threads(3)
	while (Compare == 0) {
		for (uint32_t cs = 0; cs < config.nr_streams; cs++) {

			// create for each stream its own gaus elimination
			matrix_create_random_permutation(wH[cs], wHT[cs], permutation[cs]);
			matrix_echelonize_partial_plusfix(wH[cs], config.m4ri_k, config.n-config.k, matrix_data, 0, config.n-config.k, 0, permutation[cs]);
			mzd_submatrix(H[cs], wH[cs], 0, config.n-config.k, config.n-config.k, config.n + 1);
			matrix_transpose(HT[cs], H[cs]);
			// end of the gaussian elimination phase

			// cudaEventRecord(start);
			MatrixToCuda<T>(pdHT[cs], HT[cs], &stream[cs]);
			DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError());)
			DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize());)

			kernel_runtime = 0.;
			cudaEventRecord(start);

			// start the actual kernel.
			lee_brickellnomem_step
			    <T, config.k, config.w, config.p, config.logk, config.f, config.gpu_matrix_width>
				<<<blocks, tpb, 0, stream[cs]>>>
						(pde, pdHT[cs], pdsT[cs], config.numb_p_out_of_k, dCompare);
			DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError());)
			DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize());)

			// calculate the kernel time.
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&kernel_runtime, start, stop);
			total_kernel_runtime += kernel_runtime;
		}

		for (uint32_t cs = 0; cs < config.nr_streams; cs++) {
			checkCudaErrors(cudaMemcpy(&Compare, dCompare, sizeof(bool), cudaMemcpyDeviceToHost));
			DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError());)
			DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize());)

			if (Compare) {
				// apply back perm
				mzd_t *tmpe = mzd_init(1, n);
				CudaToMatrix<T>(tmpe, de);

				for (uint32_t i = 0; i < n; i++) {
					mzd_write_bit(e, 0, permutation[cs]->values[i], mzd_read_bit(tmpe, 0, i));
				}
				mzd_free(tmpe);
				break;
			}
		}

		// print every x lines a status update.
		if (loops%config.print_loops == 0) {
			const double time2 = ((double)clock() - time)/CLOCKS_PER_SEC;
			const double ktime = total_kernel_runtime/1e3;
			std::cout << "\rloops: " <<  loops << "(" << double(loops)/double(config.Loops()) * 100. << "%), loops/s:" << loops/time2 << ", kernel runtime: " << ktime << "s (" << ktime/time2*100. << "%) , runtime: " << time2 << "s\n" << std::flush;
		}

		loops += 1;
	}

	time = ((double)clock() - time)/CLOCKS_PER_SEC;
	total_kernel_runtime /= 1e3;
	std::cout << "\nruntime: "  << time << "s, loops: " << loops << ", E[loops]: " << config.Loops() << " (" << double(loops)/double(config.Loops())*100. << "%), total kernel runtime: " << total_kernel_runtime << "s (" << total_kernel_runtime/time*100. << "%), loops/kernel s: " << loops/total_kernel_runtime << ", loops/s: " << loops/time << "\n";
	bool ret = checkCorrectness(e, A, syndromT, w);

	// free the gpu memory
	for(uint32_t i = 0; i < config.nr_streams; i++) {
		// GPU
		checkCudaErrors(cudaFree(pdHT[i]));
		cudaStreamDestroy(stream[i]);
		cudaHostUnregister(HT[i]->rows[0]);

		// CPU
		mzd_free(wH[i]);
		mzd_free(wHT[i]);
		mzd_free(HT[i]);
		mzp_free(permutation[i]);
	}

	checkCudaErrors(cudaFree(dCompare));

	// free the cpu memory
	mzd_free(A);
	mzd_free(AT);
	mzd_free(syndrom_tmp);
	mzd_free(syndromT_tmp);
	mzd_free(syndrom);
	mzd_free(syndromT);
	mzd_free(e);
	mzd_free(eT);

	return ret;
}

#endif // DECODING_CUDALEEBRICKELLNOMEM
