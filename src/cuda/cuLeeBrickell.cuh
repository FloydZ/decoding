#ifndef DECODING_CUDALEEBRICKELL_H
#define DECODING_CUDALEEBRICKELL_H

#include "cuhelper.cuh"
#include "cuMisc.cuh"

template<typename T>
struct ConfigLeeBrickell {
private:
	// disable the standard constructor
	constexpr ConfigLeeBrickell() : n(0), k(0), w(w), p(p) {}

public:
	// instance parameters
	const uint32_t n, k, w, p;

	// this is the multiplicative factor between one cpu limb (uint64_t)
	// and a gpu limb (probably uint32_t).
	const uint32_t cpu_gpu_width_factor      = sizeof(uint64_t) / sizeof(T);
	
	// rows of the cpu matrix HT
	const uint32_t cpu_matrix_rows  = k+1; // +1 for the syndrome
	const uint32_t cpu_matrix_width = (n-k+63)/64;

	// number of limbs in the gpu matrix HT;
	const uint32_t gpu_matrix_width = cpu_gpu_width_factor * cpu_matrix_width;

	// Table stuff
	// number of coordinates to enumerate
	const uint32_t kl_size        	= k;
	const uint32_t numb_p_out_of_k 	= bc(kl_size, p);

	// maximum number to enumerate at once on the gpu.
	// Note: this is actually a hardware constant, limited by the amount of 
	// memory available, then a parameter. 
	// But I cannot read this value during compile time, so you have
	// to find it yourself. 
	const uint32_t max_elements_in_list_per_stream = 2000000;

	// limit the number of elements in one strem
	const uint32_t tableSize = std::min(max_elements_in_list_per_stream, numb_p_out_of_k);

	// Number of repeated calls to the gpu kernel in one iteration of the 
	// algorithm. This is needed, because maybe not everything fits into the
	// gpu mem at once.
	const uint32_t tableReps = numb_p_out_of_k/tableSize;

	// Calculate the optimal `r` value for M4RI
	const uint32_t m4ri_k = matrix_opt_k(n - k, MATRIX_AVX_PADDING(n));

	// simply hardcoded for now.
	const uint32_t nr_streams = 1;
	const uint32_t nr_threads = 1;

	// if this value is set to a different value then -1 the program will end the
	// computation after `loops` permutation, regardless if it found the solution or not
	const uint64_t loops        = uint64_t(-1);

	// print every `print_loops` a status update of the program
	const uint64_t print_loops  = 1000;

	// check every `exit_loops` if another thread, which is working on a different permutation
	// found the solution. If so exit the computation.
	// NOTE: the check if something is found includes atomic operations, which are quite expensive
	//  so dont check every loop if something is found.
	const uint64_t exit_loops   = 1000;

	/// base constructor
	constexpr ConfigLeeBrickell(const uint32_t n, 
								const uint32_t k,
	                        	const uint32_t w,
								const uint32_t p,
								const uint32_t max_element=100000):
			n(n), k(k), w(w), p(p),
			max_elements_in_list_per_stream(max_element){};

	constexpr ConfigLeeBrickell(ConfigLeeBrickell &c) = default;

	// print some information about the configiration
	void print() const {
		std::cout << "n: " << n
		          << ", k: " << k
		          << ", n-k:" << n-k
		          << ", p: " << p
				  << ", cpu_matrix_rows: " << cpu_matrix_rows
		          << ", cpu_matrix_width: " << cpu_matrix_width
				  << ", gpu_matrix_width: " << gpu_matrix_width
		          << ", cpu_gpu_width_factor: " << cpu_gpu_width_factor
				  << ", kl_size: " << kl_size
				  << ", numb_p_out_of_k: " << numb_p_out_of_k
				  << ", max_elements_in_list_per_stream " << max_elements_in_list_per_stream
				  << ", tableSize: " << tableSize
				  << ", tableReps: " << tableReps
				  << ", nr_streams: " << nr_streams
				  << ", nr_threads: " << nr_threads
		          << "\n";
	}

	/// \return the number of loops needed in expected
	constexpr uint64_t Loops() {
		return bc(n, w) / (bc(n - k, w - p) * bc(k, p));
	}

	/// print the final information about the overall runtime of the program
	/// \param loops	total number of loops the program needed
	/// \param time 	cpu time the program needed for the loops
	/// \param total_kernel_runtime gpu time measured with
	void print_loops_fkt(const uint64_t loops, const double time, const float total_kernel_runtime) {
		const double expteded_time = Loops()/loops*time;

		std::cout << "runtime: "  << time << "s"
		          << ", loops: " << loops << "(" << Loops() <<")=" << loops / Loops() << "x"
		          << ", total kernel runtime: " << total_kernel_runtime<< "s (" << total_kernel_runtime/time*100. << "%)"
		          << ", loops/kernel s: " << loops/total_kernel_runtime
		          << ", loops/s: "<< loops/time
		          << ", E[RT]=" << expteded_time << "s(" << time/expteded_time << "x)"
		          << "\n";
	}
};

/// Important Notes:
///		this function should be called with only 1 block and up to `size_of_combinations` many thready per block.
/// \tparam width
/// \param de 				Output error vector
/// \param dH 				Input working submatrix of size n-k x k
/// \param ds 				Input working syndrome
/// \param dC 				Input combinations of vectors of weight p
/// \param wide				Number of limbs in s. Keep attention if you are working with uint64 or uint32:t
/// \param size_of_combinations
/// \param found			true if solution is found.
/// \param compare
template<typename T,
         const uint32_t n, const uint32_t k, const uint32_t w, const uint32_t p,
         const uint32_t width=4>
__global__ void lee_brickell_step(T *__restrict__ de,
                                  T *__restrict__ dH,
                                  T *__restrict__ dss,
                                  uint32_t *__restrict__ dC,
								  const uint32_t wide, const uint32_t nr_combinations,
								  bool* found, unsigned int *compare) {
	uint32_t thread_number =  blockDim.x * gridDim.x;
	uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ T ds[width];

	if (threadIdx.x < width)
		ds[threadIdx.x] = dss[threadIdx.x];

	__syncthreads();

	while(index < nr_combinations) {
		// TODO not correct for p != 1
		uint32_t weight = add_weight_row<T>(ds, dH + (dC[p*index] * wide), wide, p, w);

		if(weight <= (w-p)) {
			if( 0 != atomicCAS(compare,0,1))
				return;
			recalc_e<T, uint32_t>(de, dH, dss, dC, wide, index, n, k, p);
			*found = true;
		}

		if (*found)
			return;

		index += thread_number;
	}
}

template<typename T, const ConfigLeeBrickell<T> &config>
void CudaLeeBrickell() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0);

	thrust::host_vector<uint32_t>   combinations;
	thrust::device_vector<uint32_t> dCombinations(config.tableSize);
	if(config.numb_p_out_of_k != calc_p_combinations<uint32_t>(combinations, config.kl_size, config.p)) {
		std::cout << "ERROR: not correct combinations LeeBrickell\n";
		return;
	}
	thrust::copy(combinations.begin(), combinations.begin()+config.tableSize, dCombinations.begin());
	//dCombinations = combinations;

	// CPU stuff
	mzd_t *AT           = mzd_from_str(n, config.n-config.k, h);
	mzd_t *A            = mzd_transpose(NULL, AT);
	mzd_t *syndrom      = mzd_from_str(1, config.n-config.k, s);
	mzd_t *syndromT     = mzd_transpose(NULL, syndrom);
	mzd_t *syndrom_tmp  = mzd_init(syndrom->ncols, syndrom->nrows);
	mzd_t *syndromT_tmp = mzd_init(syndrom->nrows, syndrom->ncols);
	mzd_t *e            = mzd_init(1, config.n);
	mzd_t *eT           = mzd_init(config.n, 1);
	mzd_t *wH [config.nr_streams];
	mzd_t *wHT[config.nr_streams];
	// mzd_t  *H [nr_streams];
	mzd_t  *HT[config.nr_streams];
	mzp_t *permutation[config.nr_streams];

	for (uint32_t i = 0; i < config.nr_streams; i++) {
		wH[i]   = mzd_init(config.n-config.k, n+1);
		wHT[i]  = mzd_init(config.n+1, config.n-config.k);
		// H[i]    = mzd_init_window(wH[i], 0, n-k, n-k, n+1);
		HT[i]   = mzd_init_window(wHT[i], config.n-config.k, 0, config.n+1, config.n-config.k);
		permutation[i] = mzp_init(n);
		mzd_concat(wH[i], A, syndromT);

		// register memory
		cudaHostRegister(HT[i]->rows[0], HT[0]->width * HT[0]->nrows * sizeof(uint64_t), 0);
	}


	// constants
	constexpr uint32_t M4RI_factor  =  sizeof(uint64_t) / sizeof(T);
	const uint32_t matrix_width     =  HT[0]->width;    // internal number of rows of the submatrix
	const uint32_t matrix_rows      =  HT[0]->nrows;
	const uint32_t width            =  M4RI_factor*matrix_width;

	// timing stuff
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float kernel_runtime = 0., total_kernel_runtime = 0.;


	bool found_e        = false, set_e = false, *dFound_e = nullptr;
	uint32_t *dCompare  = nullptr;
	uint32_t zero       = 0;

	checkCudaErrors(cudaMalloc(&dCompare, sizeof(uint32_t)));
	checkCudaErrors(cudaMalloc(&dFound_e, sizeof(bool)));

	checkCudaErrors(cudaMemcpy(dCompare, &zero,    sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dFound_e, &found_e, sizeof(bool),     cudaMemcpyHostToDevice));

	thrust::device_vector<T> de(M4RI_factor*A->width);

	// GPU stuff + Pointer abstraction
	cudaStream_t stream[config.nr_streams];
	T* pdHT[config.nr_streams];
	T* pdsT[config.nr_streams];

	for(uint32_t i = 0; i < config.nr_streams; i++) {
		cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
		checkCudaErrors(cudaMalloc(&pdHT[i], M4RI_factor * matrix_rows * matrix_width * sizeof(T)));
		pdsT[i] = &pdHT[i][M4RI_factor * (matrix_rows - 1) * matrix_width];
	}

	T* pde         = thrust::raw_pointer_cast(de.data());
	uint32_t * pdC = thrust::raw_pointer_cast(dCombinations.data());


	double time = (double)clock();

	dim3 ThreadsPerBlock(config.numb_p_out_of_k, width);
	dim3 BlocksPerGrid(1, width);
	//dim3 LeeBrickell(numb_p_out_of_k);

	uint64_t loops = 0;

	// #pragma omp parallel num_threads(3)
	while (!found_e) {
		// Iterate over all streams
		// we have to iterate over the streams in the inner loop, otherwise we wouldnt be able to use this mutlithreaded.
		// #pragma omp for
		for (uint32_t current_stream = 0; current_stream < config.nr_streams; current_stream++) {
			loops += 1;

			// create for each stream its own gaus elimination
			matrix_create_random_permutation(wH[current_stream], wHT[current_stream], permutation[current_stream]);
			mzd_echelonize_m4ri(wH[current_stream], 1, 4);
			mzd_transpose(wHT[current_stream], wH[current_stream]);

			cudaEventRecord(start);
			MatrixToCuda<T>(pdHT[current_stream], HT[current_stream], &(stream[current_stream]));

			kernel_runtime = 0.;

			for (uint32_t tableRet = 0; (tableRet < config.tableReps) && !found_e; tableRet++) {
				lee_brickell_step
				        <T, config.n, config.k, config.w, config.p, config.gpu_matrix_width>
				        <<<1, config.numb_p_out_of_k, 0, stream[current_stream]>>>(
				        pde,
				        pdHT[current_stream],
				        pdsT[current_stream],
				        pdC,
				        width,
				        config.numb_p_out_of_k,
				        dFound_e,
				        dCompare);

				// fetch new set of permutations.
				if constexpr(config.tableReps != 1) {
					if (tableRet != config.tableReps-1) {
						// std::cout << "from: " << tableRet*tableSize << " to " << ((tableRet+1)*tableSize) << "\n";
						thrust::copy(combinations.begin()+(tableRet*config.tableSize),
									 combinations.begin()+((tableRet+1)*config.tableSize),
									 dCombinations.begin());
					}
				}
			}

			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&kernel_runtime, start, stop);
			total_kernel_runtime += kernel_runtime;

//			checkCudaErrors(cudaStreamSynchronize(stream[current_stream]));
//          checkCudaErrors(cudaDeviceSynchronize());
//			checkCudaErrors(cudaMemcpyAsync(&found_e, dFound_e, sizeof(bool), cudaMemcpyDeviceToHost, stream[current_stream]));
			checkCudaErrors(cudaMemcpy(&found_e, dFound_e, sizeof(bool), cudaMemcpyDeviceToHost));

			if (found_e && !set_e) {
				// this must be atomic in a multithreaded environment
				set_e = true;

				// apply back perm
				mzd_t *tmpe = mzd_init(1, n);
				CudaToMatrix<T>(tmpe, de);

				for (uint32_t i = 0; i < n; i++) {
					mzd_write_bit(e, 0, permutation[current_stream]->values[i], mzd_read_bit(tmpe, 0, i));
				}


				mzd_print(tmpe);
				mzd_print(e);
				std::cout << "stream " << current_stream << "found it\n";
				mzd_free(tmpe);
			}
		}
	}

	time = ((double)clock() - time)/CLOCKS_PER_SEC;
	total_kernel_runtime /= 1e3;
	std::cout << "runtime: "  << time << "s, loops: " << loops << ", total kernel runtime: " << total_kernel_runtime << "s (" << total_kernel_runtime/time*100. << "%), loops/kernel s: " << loops/total_kernel_runtime << ", loops/s: " << loops/time << "\n";

	// check correctness
	mzd_transpose(eT, e);
	mzd_mul_naive(syndrom_tmp, A, eT);
	mzd_transpose(syndromT_tmp, syndrom_tmp);

	std::cout << "s is:\t\t";
	mzd_print(syndrom);
	std::cout << "s should:\t";
	mzd_print(syndromT_tmp);

	for (int i = 0; i < syndrom->ncols; ++i) {
		if(mzd_read_bit(syndrom, 0, i) != mzd_read_bit(syndrom_tmp, i, 0)) {
			std::cout << "error not correct\n";
			break;
		}
	}


	for(uint32_t i = 0; i < config.nr_streams; i++) {
		// GPU
		checkCudaErrors(cudaFree(pdHT[i]));
		cudaStreamDestroy(stream[i]);
		cudaHostUnregister(HT[i]->rows[0]);

		// CPU
		mzd_free(wH[i]);
		mzd_free(wHT[i]);
		//mzd_free(H[i]);
		mzd_free(HT[i]);
		mzp_free(permutation[i]);
	}

	checkCudaErrors(cudaFree(dCompare));
	checkCudaErrors(cudaFree(dFound_e));


	mzd_free(A);
	mzd_free(AT);
	mzd_free(syndrom_tmp);
	mzd_free(syndromT_tmp);
	mzd_free(syndrom);
	mzd_free(syndromT);
	mzd_free(e);
	mzd_free(eT);
}

#endif // DECODING_CUDALEEBRICKELL_H
