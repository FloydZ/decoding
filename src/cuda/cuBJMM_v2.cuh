#pragma once

#include "cuhelper.cuh"
#include "cuMisc.cuh"
#include "cuStern.cuh"

#include <stdint.h>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cuco/static_map.cuh>

template<const uint32_t p>
struct ChangeListElement {
	uint32_t data[p];

	__host__
	ChangeListElement() : data() {}

	__host__
	ChangeListElement(uint32_t *ddata) : data() {
		for (size_t i = 0; i < p; i++) {
			data[i] = ddata[i];
		}
	}

	/// \param i
	/// \return
	__host__ __device__
	inline uint32_t & operator[](const size_t i) { ASSERT(i < p); return data[i]; }

	__host__ __device__
	const inline uint32_t & operator[](const size_t i) const { ASSERT(i < p); return data[i]; }

	__host__
	static void Transform(thrust::host_vector<ChangeListElement<p>> &out,
	                      thrust::host_vector<uint32_t> &in) {
		Transform(out, in.data(),in.size());
	}

	__host__
	static void Transform(thrust::host_vector<ChangeListElement<p>> &out,
	                      uint32_t *in2, const size_t size) {
		for (size_t i = 0; i < size; i++) {
			out.push_back(ChangeListElement<p>(&in2[i*p]));
		}
	}
};

template<const uint32_t p>
std::ostream &operator<<(std::ostream &out, const ChangeListElement<p> &obj) {
	for (uint32_t i = 0; i < p; i++) {
		out << obj.data[i] << " ";
	}

	return out;
}

template<typename T, const uint32_t limbs>
struct Columns {
	uint32_t __data[limbs];

	__host__ __device__
	Columns() : __data() {}

	__host__ __device__
	Columns(uint32_t ddata[limbs]) : __data() {
		for (uint16_t i = 0; i < limbs; i++) {
			__data[i] = ddata[i];
		}
	}

	/// \param i
	/// \return
	__host__ __device__
	inline T& operator[](const size_t i) { ASSERT(i < p); return __data[i]; }

	__host__ __device__
	const inline T& operator[](const size_t i) const { ASSERT(i < p); return __data[i]; }

	__host__
	static void Transform(thrust::device_vector<Columns<T, limbs>> &out,
	                      thrust::device_vector<T> &in) {
		for (size_t i = 0; i < in.size(); i++) {
			out.push_back(Columns<T, limbs>(&in[i*limbs]));
		}
	}


	__host__
	static void MatrixToCuda(thrust::device_vector<Columns<T, limbs>> &dOut,
	                  const mzd_t *in) {
		const size_t wide  = in->width;
		const size_t nrows = in->nrows;
		thrust::host_vector<Columns<T, limbs>> tmp(nrows);
		constexpr uint32_t factor = sizeof(uint64_t)/ sizeof(uint32_t);

		for (uint64_t i = 0; i < nrows; i++) {
			for (uint64_t j = 0; j < wide; j++) {
				if constexpr (factor == 0) {
					tmp[i][j] = in->rows[i][j];
				} else {
					constexpr uint32_t bits = sizeof(T)*8;
					constexpr uint64_t mask = (uint64_t(1u) << bits) - uint64_t(1);
					for (uint32_t f = 0; f < factor; f++) {
						tmp[i][j + f] = (in->rows[i][j] >> f*bits) & mask;
					}
				}
			}
		}

		dOut = tmp;
	}

	__host__ __device__
	T* data() { return (T *)__data; }

	__host__ __device__
	T* data() const { return (T *)__data; }

	__host__ __device__
	void print_binary() {

	}
};

template<typename T, const uint32_t limbs>
__global__ void print_matrix(const Columns<T,limbs> *d, const uint32_t rows, const uint32_t width) {
	for (uint32_t i = 0; i < rows; i++) {
		//const T *ptr = thrust::raw_pointer_cast(d->data());
		print_bit_vector(d[i].data(), width);
	}
	printf("\n");
}

template<typename T, const uint32_t gpu_width>
std::ostream &operator<<(std::ostream &out,
                         const Columns<T, gpu_width> &obj) {
	for (uint32_t i = 0; i < gpu_width; i++) {
		out << obj.__data[i] << " ";
	}

	return out;
}

template<typename T,                // limb type
         const uint32_t gpu_width,  // number of limbs
         const uint32_t p           // weight to enumerate
         >
struct FillList {
	// parity check matrix
	Columns<T, gpu_width> *dHT = nullptr;

	// Constructor
	FillList() : dHT() {}
	FillList(Columns<T, gpu_width> *dHT) : dHT(dHT) {}
	//FillList(thrust::device_vector<Columns<T, gpu_width>> *dHT) : dHT(dHT) {}

	__host__ __device__
	Columns<T, gpu_width>& operator()(const ChangeListElement<p> &dC,
	                   Columns<T, gpu_width> &in) const {
		#pragma unroll
		for(uint32_t i = 0; i < gpu_width; i++) {
			in[i] = 0;//s[i];

			#pragma unroll
			for (uint32_t j = 0; j < p; j++) {
				const uint16_t c = dC[j];
				in[i] ^= dHT[c][i];
			}
		}

		return in;
	}
};

template<typename T, const uint32_t gpu_width, const uint32_t p>
std::ostream &operator<<(std::ostream &out,
                         const FillList<T, gpu_width, p> &obj) {
	// TODO
	return out;
}

template<typename T, const uint32_t l>
struct custom_key_type {
	constexpr static uint32_t bits = sizeof(T)*8;

	// Key type larger than 8B only supported for sm_70 and up
	// must be enough bits to hold l bits
	T data;

	__host__ __device__ custom_key_type() {
		static_assert(bits > l);
	}
	__host__ __device__ custom_key_type(T x) : data{x} {
		static_assert(bits > l);
	}

	// Device equality operator is mandatory due to libcudacxx bug:
	// https://github.com/NVIDIA/libcudacxx/issues/223
	__device__ bool operator==(custom_key_type const& other) const {
		return data == other.data;
	}
};

/// custom value type for the hashmaps.
///	currently its the index to the baselist.
/// \tparam T data type of the index
template<typename T, const uint32_t limbs>
struct custom_value_type {
	T index[limbs];
	__host__ __device__ custom_value_type() {}
	__host__ __device__ custom_value_type(T x) : index() { index[0] = x; }
	__host__ __device__ custom_value_type(T x[limbs]) : index() {
		for (uint32_t i = 0; i <limbs; i++) {
			index[i] = x[i];
		}
	}
};

// User-defined device hash callable
template<typename T, const uint32_t l>
struct custom_hash {
	constexpr static uint32_t bits = sizeof(T)*8;
	constexpr static T mask = (T(1) << bits) - T(1);

	template <typename key_type>
	__device__ uint32_t operator() (key_type k) {
		static_assert(bits > l);
		return k.data & mask;
	};
};

// User-defined device key equal callable
struct custom_key_equals {
	template <typename key_type>
	__device__ bool operator()(key_type const& lhs, key_type const& rhs) {
		return lhs.data == rhs.data;
	}
};

template<typename T, const ConfigBJMM<T> &config>
void* BJMMv2_Thread(void *par) {
	// const values
	constexpr static uint32_t n = config.n; // instance parameter
	constexpr static uint32_t k = config.k; // instance parameter
	constexpr static uint32_t w = config.w; // instance parameter
	constexpr static uint32_t l = config.l; // instance parameter
	constexpr static uint32_t l1= config.l1;// instance parameter
	constexpr static uint32_t p = config.p; // instance parameter

	constexpr static uint32_t gpu_width = config.gpu_matrix_width;  // number of limbs for one column in the parity check matrix
	constexpr static uint32_t gpu_l_limbpos = config.gpu_l_limbpos; //

	// internal Datatypes
	using BucketType            = LogTypeTemplate<MAX(l1, MAX(l, 32))>;
	using CombinationsType      = uint32_t;

	//
	ParamsBJMM *params = (ParamsBJMM *)par;

	// CPU stuff
	mzd_t *syndrom_tmp  = mzd_init(params->syndrom->ncols, params->syndrom->nrows);
	mzd_t *syndromT_tmp = mzd_init(params->syndrom->nrows, params->syndrom->ncols);
	mzd_t *e            = mzd_init(1, n);
	mzd_t *eT           = mzd_init(n, 1);
	mzd_t *wH           = matrix_init(n-k, n+1);
	mzd_t *wHT          = mzd_init(wH->ncols, wH->nrows);
	mzd_t *H           =  mzd_init(n - k, k+l+1);
	mzd_t *HT           = mzd_init(H->ncols, H->nrows);
	mzp_t *permutation  = mzp_init(n);

	//matrix_concat(wH, params->A, params->syndromT);
	mzd_t *tmp = matrix_concat(nullptr, params->A, params->syndromT);
	mzd_copy(wH, tmp);
	mzd_free(tmp);

	// GPU STUFF
	thrust::device_vector<T> de(config.M4RI_factor*params->A->width);
	checkCudaErrors(cudaPeekAtLastError());

	// Copy as much as possible on the cuda device. If it is not possible to pack everything on the device
	// it will be chunked. Also, every CPU thread needs its own `dCombination` vector. Otherwise, they could not concurrently work.
	thrust::host_vector<ChangeListElement<p>> hCombinations;
	thrust::device_vector<ChangeListElement<p>> dCombinations(params->tableSize);
	ChangeListElement<p>::Transform(hCombinations, *params->combinations);
	thrust::copy(hCombinations.begin(), hCombinations.end(), dCombinations.begin());

	// fill k+l bits
	thrust::device_vector<Columns<T, gpu_width>> dL1(params->tableSize);

	//
	uint32_t *dCompare  = nullptr, compare=0;
	checkCudaErrors(cudaMalloc(&dCompare, sizeof(uint32_t)));


	// GPU stuff + Pointer abstraction
	thrust::device_vector<Columns<T, gpu_width>> dHT(HT->nrows);

	// timing stuff
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float kernel_runtime = 0., total_kernel_runtime = 0.;

	double time = (double)clock();
	uint64_t loops = 0;
	uint32_t m4ri_k = matrix_opt_k(n - k, n);
	customMatrixData *matrix_data = init_matrix_data(wH->ncols);

	while (1u) {
		loops += 1;

		// create for each stream its own gaus elimination
		matrix_create_random_permutation(wH, wHT, permutation);
		matrix_echelonize_partial_plusfix(wH, m4ri_k, n-k-l, matrix_data, 0, n - k - l, 0, permutation);

		mzd_submatrix(H, wH, 0, n-k-l, n-k, n+1);
		matrix_transpose(HT, H);

		cudaEventRecord(start);

		Columns<T, gpu_width>::MatrixToCuda(dHT, HT);
		Columns<T, gpu_width> *pdHT = thrust::raw_pointer_cast(dHT.data());
		checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		kernel_runtime = 0.;

		// fill the Left base list
		thrust::transform(dCombinations.begin(), dCombinations.end(),
		                  dL1.begin(), dL1.begin(),
		                  FillList<T, gpu_width, p>(pdHT));

		// create the hashmap
		using BLKey = custom_key_type<T, l1>;
		using BLValue = custom_value_type<T,1>;
		auto const empty_key_sentinel   = BLKey{T(-1)};
		auto const empty_value_sentinel = BLValue{T(-1)};
		cuco::static_map<BLKey, BLValue> map{dL1.size()*2, empty_key_sentinel, empty_value_sentinel};

		// fill the hashmap
		///map.insert(dL1.begin(), dL1.end(), custom_hash<T, l1>{}, custom_key_equals{});

		checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		print_matrix<<<1,1>>>(pdHT, k+l, gpu_width);
		for (size_t i = 0; i < dL1.size(); i++) {
			std::cout << dL1[i] << ", " << dCombinations[i] << "\n";
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&kernel_runtime, start, stop);
		total_kernel_runtime += kernel_runtime;

		if (!compare) {
			// apply back perm
			mzd_t *tmpe = mzd_init(1, n);
			CudaToMatrix<T>(tmpe, de);
			for (uint32_t i = 0; i < n; i++) {
				mzd_write_bit(e, 0, permutation->values[i], mzd_read_bit(tmpe, 0, i));
			}

			mzd_print(tmpe);
			mzd_print(e);
			mzd_free(tmpe);
			break;
		}
		if ((loops%10000) == 0) {
			const uint64_t cloops = loops;
			const double time2 = ((double)clock() -  time)/CLOCKS_PER_SEC;
			const double ktime = total_kernel_runtime/1e3;
			std::cout << "loops: " << cloops << ", loops/s:" << cloops/time2 << ", kernel runtime: " << ktime << "s (" << ktime/time2*100. << "%) , runtime: " << time2 << "s" << "\n" << std::flush;
		}
	}

	time = ((double)clock() - time)/CLOCKS_PER_SEC;
	total_kernel_runtime /= 1e3;
	std::cout << "runtime: "  << time << "s, loops: " << loops << ", total kernel runtime: " << total_kernel_runtime << "s (" << total_kernel_runtime/time*100. << "%), loops/kernel s: " << loops/total_kernel_runtime << ", loops/s: " << loops/time << "\n";

	// check correctness
	mzd_transpose(eT, e);
	mzd_mul_naive(syndrom_tmp, params->A, eT);
	mzd_transpose(syndromT_tmp, syndrom_tmp);

	//mzd_print(A);
	std::cout << "s is:\t\t";
	mzd_print(syndromT_tmp);
	std::cout << "s should:\t";
	mzd_print(params->syndrom);


	for (int i = 0; i < params->syndrom->ncols; ++i) {
		if(mzd_read_bit(params->syndrom, 0, i) != mzd_read_bit(syndrom_tmp, i, 0)) {
			std::cout << "error not correct\n";
			break;
		}
	}

	// CPU
	mzd_free(wH);
	mzd_free(wHT);
	mzd_free(HT);
	mzd_free(H);
	mzp_free(permutation);
	mzd_free(syndrom_tmp);
	mzd_free(syndromT_tmp);
	mzd_free(e);
	mzd_free(eT);

	checkCudaErrors(cudaFree(dCompare));
	return nullptr;
}


template<typename T, const ConfigBJMM<T> &config>
int CudaBJMMv2() {
	using BucketType            = LogTypeTemplate<MAX(config.l, 32)>;
	using CombinationsType      = uint32_t;

	srand(time(nullptr));
	config.print();

	// Try to allocate as much as possible in one thread and pass it to the other threads
	thrust::host_vector<CombinationsType> combinations;
	double time = (double)clock();
	auto numb_p_out_of_k_calc = calc_p_combinations<CombinationsType>(combinations, config.klpart_size, config.p);
	time = ((double)clock() - time)/CLOCKS_PER_SEC;
	if(config.numb_p_out_of_k != numb_p_out_of_k_calc) {
		std::cout << "ERROR: not correct combinations Dumer\n";
		std::cout << config.numb_p_out_of_k << " vs " << numb_p_out_of_k_calc << "\n";
		return -1;
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0);

	const uint64_t max_mem        = prop.totalGlobalMem;
	const uint64_t max_needed_mem = calc_bjmm_mem_consumption<T, config>(config.numb_p_out_of_k);
	const uint64_t tableReps      = ceil(double(max_needed_mem)/double(max_mem));
	const uint64_t tableSize      = config.numb_p_out_of_k/tableReps;   // size per repetition => Actually number of elements on the device
	const uint64_t needed_mem     = calc_bjmm_mem_consumption<T, config>(tableSize);
	assert(tableReps > 0);

	//const uint32_t number_sm    = prop.multiProcessorCount;
	constexpr uint64_t tpb        = MIN(256, config.numb_p_out_of_k);
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
	std::cout << "starting:             " << config.nr_threads << " threads with each: " << blocks << " blocks with each " << tpb << ", in total: : " << nr_gpu_threads << "\n";

	ParamsBJMM params (&combinations,blocks, tpb,tableSize, tableReps, mzd_from_str(n, n-k, h), mzd_from_str(1, n-k, s));
	params.A        = mzd_transpose(nullptr, params.AT);
	params.syndromT = mzd_transpose(nullptr, params.syndrom);

	BJMMv2_Thread<T, config>((void *)&params);
	cudaDeviceReset();


	// free everything we allocated
	mzd_free(params.syndrom);
	mzd_free(params.syndromT);
	mzd_free(params.A);
	mzd_free(params.AT);
	return 1;
}
