#pragma once
#include "cuhelper.cuh"


// TODO move to combinations.h
template <typename Iterator>
bool next_combination(const Iterator first, Iterator k, const Iterator last) {
	// src: https://stackoverflow.com/questions/127704/algorithm-to-return-all-combinations-of-k-elements-from-n
	// Credits: Mark Nelson http://marknelson.us
	if ((first == last) || (first == k) || (last == k))
		return false;
	Iterator i1 = first;
	Iterator i2 = last;
	++i1;
	if (last == i1)
		return false;
	i1 = last;
	--i1;
	i1 = k;
	--i2;
	while (first != i1) {
		if (*--i1 < *i2) {
			Iterator j = k;
			while (!(*i1 < *j)) ++j;
			std::iter_swap(i1,j);
			++i1;
			++j;
			i2 = k;
			std::rotate(i1,j,last);
			while (last != j) {
				++j;
				++i2;
			}
			std::rotate(k,i2,last);
			return true;
		}
	}

	std::rotate(first,k,last);
	return false;
}

/// generate n over p combinations and save the result in combinations
/// \tparam T limb type
/// \param combinations output of the form: (10, 2):
///				[0, 1, 0, 2, 0, 3, ..., 0, 9, 1, 2, 1, 3, ....]
///			get the i-th element easily via
///				combinations[i*p + j]
/// \param n	input parameter
/// \param p	input parameter
/// \return	number of combinations
//template<typename T = uint32_t>
//uint64_t calc_p_combinations(thrust::host_vector<T> &combinations, uint64_t nn, uint64_t pp) {
//	uint64_t ctr = 0;
//
//	std::vector<T> s(nn);
//	for (uint64_t i = 0; i<nn; i++){
//		s[i]=i;
//	}
//
//	do {
//		for (uint64_t i = 0; i < pp; i++) {
//			combinations.push_back(s[i]);
//		}
//		ctr += 1;
//	} while (next_combination(s.begin(), s.begin() + pp, s.end()));
//
//	return ctr;
//}

/// generate n over p combinations and save the result in combinations
/// \tparam T limb type
/// \param combinations output of the form: (10, 2):
///				[0, 1, 0, 2, 0, 3, ..., 0, 9, 1, 2, 1, 3, ....]
///			get the i-th element easily via
///				combinations[i*p + j]
/// \param n	input parameter
/// \param p	input parameter
/// \return	number of combinations
template<typename T = uint32_t>
uint64_t calc_p_combinations(T *combinations, uint64_t nn, uint64_t pp) {
	uint64_t ctr = 0;

	std::vector<T> s(nn);
	for (uint64_t i = 0; i < nn; i++){
		s[i] = i;
	}

	do {
		// copy it back
		for (uint64_t i = 0; i < pp; i++) {
			combinations[ctr*pp + i] = s[i];
		}

		ctr += 1;
	} while (next_combination(s.begin(), s.begin() + pp, s.end()) && ctr < nn);

	return ctr;
}

template<typename T = uint32_t>
void print_combinations(T* combinations, const uint64_t pp, const uint64_t size, const uint64_t start=0) {
	for (uint64_t i = start; i < size; i++) {
		for (uint64_t j = 0; j < pp; j++) {
			std::cout << combinations[i*pp + j] << " ";
		}
		std::cout << "\n";
	}
}

// template<typename T = uint32_t>
// void print_combinations(thrust::host_vector<T> &combinations, const uint64_t pp,
// 						const uint64_t size, const uint64_t start=0) {
// 	print_combinations(combinations.data(), pp, size, start);
// }


template<typename T=uint32_t>
__global__ void print_vector(const T *d, const uint32_t len) {
	for (uint32_t i = 0; i < len; i++) {
		printf("%lu ", d[i]);
	}
	printf("\n");
}

template<typename T=uint32_t>
__device__ void print_bit_limb(const T d, const uint32_t len, const char *s=nullptr) {
	T data = d;

	for (uint32_t i = 0; i < len; i++) {
		if (data & 1)
			printf("1");
		else
			printf("0");

		data >>= 1;
	}
	if (s != nullptr) {
		printf(s);
	}

	printf("*\n");

}


/// simply print a binary vector
/// \tparam T limb data type
/// \param d  pointer to the data
/// \param len number of limbs
template<typename T=uint32_t>
__device__ void print_bit_vector(const T *d, const uint32_t len) {
	constexpr static uint32_t bits = sizeof(T)*8;

	printf("[");

	for (uint32_t i = 0; i < len; i++) {
		T data = d[i];
		for (uint32_t j = 0; j < bits; j++) {
			if ((j % 4 == 0) && (j != bits - 1) && (j != 0))
				printf(":");

			if (data & 1)
				printf("1");
			else
				printf(" ");

			data >>= 1;
		}
		if (i != len-1)
			printf("|");
	}
	printf("]*\n");
}


/// kernel to print a matrix in the kernel space.
/// \tparam T limb type
/// \param d pointer to the data in the gpu kernel
/// \param rows number of rows
/// \param width number of limbs of type T in each row
template<typename T=uint32_t>
__global__ void print_matrix(const T *d, const uint32_t rows, const uint32_t width) {
	for (uint32_t i = 0; i < rows; i++) {
		print_bit_vector(&d[i*width], width);
	}
	printf("\n");
}

/// copies a m4ri matrix into a device array. This function does not allocate memory. You are responsible for that.
/// \tparam T	uint64_t/uint32_t. Anything else is not tested
/// \param dOut Output: Pointer to data array
/// \param in	Input: m4ri matrix.
template<typename T=uint32_t>
void MatrixToCuda(T *dOut, const mzd_t *in) {
	const uint64_t wide  = in->width;
	const uint64_t nrows = in->nrows;
	uint64_t *tmp = (uint64_t *)calloc(nrows * wide, sizeof(uint64_t));

	for(uint64_t i = 0; i < nrows; i++ ) {
		for(uint64_t j = 0; j < wide; j++){
			tmp[i * wide + j] = in->rows[i][j];
		}
	}

	checkCudaErrors(cudaMemcpy(dOut, tmp, wide * nrows * sizeof(uint64_t), cudaMemcpyHostToDevice));
	free(tmp);
}

/// reduces the size of the input matrix limb wise
/// \tparam T
/// \param dOut
/// \param in
template<typename T=uint32_t>
void MatrixToCudaReduce(T *dOut, const mzd_t *in, const uint32_t out_limbs) {
	const uint64_t wide  = in->width;
	const uint64_t nrows = in->nrows;
	T *tmp = (T *)calloc(nrows * out_limbs, sizeof(T));

	for(uint64_t i = 0; i < nrows; i++ ) {
		for(uint64_t j = 0; j < out_limbs; j++){
			// TODO not correct: e.g out_limbs = 3;
			// ATTENTION: Implicit cast
			tmp[i * wide + j] = in->rows[i][j];
		}
	}

	checkCudaErrors(cudaMemcpy(dOut, tmp, out_limbs * nrows * sizeof(T), cudaMemcpyHostToDevice));
	free(tmp);
}

/// same as `MatrixToCuda` but an stream can be specified for asynchronous memcpy execution.
template<typename T=uint32_t>
void MatrixToCuda(T *dOut, const mzd_t *in, cudaStream_t *stream) {
	const uint64_t wide  = in->width;
	const uint64_t nrows = in->nrows;
	uint64_t *tmp = (uint64_t *)calloc(nrows * wide, sizeof(uint64_t));

	for(uint64_t i = 0; i < nrows; i++ ) {
		for(uint64_t j = 0; j < wide; j++){
			tmp[i * wide + j] = in->rows[i][j];
		}
	}

	checkCudaErrors(cudaMemcpyAsync(dOut, tmp, wide * nrows * sizeof(uint64_t), cudaMemcpyHostToDevice, *stream));
	free(tmp);
}

// wrapper around `MatrixToCuda` for thrust arrays
template<typename T=uint32_t>
void MatrixToCuda(thrust::device_vector<T> &dOut, const mzd_t *in) {
	const uint64_t wide  = in->width;
	const uint64_t nrows = in->nrows;
	thrust::host_vector<uint64_t> tmp(wide*nrows);

	for (uint64_t i = 0; i < nrows; i++) {
		for (uint64_t j = 0; j < wide; j++) {
			tmp[i * wide + j] = in->rows[i][j];
		}
	}

	dOut = tmp;
}

/// inverse function of MatrixToCuda. Does not allow asynchronous execution. Doese not allocate space for the
/// m4ri output matrix. You are responsible for this.
/// \tparam T 	uint64_t/uint32_t. Anything else is not tested
/// \param out 	Output:	must be allocated
/// \param dIn 	input of length: `out->width*out->nrows`
template<typename T=uint32_t>
void CudaToMatrix(mzd_t *out, const thrust::device_vector<T> &dIn) {
	const uint64_t wide  = out->width;
	const uint64_t nrows = out->nrows;
	uint64_t *tmp = (uint64_t *)calloc(nrows * wide, sizeof(uint64_t));

	checkCudaErrors(cudaMemcpy(tmp,
	                           thrust::raw_pointer_cast(dIn.data()),
	                           wide * nrows * sizeof(uint64_t), cudaMemcpyDeviceToHost));

	for (uint64_t i = 0; i < nrows; i++) {
		for (uint64_t j = 0; j < wide; j++) {
			out->rows[i][j] = tmp[i * wide + j];
		}
	}

	free(tmp);
}

/// partly add up ds and h with an early exit if the weight is already to high
/// \tparam T
/// \param ds
/// \param h
/// \param wide
/// \param p
/// \param w
/// \return
template<typename T = uint32_t>
__device__ __forceinline__ uint32_t add_weight_row(const T *ds, const T *h,
												   const uint32_t wide,
												   const uint32_t p,
												   const uint32_t w) {
	// uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t weight = 0;

	for(int i = 0; (i < wide) && (weight <= (w-p)); i++) {
		T test_vector_segment = ds[i] ^ h[i];
		if constexpr(sizeof(T) == 8) {
			weight += (uint32_t) __popcll(test_vector_segment);
		} else {
			weight += (uint32_t) __popc(test_vector_segment);
		}
	}

	return weight;
}

/// In contrast to `add_weight_row` this function adds two vector where each thread in the y-Dimension is adding one limb.
template<typename T = uint32_t>
__device__ __forceinline__ uint32_t add_weight_row_ythread(const T *ds, const T *h,
                                                           const uint32_t wide,
                                                           const uint32_t p,
                                                           const uint32_t w) {
	uint32_t i = threadIdx.y;
	__shared__ uint32_t weight;
	T test_vector_segment = ds[i] ^ h[i];

	if constexpr(sizeof(T) == 8) {
		atomicAdd(&weight, __popcll(test_vector_segment));
	} else {
		atomicAdd(&weight, __popc(test_vector_segment));
	}

	return weight;
}


/// add in1 and in2 and s on n-k-l coordinates. And returns the weight
template<typename T, const uint32_t upper, const T umask>
__device__ __forceinline__ uint32_t add_upper_weight(const T *__restrict__ in1,
													 const T *__restrict__ in2) {
	uint32_t weight = 0;
	T tmp;

	#pragma unroll
	for (uint32_t i = 0; i < upper - 1; i++) {
		tmp = in1[i] ^ in2[i];
		if constexpr(sizeof(T) == 8) {
			weight += (uint32_t) __popcll(tmp);
		} else {
			weight += (uint32_t) __popc(tmp);
		}
	}

	tmp = (in1[upper - 1] ^ in2[upper - 1]) & umask;
	if constexpr(sizeof(T) == 8) {
		weight += (uint32_t) __popcll(tmp);
	} else {
		weight += (uint32_t) __popc(tmp);
	}

	return weight;
}

/// add in1 and in2 and s on n-k-l coordinates.
template<typename T, const uint32_t upper, const T umask>
__device__ __forceinline__ void add_upper(T *__restrict__ out, const T *__restrict__ in1, const T *__restrict__ in2) {
	#pragma unroll
	for (uint32_t i = 0; i < upper - 1; i++) {
		out[i] = in1[i] ^ in2[i];
	}
	out[upper - 1] = (in1[upper - 1] ^ in2[upper - 1]) & umask;
}


/// add in1 and in2 and s on n-k-l coordinates.
template<typename T,                // limb type
         const uint32_t gpu_width,  // width of the parity check matrix in limbs
         const uint32_t p,          // weight which is enumerated in the change list
         const uint32_t single_limb // if true the lpart is only in one limb
         >
__device__ __forceinline__
        void matrix_vector_mult(T *__restrict__ out,        // output array
                           const T *__restrict__ dH,        // parity check matrix
                           const uint16_t *__restrict__ dC, // array containing bit pos of "1" in error vector
                           const uint32_t index,            // thread id
                           const uint64_t target            // target, IMPORTANT must be correctly aligned for the lpart
                                   ) {
	#pragma unroll
	for(uint32_t i = 0; i < gpu_width; i++) {
		out[i] = s[i];

		#pragma unroll
		for (uint32_t j = 0; j < p; j++) {
			const uint16_t c = dC[p*index + j];
			out[i] ^= dH[(c * gpu_width) + i];
		}
	}

	// finally, add the target.
	if constexpr (single_limb) {
		out[gpu_width-1] ^= target;
	} else {
		// work compiler
		*(uint64_t *)(&out[gpu_width-2]) ^= target;
	}
}

///  only valid for Lee-Brickell
/// \tparam T	uint32_t/uint64_t
/// \tparam CombinationsType	uint32_t
/// \param de	output e
/// \param dH	current working sub matrix in transposed version
/// \param ds	syndrom
/// \param dC	combinations array
/// \param wide	number of `T` limbs needed to hold n-k coordinates
/// \param index	current thread number
/// \param n	instance parameter
/// \param k	instance parameter
/// \param p	instance parameter
template<typename T = uint32_t, typename CombinationsType = uint32_t>
__device__ __forceinline__ void recalc_e(T *de, T *dH, T *ds,
										 CombinationsType *dC,
										 const uint32_t wide, const uint32_t index,
										 const uint32_t n, const uint32_t k, const uint32_t p) {
	constexpr uint32_t T_bitwidth = sizeof(T) * 8;

	for(int i = 0; i < wide; i++) {
		T temp_element = ds[i];
		temp_element ^= dH[dC[index] * wide + i];
		de[i] = temp_element;
	}

	for(int j = 0; j < p; j++) {
		de[(dC[p*index + j] + n-k) / T_bitwidth] ^= (T(1)) << ((dC[p*index + j] + n-k) % T_bitwidth);
	}
}


// Extract the l the highest bits of dS
///
/// \tparam T limb type
/// \tparam BucketType lpart limb type
/// \tparam n instance parameter
/// \tparam k instance parameter
/// \tparam l instance parameter
/// \tparam w instance parameter
/// \tparam p instance parameter
/// \tparam width number of limbs
/// \tparam llimbpos limb position of the lpart
/// \param dS outpur
/// \param index current thread id
/// \return l part, zero aligned
template<typename T, typename BucketType,
		const uint32_t n, const uint32_t k, const uint32_t l, const uint32_t w, const uint32_t p,
		const uint32_t width, const uint32_t llimbpos>
__device__ __forceinline__
BucketType extract_l_bit(const T *__restrict__ dS,       // radix/counting search helper struct
                         const uint32_t index){          // current thread index
	constexpr uint32_t T_bitwidth = sizeof(T) * 8;
	constexpr uint32_t nk = (n - k) % T_bitwidth;

	// TODO not fully correct for any n-k-l
	if constexpr (((n-k-l) / T_bitwidth) == ((n-k) / T_bitwidth)) {
		// IMPORTANT: Implicit type cast
		//printf("%d %d %d %d %d %d %d %d\n", n, k, l, nk, width, llimbpos, (n-k-l) / T_bitwidth, (n-k) / T_bitwidth);
		ASSERT(nk > l);
		T ret = dS[index*width + llimbpos] >> (nk-l);
		ASSERT((ret >> l) == 0);
		return ret;
	} else {
		constexpr uint32_t shift = (n-k-l)%64;
		return (*(uint64_t *)&dS[index*width + llimbpos]) >> shift;

	}
}

template<typename T,            // limb type
         typename BucketType,   // output type containing l bits
		 const uint32_t n,      // instance parameter
         const uint32_t k,      // instance parameter
         const uint32_t l>      // instance parameter
__device__ __forceinline__
BucketType extract_l_bit(const T *__restrict__ limbs) {
	static_assert(l < sizeof(BucketType)*8);
	static_assert(l < 64);

	constexpr uint32_t T_bitwidth = sizeof(T) * 8;
	constexpr uint32_t nk           = n-k;
	constexpr uint32_t nkl          = n-k-l;
	constexpr uint32_t l_limb       = nkl/T_bitwidth;
	if constexpr (l_limb == (nk/ T_bitwidth)) {
		// IMPORTANT: Implicit type cast
		constexpr uint32_t shift    = nkl%T_bitwidth;
		return limbs[l_limb] >> shift;
	} else {
		constexpr uint32_t shift    = nkl%T_bitwidth;
		//constexpr uint32_t off      = nk - ((l_limb+1) * T_bitwidth);
		constexpr uint32_t off      = T_bitwidth - shift;
		// printf("off: %d %d %d %d\n", nk,l_limb, off, shift);
		const T ret = (limbs[l_limb] >> shift) ^ (limbs[l_limb + 1] << (off));
		return ret;
	}
}

template<typename T,            // limb type
		 typename BucketType,   // output type containing l bits
		 const uint32_t n,      // instance parameter
		 const uint32_t k,      // instance parameter
		 const uint32_t l,      // instance parameter
         const uint32_t l1>     // instance parameter
__device__ __forceinline__
BucketType extract_l_bit(const T *__restrict__ limbs) {
	static_assert(l < sizeof(BucketType)*8);
	static_assert(l < 64);
	static_assert(l1<l);

	constexpr uint32_t T_bitwidth = sizeof(T) * 8;
	constexpr uint32_t nkl          = n-k-l;
	constexpr uint32_t nkl1         = n-k-l1;

	constexpr uint32_t l_limb       = nkl/T_bitwidth;
	constexpr uint32_t l1_limb      = nkl1/T_bitwidth;
	if constexpr (l_limb == l1_limb) {
		// IMPORTANT: Implicit type cast
		constexpr uint32_t shift    = nkl%T_bitwidth;
		constexpr T mask            = (T(1u) << l) - T(1u);
		return (limbs[l_limb] >> shift) & mask;
	} else {
		constexpr uint32_t shift     = nkl%T_bitwidth;
		constexpr uint64_t mask      = (T(1u) << l) - T(1u);
		return ((*(uint64_t *)&limbs[l_limb]) >> shift) & mask;
	}
}


///
/// \param e 	1 x n
/// \param A 	n-k x n(+1)
/// \param s 	n-k x 1
/// \param w	max weight expected on `e`
/// \param col if != -1 s is ignored and the syndrom is extracted as the n column in A
/// \return
bool checkCorrectness(const mzd_t *e, const mzd_t *A, const mzd_t *s, const uint32_t w, const int32_t col= -1) {
	assert(e->nrows == 1);
	assert(e->ncols <= A->ncols);
	assert(s->ncols == 1);

	bool ret            = true;
	mzd_t *syndrom      = (mzd_t *)s;
	if (col != -1) {
		syndrom = mzd_init(A->nrows, 1);
		for (uint32_t i = 0; i < A->nrows; i++)
			mzd_write_bit(syndrom, i, 0, mzd_read_bit(A, i, col));
	}

	// check correctness
	mzd_t *eT           = mzd_transpose(nullptr, e);
	mzd_t *syndrom_tmp  = mzd_mul_naive(nullptr, A, eT);
	mzd_t *syndromT_tmp = mzd_transpose(nullptr, syndrom_tmp);
	mzd_t *syndromT     = mzd_transpose(nullptr, syndrom);

	//mzd_print(A);
	std::cout << "s is:\t\t";
	mzd_print(syndromT_tmp);
	std::cout << "s should:\t";
	mzd_print(syndromT);

	// check syndrome correctness
	for (uint32_t i = 0; i < syndrom->ncols; ++i) {
		if(mzd_read_bit(syndrom, i, 0) != mzd_read_bit(syndrom_tmp, i, 0)) {
			std::cout << "error not correct\n";
			ret = false;
			break;
		}
	}

	// check weight
	uint32_t weight = 0;
	for (uint32_t i = 0; i < e->ncols; i++) {
		weight += mzd_read_bit(e, 0, i);
	}
	std::cout << "weight: " << weight << "\n";
	mzd_print(e);
	ret &= (weight <= w);

	mzd_free(syndromT);
	mzd_free(syndrom_tmp);
	mzd_free(syndromT_tmp);
	mzd_free(eT);
	if (col != -1) {
		mzd_free(syndrom);
	}

	return ret;
}


