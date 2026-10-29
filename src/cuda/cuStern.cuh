#pragma once

#undef NDEBUG
#define DEBUG 1

//#include "cuhelper.cuh"
#include "cuMisc.cuh"

#include <stdint.h>
//#include <cuda/atomic>

// NOTE: must be false
// Allows to switch on/off the internal access to the hashmap.
constexpr bool CUDAATOMIC = false;

#define BUCKET_SIZE (1)

template<typename T>
class ConfigDumer {
	/// TODO wrong
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
	const uint32_t p; // weight of enumerate in the baselists, not the total weight on the k+l coordinates, which is going to be 2*p
	const uint32_t l; // number of bits to match on the first level.

	// Additional parameters
	// multiplicative factor between the number of limbs needed to represent a column
	// (or row in the transposed matrix) on the cpu and gpu.
	const uint32_t cpu_gpu_width_factor = sizeof(uint64_t) / sizeof(T);
	const uint32_t cpu_matrix_row   = k+l+1;        // CPU Matrix rows of HT, +1 for the syndrome
	const uint32_t cpu_matrix_width = (n-k+63)/64;  // CPU Matrix width in limbs of HT

	// number of limbs per column (or row in the transposed matrix)
	const uint32_t gpu_matrix_width = cpu_gpu_width_factor * cpu_matrix_width;

	// zero aligned limb position of the 'l' part.
	//IMPORTANT: must be 0 aligned
	const uint32_t llimbpos         = (n-k-l-1)/(sizeof(T)*8);

	// if set to true 'Stern' algorithm will be used, else dumers.
	// TODO implement.
	const bool stern = false;

	// TODO describe
	const uint32_t iHwidth = l <= 32 ? 1 : 2;

	// k + l...
	const uint32_t kl_size          = k + l;

	// on how many coordinates does the mitm enumerate.
	const uint32_t klpart_size      = cceil(float(kl_size) / 2.);

	// (k+l)/2 over p
	const uint64_t numb_p_out_of_k  = bc(uint64_t(klpart_size), uint64_t(p));

	// internal table size for the collision search.
	const uint64_t l_table_size     = uint32_t(1) << l;

	// Calculate the optimal `r` value for M4RI
	const uint32_t m4ri_k = matrix_opt_k(n - k, MATRIX_AVX_PADDING(n));

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
		const __uint128_t a1 = bc(n-kl_size, w-p);
		const __uint128_t a2 = bc(n, w) / a1;
		const __uint128_t a3 = a2 / bc(kl_size, p);
		return a3;
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
		          << ", cpu_matrix_row: " << cpu_matrix_row
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

///
class ParamsDumer {
public:
	// Cuda parameters
	//thrust::host_vector<uint32_t> *combinations;
	uint32_t *combinations;

	uint64_t blocks = 0;
	uint64_t tpb = 0;

	// total size of a table
	uint64_t tableSize = 0;

	// number of times the table must be loaded onto the GPU until all
	// values have been enumerated.
	uint64_t tableReps = 0;

	// CPU Parameters
	// Parity Check Matrix Transposed: (n-k) x n
	mzd_t *AT = nullptr;
	// Parity Check Matrix n x (n-k)
	mzd_t *A = nullptr;

	// syndrome n x 1
	mzd_t *syndrom = nullptr;
	mzd_t *syndromT = nullptr;

	///
	/// \param b pointer to ?
	/// \param blocks number of blocks to launch
	/// \param tpb threads per block to launch
	/// \param tableSize
	/// \param tableReps
	/// \param AT
	/// \param syndrome
	// ParamsDumer (thrust::host_vector<uint32_t> *b,
	//             uint64_t blocks,
	//             uint64_t tpb,
	//             uint64_t tableSize,
	//             uint64_t tableReps,
	//             mzd_t *AT,
	//             mzd_t *syndrome) :
	// 		combinations(b), 
	// 		blocks(blocks), 
	// 		tpb(tpb), 
	// 		tableSize(tableSize), 
	// 		tableReps(tableReps), 
	// 		AT(AT), 
	// 		syndrom(syndrome) {
	//     ASSERT(b->size() > 0);
	// };

	ParamsDumer (uint32_t *b,
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
			syndrom(syndrome) {
	    // ASSERT(b->size() > 0);
	};

	void print() const {
		std::cout << "blocks: " << blocks
		          << ", threads per block: " << tpb
		          << ", tableSize: " << tableSize
		          << ", tableReps: " << tableReps
		          << "\n";
	}
};


/// calculates H2*e2 + s and prepare the counting sort algorithm
/// \tparam T limb type
/// \tparam BucketType type of the l part
/// \tparam n instance parameter
/// \tparam k instance parameter
/// \tparam l instance parameter
/// \tparam w instance parameter
/// \tparam p instance parameter
/// \tparam width width in limbs for one column in the parity check matrix
/// \tparam llimbpos limb position of the lpart
/// \param dH parity check matrix
/// \param dss syndrome (copied into shared mem)
/// \param dC change list, contains the
/// \param nr_combinations size of the list to enumerate: (k+l)/2 over p = |dC| = size of the change list.
template<typename T, typename BucketType,
		 const uint32_t n,
         const uint32_t k,
         const uint32_t l,
         const uint32_t w,
         const uint32_t p,
		 const uint32_t width,
         const uint32_t llimbpos>
__device__ void dumer_step1(const T *__restrict__ diH,                   // parity check matrix
							const T dis,
							const uint32_t *__restrict__ dC,    // array containing bit pos of "1" in error vector
                            uint32_t* pdBucket,
                            uint32_t* pdCounter,
                            const uint64_t nr_combinations){
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	while(idx < nr_combinations) {
		__syncthreads();
		ASSERT(dC[p*idx] < ((k+l)/2));
		// left column + syndrome and extract lower l bit
		T a = dis; //extract_l_bit<T, BucketType, n, k, l>(ds);

		#pragma unroll
		for (uint32_t i = 0; i < p; i++) {
			// const uint32_t j = width * dC[idx*p + i];
			// a ^=  extract_l_bit<T, BucketType, n, k, l>(&(dH[j]));
			const uint32_t j = dC[idx*p + i];
			a ^= diH[j]; 
		}

    	// const uint32_t a = deviceExtractLBit(ds[0] ^ dH[dC[idx]]);
    	ASSERT(a < (1u << l));

    	const uint32_t pos = MIN(BUCKET_SIZE, atomicAdd((uint32_t*) &pdCounter[a], 1));
		pdBucket[a*BUCKET_SIZE + pos] = idx;
		//printf("%d: inserted %d at %d\n", idx, a, pos);

		idx += blockDim.x * gridDim.x;
	}
}


/// \tparam T limb type of the rows/columns
/// \tparam BucketType minimal datatype type to hold `l` bits
/// \tparam n code length
/// \tparam k code dimension
/// \tparam l number of bits to match on in the first lvl
/// \tparam w code weight
/// \tparam p number of bits to enumerate in the base lists
/// \tparam width number of limbs each row (or column in the not transposed matrix) consists of.
/// \tparam llimbpos limb position of the `l` part. NOTE: this is only the lowest limb, if the `l` part extends over mutliple limbs
/// \param de output: error vector
/// \param dH input: parity check matrix
/// \param ds input: syndrome
/// \param dC input: combinations of the error vector we are enumerating in the base lists
/// \param nr_combinations number of of elements in the second list (right list) 
/// \param compare output: set to 1 if the solution was found, 0 otherwise 
template<typename T,
         typename BucketType,
		 const uint32_t n,
         const uint32_t k,
         const uint32_t l,
         const uint32_t w,
         const uint32_t p,
		 const uint32_t width,
         const uint32_t llimbpos>
__device__ void dumer_step3(T *__restrict__ de,                         // output error vector
                            const T *__restrict__ dH,                   // parity check matrix
                            const T *__restrict__ ds,                   // syndrome
                            const T *__restrict__ diH,                   // parity check matrix
							const T dis,
                            const uint32_t *__restrict__ dC,            // table
                            const uint32_t* __restrict__ pdBucket,
                            const uint32_t* __restrict__ pdCounter,
                            const uint32_t nr_combinations,             // size of table
                            uint32_t *compare) {  						// TODO reduce to only one parameter
	constexpr uint32_t kl_half      = cceil(double(k+l)/2.); 
	constexpr uint32_t T_bitwidth   = sizeof(T) * 8u;
	constexpr uint32_t nkl          = n-k-l;
	constexpr uint32_t weightt      = w - (2u*p);
	constexpr T umask               = (T(1u) << (nkl % T_bitwidth)) - 1;

	const uint32_t block_number     = blockDim.x * gridDim.x;
	uint32_t index_right            = threadIdx.x + blockIdx.x * blockDim.x;

	// security check
	//	assert(blockDim.x >= k+l);
	//
	//	// copy the syndrome into the shared memory
	//	__shared__ T s[width], dH[(k+l)*width];
	//	if (threadIdx.x < (k+l)) {
	//		// NOTE: see `cuLeeBrickellNoMem` to understand what im doing
	//		for (uint32_t i = 0; i < width; i++) {
	//			dH[threadIdx.x + threadIdx.x*i] = dHH[threadIdx.x + threadIdx.x*i];
	//		}
	//	}

	
	while(index_right < nr_combinations) {
    	// no width needed, only 1 int per column
		// NOTE same as in `dumer_step1` but for the right list

		T a_right = 0;	

		#pragma unroll
		for (uint32_t i = 0; i < p; i++) {
			// const uint32_t j = width * (dC[index_right *p + i] + kl_half);
			// a_right ^=  extract_l_bit<T, BucketType, n, k, l>(&(dH[j]));
			const uint32_t j = dC[index_right*p + i] + kl_half;
			a_right ^= diH[j];
		}

		// some security check
		assert(a_right < (1u << l));

		//if (pdCounter[a_right] > 0)
		//	printf("%d: counter: %d at %d\n", index_right, pdCounter[a_right], a_right);

		__syncthreads();
		for (uint32_t j = 0; j < MIN(BUCKET_SIZE, pdCounter[a_right]); j++) {
			__syncthreads();
			// get the pdC index for the collision
			const uint32_t index_left = pdBucket[a_right * BUCKET_SIZE + j];

			// just some sanity check
			// TODO only correct for p == 1assert(index_left < kl_half);
			// assert(index_right < kl_half);
			assert(index_left < nr_combinations);
			assert(index_right < nr_combinations);

			//printf("%d %d %d %d %d %d %d %d %d\n", pdCounter[a_right], index_right, index_left, dis, a_right, dC[index_left], p, width, nr_combinations);
			// TODO only correct for p = 1
			const T abc = dis ^ a_right ^ diH[dC[index_left*p + 0]];
			//if (abc != 0){
			//	printf("%d %d %d %d\n", abc, dis, a_right, diH[dC[index_left*p + 0]]);
			//	printf("%d %d %d %d %d %d %d %d %d %d\n", pdCounter[a_right], index_right, index_left, dis, a_right, dC[index_left], p, width, nr_combinations, j);
			//}
			if constexpr (p == 1)
				assert(0 == abc);

			//printf("%d %d %d\n", llimbpos, p, width);
			// compute the remaining n-k-l coordinates
			uint32_t weight = 0;

			// TODO switch loop, thats not easy
			// TODO only correct for p==1, because of the double add of dS
			#pragma unroll
			for (uint32_t i = 0; i < llimbpos; i++) {
				#pragma unroll
				for (uint32_t r = 0; r < p; r++) {
					weight += __popc(ds[i] ^
					                 dH[ (           dC[index_left  * p + r]  * width) + i] ^
					                 dH[((kl_half  + dC[index_right * p + r]) * width) + i]);
				}
			}

			// TODO kann glaube ich weggelassen werden
			// add the last limb

			for (uint32_t r = 0; r < p; r++) {
				weight += __popc((ds[llimbpos]                                                 ^
				                  dH[           (dC[index_left  * p + r]  * width) + llimbpos] ^
				                  dH[((kl_half + dC[index_right * p + r]) * width) + llimbpos]) &
				                 umask);
			}

			//printf("w: %d %d %d %d\n", weight, weightt, umask, llimbpos);
			// if we pass this check we found a solution
			if (weight <= weightt) {
				if (0 != atomicCAS(compare, 0, 1)) {
					return;
				}

				// printf("\n");
				// printf("Weight: %d\n", weight);
				// printf("dC[left]: %d\n",  dC[index_left]);
				// printf("dC[right]: %d\n", dC[index_right]);

				// copy back the solution
				for (uint32_t i = 0; i < p; i++) {
					de[2*i + 0] = dC[index_left *p + i];
					de[2*i + 1] = dC[index_right*p + i];
				}
		//
		//		// compute the rest
		//		for (uint32_t i = 0; i < llimbpos; i++) {
		//			de[i] = ds[i] ^ dH[dC[index_left*p + i]] ^ dH[kl_half + dC[index_right*p + i]];
		//		}
		//		de[llimbpos] ^= (ds[llimbpos] ^ dH[dC[index_left*p + llimbpos]] ^ dH[kl_half + dC[index_right*p + llimbpos]]) & umask;

		//		for (uint32_t i = 0; i < p; i++) {
		//			const uint32_t pos1     = (nkl + kl_half +     dC[p*index_right  + i]) / T_bitwidth;
		//			const uint32_t pos2     = (nkl               + dC[p*index_left   + i]) / T_bitwidth;
		//			const uint32_t shift1   = (nkl + kl_half +     dC[p*index_right  + i]) % T_bitwidth;
		//			const uint32_t shift2   = (nkl               + dC[p*index_left   + i]) % T_bitwidth;

		//			printf("weight: %u\n", weight);
		//			printf("weightt: %u\n", weightt);
		//			printf("umask: %u\n", umask);
		//			printf("nkl: %u\n", nkl);
		//			printf("(nkl mod T_bitwidth): %u\n", (nkl % T_bitwidth));
		//			//printf("a: %u\n", (nkl + kl_half + dC[p*index_right + i]));
		//			//printf("b: %u\n", dC[p*index_right + i]);
		//			//printf("a: %u\n", (nkl + kl_half + dC[p*index_right + i]));
		//			//printf("b: %u\n", dC[p*index_left + i]);

		//			printf("pos1: %u\n", pos1);
		//			printf("pos2: %u\n", pos2);
		//			printf("shift1: %u\n", shift1);
		//			printf("shift2: %u\n", shift2);
		//			printf("index: %u\n", index_right);
		//			printf("index1: %u\n", index_left);

		//			printf("bit1: %u\n", pos1*T_bitwidth + shift1);
		//			printf("bit2: %u\n", pos2*T_bitwidth + shift2);

		//			de[pos1] ^= T(1u) << (shift1);
		//			de[pos2] ^= T(1u) << (shift2);
		//		}
			}
		}

		// this sync is important.
		// but also slows down the whole thing
		__syncthreads();
		index_right += block_number;
	}
}

/// TODO explain
/// \tparam T
/// \tparam BucketType
/// \tparam n
/// \tparam k
/// \tparam l
/// \tparam w
/// \tparam p
/// \tparam width
/// \tparam llimbpos
/// \param pde
/// \param pdH
/// \param pds
/// \param pdC
/// \param nr_combinations
/// \param compare
template<typename T,
         typename BucketType,
		 const uint32_t n,
         const uint32_t k,
         const uint32_t l,
         const uint32_t w,
         const uint32_t p,
		 const uint32_t width,
         const uint32_t llimbpos>
__global__ void KernelDumer(T *__restrict__ pde,                // TODO explain
                            const T *__restrict__ pdH,          //
                            const T *__restrict__ pds,          //
                            const T *__restrict__ pdiH,         // parity check matrix
							const T dis,                        // this limits l < 32
                            const uint32_t *__restrict__ pdC,   //
                            uint32_t* __restrict__ pdBucket,	//
                            uint32_t* __restrict__ pdCounter,	//
                            const uint64_t nrComb,      		// TODO compiletime argument
                            uint32_t *compare) {                //
	// clear the bucket counter
    // thrust::fill(thrust::device, pdCounter, pdCounter + (1u << l), 0);
	
	// TODO not correct
	uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < (1u << l)) {
		pdCounter[idx] = 0;
		idx += blockDim.x * gridDim.x;
	}

	__syncthreads();

    dumer_step1
		<T, BucketType, n, k, l, w, p, width, llimbpos>
		(pdiH, dis, pdC, pdBucket, pdCounter, nrComb);
	__syncthreads();

     dumer_step3
	 	<T, BucketType, n, k, l, w, p, width, llimbpos>
	 	(pde, pdH, pds, pdiH, dis, pdC, pdBucket, pdCounter, nrComb, compare);
     __syncthreads();
}

template<typename T, const uint32_t n, const uint32_t k, const uint32_t l>
T kekw_exctract_l_bit(const mzd_t *__restrict__ wH) {
	T ret = 0;
	//mzd_print((mzd_t *)wH);
	for (uint32_t i = n-k-l; i < n-k; i++) {
		uint32_t bit = mzd_read_bit(wH, i, n);
		ret ^= (bit << (i-(n-k-l)));
	}
	return ret;
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

	constexpr static uint32_t width 	= config.gpu_matrix_width;  //
	constexpr static uint32_t llimbpos 	= config.llimbpos;       	//

	// TODO generalize
	using BucketType            = LogTypeTemplate<MAX(l, 32)>;
	using CombinationsType      = uint32_t; // uint16_t;

	std::cout << "|BucketType|: " << sizeof(BucketType) * 8
			  << ", |CombinationsType|: " << sizeof(CombinationsType) * 8 
			  << std::endl;

	// TODO describe
	ParamsDumer *params = (ParamsDumer *)par;

	// CPU stuff
	mzd_t *e            = mzd_init(1, config.n);
	mzd_t *wH           = matrix_init(config.n-config.k, config.n+1);

	mzd_t *wHT          = mzd_init(wH->ncols, wH->nrows);
	mzd_t *H           =  mzd_init(config.n-config.k, config.k+config.l+1);
	mzd_t *HT           = mzd_init(H->ncols, H->nrows);
	mzd_t *iHT          = mzd_init(config.k+config.l, config.l);             // k \times l matrix containing only the `l` part of the transposed parity check matrix
	mzp_t *permutation  = mzp_init(config.n);

	//matrix_concat(wH, params->A, params->syndromT);
	mzd_t *tmp = matrix_concat(nullptr, params->A, params->syndromT);
	mzd_copy(wH, tmp);
	mzd_free(tmp);

	// Copy as much as possible on the cuda device. If it is not possible to pack everything on the device
	// it will be chunked. Also, every CPU thread needs its own `dCombination` vector. Otherwise, they could not concurrently work.
	// thrust::device_vector<CombinationsType> dCombinations(p*params->tableSize);
	// thrust::copy(params->combinations->begin(), params->combinations->begin()+(p*params->tableSize), dCombinations.begin());

	uint32_t *dCompare  = nullptr, Compare = 0;

	checkCudaErrors(cudaMalloc(&dCompare, sizeof(uint32_t)));
	checkCudaErrors(cudaMemcpy(dCompare, &Compare, sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaDeviceSynchronize());
	
	// GPU stuff + Pointer abstraction
	T* pdHT;
	checkCudaErrors(cudaMalloc(&pdHT, config.cpu_gpu_width_factor * config.cpu_matrix_row * config.cpu_matrix_width * sizeof(T)));
	T* pdsT                 = &pdHT[config.cpu_gpu_width_factor * (config.cpu_matrix_row - 1) * config.cpu_matrix_width];
	// CombinationsType* pdC   = thrust::raw_pointer_cast(dCombinations.data());
	CombinationsType* pdC   = params->combinations;
	checkCudaErrors(cudaDeviceSynchronize());

	uint32_t *pdBucket, *pdCounter;
	checkCudaErrors(cudaMalloc(&pdBucket, (1u << l)* BUCKET_SIZE * sizeof(uint32_t)));
	// pdBucket[x] = idx, sd. H*idx == x auf l coordinate
	checkCudaErrors(cudaMalloc(&pdCounter, (1u << l)* sizeof(uint32_t)));
	checkCudaErrors(cudaDeviceSynchronize());
	printf("BucketSize: %d\n", BUCKET_SIZE);

	// small matrix
	T *pdiHT;
	checkCudaErrors(cudaMalloc(&pdiHT, config.iHwidth * config.cpu_matrix_row * sizeof(T)));

	T *pde;
	cudaMallocManaged(&pde, sizeof(T) * 2 * p);

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

		// TODO opt memcopy in this functions
		MatrixToCuda<T>(pdHT, HT);
		DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError()));
		DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize()));


		mzd_submatrix(iHT, HT, 0, config.n-config.k-config.l, config.k+config.l, config.n-config.k);
		MatrixToCudaReduce<T>(pdiHT, iHT, config.iHwidth);
		checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		// also extract the syndrome
		const T iSyndrom = kekw_exctract_l_bit<T, config.n, config.k, config.l>(wH);

		// DO NOT DELETE, i have to remember to correct calling values for the cuda stuff
		//		mzd_print(wH);
		//		std::cout << "\n";
		//		mzd_print(H);
		//		std::cout << "\n";
		//		mzd_print(HT);
		//		std::cout << "\n";
		//		print_matrix<<<1,1>>>(pdHT, config.matrix_rows, 2*config.matrix_width);
		//mzd_print(wH);
		//mzd_print(H);

		//mzd_print(HT);
		//checkCudaErrors(cudaDeviceSynchronize());

		//printf("%d %d %d %d\n", HT->nrows, HT->ncols, iHT->nrows, iHT->ncols);
		//std::cout << "pdHT\n";
		//print_matrix<T><<<1,1>>>(pdHT, k+l+1, config.gpu_matrix_width);
		//checkCudaErrors(cudaDeviceSynchronize());

		//std::cout << "pdsT\n";
		//print_matrix<T><<<1,1>>>(pdsT, 1, config.gpu_matrix_width);
		//checkCudaErrors(cudaDeviceSynchronize());

		//std::cout << "pdiHt\n";
		//print_matrix<T><<<1,1>>>(pdiHT, k+l, config.iHwidth);
		//checkCudaErrors(cudaDeviceSynchronize());

		//std::cout << "\n" << std::flush;
		//std::cout << iSyndrom << " iSyndrom\n";
		//break;

		cudaEventRecord(start);
		kernel_runtime = 0.;
		for (uint32_t tableRet = 0; tableRet < params->tableReps; tableRet++) {
			// start the actual kernel
			KernelDumer<T, BucketType, n, k, l, w, p, width, llimbpos>
				<<<params->blocks, params->tpb>>>
					(pde, pdHT, pdsT, pdiHT, iSyndrom,
					 pdC, pdBucket, pdCounter,
			         params->tableSize, dCompare);

			checkCudaErrors(cudaPeekAtLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			// fetch new set of combinations.
			// if (params->tableReps != 1) {
			// 	if (tableRet != params->tableReps-1) {
			// 		const uint64_t from = p*tableRet*params->tableSize;
			// 		const uint64_t to   = p*(tableRet+1)*params->tableSize;
			// 		std::cout << "mem copy from: " << from << " to " << to << "\n";
			// 		thrust::copy(params->combinations->begin()+from,
			// 		             params->combinations->begin()+to,
			// 		             dCombinations.begin());
			// 	}
			// }
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&kernel_runtime, start, stop);
		total_kernel_runtime += kernel_runtime;

		checkCudaErrors(cudaMemcpy(&Compare, dCompare, sizeof(bool), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaDeviceSynchronize());

		if (Compare) {
			T cpu_e[2*p] = {0};

			checkCudaErrors(cudaMemcpy(cpu_e, pde, 2*p*sizeof(T), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaDeviceSynchronize());


			// TODO das auslagern
			using TT = uint64_t;
			constexpr uint32_t kl_half = (k+l)/2;
			constexpr uint32_t T_bitsize = sizeof (TT) * 8;
			constexpr uint32_t limbs = (n-k-l+T_bitsize -1 )/T_bitsize - 1u;
			constexpr TT limb_mask = (TT(1u) << ((n-k-l) % (T_bitsize))) - 1u;

			printf("\n");
			mzd_print_row(HT, k+l);
			mzd_print_row(HT, cpu_e[0]);
			mzd_print_row(HT, kl_half + cpu_e[1]);

			mzd_t *e1 = mzd_init(1, n);
			TT *ptr = (TT *)e1->rows[0];

			// add the syndrome
			for (uint32_t j = 0; j < limbs; j++) {
				ptr[j] ^= HT->rows[k+l][j];
			}
			ptr[limbs] ^= (HT->rows[k+l][limbs] & limb_mask);

			for (uint32_t i = 0; i < p; i++) {
				for (uint32_t j = 0; j < limbs; j++) {
					ASSERT(cpu_e[2*i+0] < (k+l)/2);
					ASSERT(cpu_e[2*i+1] < (k+l)/2);

					ptr[j] ^= HT->rows[          cpu_e[2*i+0]][j];
					ptr[j] ^= HT->rows[kl_half + cpu_e[2*i+1]][j];
				}

				// and to the limb part
				ptr[limbs] ^= (HT->rows[          cpu_e[2*i+0]][limbs]&limb_mask);
				ptr[limbs] ^= (HT->rows[kl_half + cpu_e[2*i+1]][limbs]&limb_mask);

				mzd_print(e1);

				// write the epart
				ASSERT(mzd_read_bit(e1, 0, n-k-l + cpu_e[2*i+0]) == 0);
				ASSERT(mzd_read_bit(e1, 0, n-k-l + cpu_e[2*i+1]) == 0);
				mzd_write_bit(e1, 0, n-k-l + cpu_e[2*i+0], 1);
				mzd_write_bit(e1, 0, n-k-l + cpu_e[2*i+1], 1);
			}

			// mzd_print(e1);

			for (uint32_t i = 0; i < n; i++) {
				mzd_write_bit(e, 0, permutation->values[i], mzd_read_bit(e1, 0, i));
			}

			mzd_free(e1);
			break;
		}

		if ((loops%config.print_loops) == 0) {
			const uint64_t cloops = loops;
			const double time2 = ((double)clock() -  time)/CLOCKS_PER_SEC;
			const double ktime = total_kernel_runtime/1e3;
			std::cout << "\rloops: " << cloops 
					  << "(" << double(cloops)/double(config.Loops())*100. << "), loops/s:" << cloops/time2 
					  << ", kernel runtime: " << ktime << "s (" << ktime/time2*100. 
					  << "%) , runtime: " << time2 << "s" << std::flush;
		}

		//printf("loops %d\n", loops);
		//if (loops > 1) {
		//	break;
		//}
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
	using BucketType   = LogTypeTemplate<MAX(config.l, 32)>;
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

	using CombinationType = uint32_t;

	// Try to allocate as much as possible in one thread and pass it to the other threads
	// thrust::host_vector<uint32_t> combinations;
	const uint64_t combination_size = config.p * config.numb_p_out_of_k * sizeof(uint32_t);
	CombinationType *combinations, *ccom;
	checkCudaErrors(cudaMalloc(&combinations, combination_size));
	ccom = (CombinationType *)malloc(combination_size);
	checkCudaErrors(cudaDeviceSynchronize());

	double time = (double)clock();
	auto numb_p_out_of_k_calc = calc_p_combinations<CombinationType>(ccom, config.numb_p_out_of_k, config.p);
	time = ((double)clock() - time)/CLOCKS_PER_SEC;
	if(config.numb_p_out_of_k != numb_p_out_of_k_calc) {
		std::cout << "ERROR: not correct combinations Dumer\n";
		std::cout << config.numb_p_out_of_k << " vs " << numb_p_out_of_k_calc << "\n";
		return -1;
	}

	// print_combinations(ccom, config.p, numb_p_out_of_k_calc);
	checkCudaErrors(cudaMemcpy(combinations, ccom, combination_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaDeviceSynchronize());
	free(ccom);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	const uint64_t max_mem        = prop.totalGlobalMem;
	const uint64_t max_needed_mem = calc_stern_mem_consumption<T, config>(config.numb_p_out_of_k);
	const uint64_t tableReps      = 1;//MAX(1., ceil(double(max_needed_mem)/double(max_mem)));
	const uint64_t tableSize      = config.numb_p_out_of_k/tableReps;   // size per repetition => Actually number of elements on the device
	const uint64_t needed_mem     = calc_stern_mem_consumption<T, config>(tableSize);
	assert(tableReps > 0);

	//const uint32_t number_sm    = prop.multiProcessorCount;
	constexpr uint64_t tpb        = MIN(MIN(config.k+config.l, config.numb_p_out_of_k), 1024);
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
	std::cout << "starting:             " << config.nr_threads 
			  << " threads with each: "   << blocks 
			  << " blocks with each " 	  << tpb 
			  << ", in total: : " 		  << nr_gpu_threads 
			  << "\n" << std::flush;

	// ParamsDumer params (&combinations, blocks, tpb, tableSize, tableReps,
	// 					mzd_from_str(n, n-k, h), mzd_from_str(1, n-k, s));

	ParamsDumer params (combinations, blocks, tpb, tableSize, tableReps,
						mzd_from_str(n, n-k, h), mzd_from_str(1, n-k, s));


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
