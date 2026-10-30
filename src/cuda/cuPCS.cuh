#ifndef DECODING_CUDAPCS
#define DECODING_CUDAPCS

#include <curand_kernel.h>
#include <math.h>

#include "cuhelper.cuh"
#include "cuMisc.cuh"
#include "cuHashMap.cuh"
#include "cuPollard.cuh"

// TODO think of something
template<class HMElement>
__device__ size_t PCSHashFkt(const HMElement &e) {
	return size_t(e.start);
}

/// configuration struct containing all important flags.
/// \tparam T	base limb type of everything on the gpu.
template<typename T>
class ConfigPCS {
private:
	// disable the default constructor
	constexpr ConfigPCS() : n(0), k(0), w(0), l(0), p(0) {}
public:
	// instance parameters
	const uint32_t n;   // code length
	const uint32_t k;   // code size
	const uint32_t w;   // code weight
	const uint32_t p;   // weight in the base lists. In this case the weight which is
						// mapped to by the collision functions
	const uint32_t l;   // bits to match on by the two collision functions

	// number of columns to swap before gaussian elimination
	const uint32_t gaus_c = 10;

	// if set to `true` the algorithm uses a perfect bijection to calculate the
	// hashing function.
	const bool bijection            = false;

	// if set to true: stern is used
	// if set to false: dumer is used
	const bool stern                = true;

	// length of the k part
	const uint32_t kl_length        = stern ? k : k+l;

	const uint32_t D                = 1;
	const size_t HashMapSize        = size_t(1) << 10;
	const uint32_t walk_length      = 1;

	// Additional parameters
	const uint32_t cpu_gpu_matrix_factor = sizeof(uint64_t) / sizeof(T); // size factor between a limb of the cpu and gpu
	const uint32_t cpu_matrix_rows  = kl_length+1;  // CPU Matrix rows of HT
	const uint32_t cpu_matrix_width = (n-k+63)/64;  // CPU Matrix width in limbs of HT
	const uint32_t gpu_matrix_width = cpu_gpu_matrix_factor * cpu_matrix_width; // same as `cpu_matrix_with` but for the gpu

	// Width of the lpart is always 1. => l < 32, l < 164.
	const uint32_t iHwidth          = l <= 32 ? 1 : 2;

	// position of the `l` part in the label.
	const uint32_t llimbpos         = (n-k-1)/(sizeof(T)*8); //IMPORTANT: must be 0 aligned

	// Calculate the optimal `r` value for M4RI
	const uint32_t m4ri_k = matrix_opt_k(n-k, MATRIX_AVX_PADDING(n));

	// if this value is set to a different value then -1 the program will end the
	// computation after `loops` permutation, regardless if it found the solution or not
	const uint64_t loops        = uint64_t(-1);

	// print every `print_loops` iterations a status update of the program
	const uint64_t print_loops  = 1000;

	// check every `exit_loops` iterations if another thread, which is working on a different permutation
	// found the solution. If so exit the computation.
	// NOTE: the check if something is found includes atomic operations, which are quite expensive
	//  so dont check every loop if something is found.
	const uint64_t exit_loops   = 1024;

	// Nr of stream per thread.
	// currently unused
	const uint32_t nr_streams   = 1;
	// number of different cpu threads working on different permutations.
	// currently unused
	const uint32_t nr_threads   = 1;

	/// TODO describe
	const uint32_t cuda_threads = 1 << (l-D);

	// TODO describe
	const uint32_t nr_iters     = 1;
	
	// TODO describe
	const uint32_t nr_flavors   = 1;

	/// standard constructor
	constexpr ConfigPCS(const uint32_t n,
	                        const uint32_t k,
	                        const uint32_t l,
	                        const uint32_t w,
	                        const uint32_t p,
	                        const uint32_t D=1,
	                        const uint32_t hash_map_size=1000,
	                        const uint32_t walk_length=100,
	                        const bool bijection=false,
	                        const bool stern=true,
	                        const uint32_t nr_iters=1,
	                        const uint32_t nr_flavors=1
	                        ) :
			n(n), k(k), l(l), w(w), p(p),
	        D(D), HashMapSize(hash_map_size), walk_length(walk_length),
	        bijection(bijection), stern(stern), nr_iters(nr_iters), nr_flavors(nr_flavors) {};

	constexpr ConfigPCS(ConfigPCS &c) = default;

	// returns the expected number of iterations
	constexpr uint64_t Loops() const noexcept {
#ifdef EXPECTED_PERMUTATIONS
		return EXPECTED_PERMUTATIONS;
#else
		return (bc(n, w) / bc(n-k-l, w-p)) / bc(k+l, p);
#endif
	}

	// print some information about the configuration
	void print() const noexcept {
		std::cout << "cuPCS n: " << unsigned(n)
		          << ", k: " << unsigned (k)
		          << ", l: " << unsigned (l)
		          << ", n-k:" << unsigned (n-k)
		          << ", p: " << unsigned (p)
				  << ", c: " << unsigned (gaus_c)
		          << ", cpu_gpu_matrix_factor: " << cpu_gpu_matrix_factor
		          << ", gpu_matrix_width: " << gpu_matrix_width
		          << ", cpu_matrix_width: " << cpu_matrix_width
		          << ", cpu_matrix_rows: " << cpu_matrix_rows
		          << ", llimbpos: " << llimbpos
		          << ", iHwidth: " << iHwidth
				  << ", nr_iters: " << nr_iters
				  << ", nr_flavors: " << nr_flavors
		          << ", bijection: " << bijection
				  << ", stern: " << stern
				  << ", D: " << D
				  << ", HashMapSize: " << HashMapSize
				  << ", walk_length: " << walk_length
		          << ", n-k-l: " << n-k-l << "\n";
	}

	/// \param loops current number of iterations computed
	/// \param time current runtime in seconds
	/// \param total_kernel_runtime current runtime of a CUDA kernel in secods
	void print_loop(const uint64_t loops, const double time, const float total_kernel_runtime) const noexcept {
		const double exptected_time = Loops()/loops*time;

		std::cout << "runtime: "  << time << "s"
		          << ", loops: " << loops << "(" << Loops() <<")=" << loops / Loops() << "x"
		          << ", total kernel runtime: " << total_kernel_runtime<< "s (" << total_kernel_runtime/time*100. << "%)"
		          << ", loops/kernel: " << loops/total_kernel_runtime
		          << ", loops/s: "<< loops/time
		          << ", E[RT]=" << exptected_time << "s(" << time/exptected_time << "x)"
		          << "\r" << std::flush; //"\n";
	}
};

/// simple function to determine if a given value is a Distinguishde point or not
/// \tparam T minim datatype to hold `l` bits
/// \tparam D number of bits to check if there are zero. Zero based check
/// \param value to check
/// \return true/false
template<typename T, const uint32_t D>
__device__ __forceinline__
bool is_distinguished(const T value) {
	constexpr T DMask = (T(1u) << D) - T(1u);
	return (value & DMask) == 0;
}

///
/// \tparam T
/// \param value
/// \param flavour
/// \return
template<typename T>
__device__ __forceinline__
bool compute_next(const T value, const T flavour) {
	static_assert(sizeof(T) == 4);
	// first calculates the inner product between
	// `value` and `flavor`
	// TODO generalise popc function to T
	return __popc(value & flavour) & T(1u);
}

///
/// \tparam T
/// \tparam HMElement
/// \tparam k
/// \tparam p
/// \tparam l
/// \tparam iHwidth
/// \tparam bijection
/// \tparam stern
/// \param bitset1
/// \param bitset2
/// \param e1
/// \param e2
/// \param iHT
/// \param flavour
/// \param iSyndrom
/// \return
template<typename T,
         typename HMElement,
         const uint32_t k,
         const uint32_t p,
         const uint32_t l,
         const uint32_t iHwidth,
         const bool bijection,
         const uint32_t stern>
__device__
bool valid_collision(uint16_t bitset1[p],
                     uint16_t bitset2[p],
                     const HMElement &e1,
                     const HMElement &e2,
                     const T *__restrict__ iHT,
                     const T flavour,
                     const T iSyndrom
) {
	// init the step counter
	uint32_t ctr = 0;

	// init the start points
	const T sp1 = e1.start,
		    sp2 = e2.start;

	// init the collision values
	T l1 = sp1, l2 = sp2;

	// init the length of the walks
	const uint32_t len1 = e1.length,
	               len2 = e2.length,
	               mlen = max(e1.length, e2.length);

	//print_bit_limb<T>(l1, 16, " init l1");
	//print_bit_limb<T>(l2, 16, " init l2");

	// TODO think of something smarter
	// first we need to advance the longer chain
	// to be on the same level as the shorter.
	if (len1 == mlen) {
		for (; ctr < mlen-len2; ctr++) {
			step<T, k, p, l, iHwidth, bijection, stern>(l1, bitset1, iHT, flavour, iSyndrom);
		}
	} else {
		for (; ctr < mlen-len1; ctr++) {
			step<T, k, p, l, iHwidth, bijection, stern>(l2, bitset2, iHT, flavour, iSyndrom);
		}
	}

	// init the helper values to recompute the collision
	T l1_tmp = l1, l2_tmp = l2;

	//printf("e1.start: %d, e2.start: %d\n", e1.start, e2.start);
	//printf("e1.len: %d, e2.len: %d, ctr: %d\n", len1, len2, ctr);
	//print_bit_limb<T>(l1, 16, " pre l1");
	//print_bit_limb<T>(l2, 16, " pre l2");

	for (; ctr < mlen; ctr++) {
		// exit after we found the collision
		if (l1 == l2)
			break;

		//print_bit_limb<T>(l1, 16, " l1");
		//print_bit_limb<T>(l2, 16, " l2");

		l1_tmp = l1;
		l2_tmp = l2;

		step<T, k, p, l, iHwidth, bijection, stern>(l1, bitset1, iHT, flavour, iSyndrom);
		step<T, k, p, l, iHwidth, bijection, stern>(l2, bitset2, iHT, flavour, iSyndrom);
	}

	//print_bit_limb<T>(l1, 16, " final l1");
	//print_bit_limb<T>(l2, 16, " final l2");
	//print_bit_limb<T>(l1_tmp, 16, " l1_tmp");
	//print_bit_limb<T>(l2_tmp, 16, " l2_tmp");

	// returns `true` if the collision happened in
	// two different functions.
	return compute_next<T>(l1_tmp, flavour) !=
	       compute_next<T>(l2_tmp, flavour);
}

template<typename T,
		 const uint64_t n,
         const uint32_t k,
         const uint32_t w,
         const uint32_t p,
         const uint32_t l,
		 const uint32_t Hwidth,
         const bool stern>
__device__ __noinline__
void PCS_recompute_solution(T *__restrict__ pde,         //
                            const T *__restrict__ pdHT,  //
                            const T *__restrict__ pdsT,  //
                            const T *__restrict__ x,     //
                            const uint16_t bitset1[p],
                            const uint16_t bitset2[p]) {
	constexpr uint32_t T_bitwidth = sizeof(T) * 8;
	const uint32_t offset = stern ? n-k : n-k-l;

	// copy the final n-k-l coordinates of the label into the
	// error vector.
	for (uint32_t i = 0; i < Hwidth; i++) {
		pde[i] = x[i];
	}

	// and set the last k+l coordinates to the bit positions
	// which result into the label with low weight.
	for (uint32_t i = 0; i < p; i++) {
		// first calculate the limb of the bit position to set
		const uint32_t pos1 = (offset + bitset1[i]) / T_bitwidth;
		const uint32_t pos2 = (offset + bitset2[i]) / T_bitwidth;

		// then calculate the bit position within the limb to set
		const uint32_t shift1 = (offset + bitset1[i]) % T_bitwidth;
		const uint32_t shift2 = (offset + bitset2[i]) % T_bitwidth;

		// and finally set the bits.
		pde[pos1] ^= (T(1) << shift1);
		pde[pos2] ^= (T(1) << shift2);
	}
}

template<typename T,
         const uint32_t n,
		 const uint32_t k,
         const uint32_t w,
		 const uint32_t p,
		 const uint32_t l,
		 const uint32_t Hwidth,
         const bool stern>
__device__
void compute_fullcollision(T *__restrict__ pde,
                           const uint16_t bitset1[p],     //
                           const uint16_t bitset2[p],     //
                           const T *__restrict__ pdHT,    //
                           const T  *__restrict__ pds,    //
                           uint32_t *__restrict__ compare //
) {
	uint32_t weight = 0;
	T x[Hwidth] = {0};

	// recalc the weight of the element on the full length
	#pragma unroll
	for (uint32_t i = 0; i < Hwidth; i++) {
		x[i] = pds[i];

		#pragma unroll
		for (uint32_t j = 0; j < p; j++) {
			x[i] ^= pdHT[bitset1[j] * Hwidth + i];
			x[i] ^= pdHT[bitset2[j] * Hwidth + i];
		}

		weight += (uint32_t) __popc(x[i]);// TODO generalize to T
	}

	//printf("bitset1[0]=%d, \tl1=%u, \tl1_old=%u\n", bitset1[0], l1, l1_tmp);
	//printf("bitset2[0]=%d, \tl2=%u, \tl2_old=%u\n", bitset2[0], l2, l2_tmp);
	//print_bit_vector(&pdHT[bitset1[0] * Hwidth], Hwidth);
	//print_bit_vector(&pdHT[bitset2[0] * Hwidth], Hwidth);
	//print_bit_vector(pds, Hwidth);
	//print_bit_limb(iSyndrom, l);
	//print_bit_vector(x, Hwidth);

	// check if the element fulfills the weight restriction
	// on the full n-k coordinates
	if (unlikely(weight <= w-2*p)) {
		// make the atomic check if we really found a solution.
		if (0 != atomicCAS(compare, 0, 1))
			return;

		// recompute the solution
		return PCS_recompute_solution<T,n,k,w,p,l,Hwidth,stern>(pde, pdHT, pds, (T *)x, bitset1, bitset2);
	}
}

/// Main kernel.
/// \tparam T base limb type of the parity check matrix
/// \tparam lT minimal data type to represent `l` bit
/// \tparam HM HashMap type
/// \tparam n code length
/// \tparam k code dimension
/// \tparam w code weight
/// \tparam p number of bit positions or number row to add up
/// \tparam l number of bits to match on
/// \tparam Hwidth number of `T` limbs in the parity check matrix HT (must be passed as transposed)
/// \tparam iHwidth number `lT` limb needed to hold the `l` part of the parity check matrix of one row. Most likely == 1
/// \tparam llimbpos limb position of the `l` part in each row of the parity check matrix `HT`
/// \tparam bijection enforce the internal hash function to be a perfect (but slow) bijection
/// \tparam stern match on k bits. If set to `false` (k+l)
/// \tparam nr_iters
/// \tparam walk_length
/// \tparam D bits of distinguished point
/// \param pde output: reconstructed error vector
/// \param pdHT input: transposed parity check matrix of size n-k \times k
/// \param pdiHT2 input: same as `pdHT` but only the `l` part of each row (aligend to zero)
/// \param pds2 input: syndrome
/// \param iSyndrom input: `l` part of the syndrome
/// \param compare output: set to 1 if the solution was found, 0 otherwise
template<typename T, typename lT, class HM,
         const uint32_t n,
         const uint32_t k,
         const uint32_t w,
         const uint32_t p,
         const uint32_t l,
         const uint32_t Hwidth,
         const uint32_t iHwidth,
         const uint32_t llimbpos,
         const bool bijection,
         const bool stern,
         const uint32_t nr_iters,
         const uint32_t nr_flavours,
         const uint32_t walk_length,
         const T D>
__global__ 
void KernelPCS(T *__restrict__ pde,            //
               const T  *__restrict__ pdHT,    //
               const lT *__restrict__ pdiHT,   //
               const T  *__restrict__ pds,     //
               const lT iSyndrom,              //
               lT flavour,               	   //
               HM *hm,                         //
               curandState *__restrict__ state,//
               uint32_t *__restrict__ compare) //
{
	static_assert(l < sizeof(T)*8);
	static_assert(D < sizeof(T)*8);

	//constexpr uint32_t kl_length  = stern ? k : k+l;// number of rows in the matrix
	constexpr lT lMask = (T(1) << l) - 1;   // bit mask to check the first `l` bits
	constexpr lT DMask = (T(1) << D) - 1;   // bit mask to check the first `D` bits

	// internal types
	using HMElement     = typename HM::data_type;
	using hm_index_type = typename HM::index_type;

	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	//	// TODO copy pdHT into shared mem
	//	// first copy everything into the shared memory to speed up things
	//	__shared__ lT pdiHT[iHwidth*kl_length];
	//	__shared__ T pds[Hwidth];
	//
	//	// make sure that the shared mem is actually filled completely
	//	assert(blockDim.x >= kl_length);
	//
	//	// copy the syndrome and the parity check matrix into the shared memory
	//	if (threadIdx.x < kl_length) {
	//		// NOTE: see `cuLeeBrickellNoMem` to understand what im doing
	//		#pragma unroll
	//		for (uint32_t i = 0; i < iHwidth; i++) {
	//			pdiHT[threadIdx.x + threadIdx.x*i] = pdiHT2[threadIdx.x + threadIdx.x*i];
	//		}
	//	}
	//	if (threadIdx.x < Hwidth) {
	//		pds[threadIdx.x] = pds2[threadIdx.x];
	//	}
	//
	//	// the shared memory is allocated for every block. So we have to make sure that
	//	// all threads in a block a synchronized again
	//	__syncthreads();

	lT      sp, // start point
			kv; // value each thread operators on
	// helper variables: representing a value of length `n` with weight `p`
	uint16_t bitset1[p], bitset2[p];
	//for (uint32_t flov = 0; flov < nr_flavours; flov++) {
	//	flavour = lT(curand(&state[0])) & lMask;
		for (uint32_t iters = 0; iters < nr_iters;) {
			// TODO is this really needed?
			hm->clear();

			// length of the chain which is going to be computed
			uint32_t ctr = 0;

			// init the start point and the value to match on
			sp = lT(curand(&state[idx])) & lMask;
			kv = sp;

			// start the collision search
			while ((ctr < walk_length) && !is_distinguished<lT, D>(kv)) {
				step<lT, k, p, l, iHwidth, bijection, stern>(kv, bitset1, pdiHT, flavour, iSyndrom);
				// printf("bitset1[0]=%d, \tl1=%u\n", bitset1[0], kv);
				ctr += 1;
			};

			// make sure we really have a distinguished point
			if (!is_distinguished<lT, D>(kv))
				continue;

			// recompute the collision or insert in hashmap
			// first create an object which is going to be inserted
			// in the hashmap.
			const HMElement hme1 = HMElement(sp, ctr);
			ASSERT(hme1.start != lT(-1));

			// helper lambda function
			//auto internal_hash_function = [](const lT e) {
			//	return e >> D;
			//};

			// calculate the index within in the hashmap,
			// where the element should be inserted
			const hm_index_type hm_index = kv >> D;

			if (hm->probe_internal(hm_index)) {
				// in this case there is something in the hashmap.
				// So we have to check for a collision.
				// Receive the element from the hashmap
				const HMElement hme2 = hm->__internal_hashmap_array[hm_index];
				ASSERT(hme2.start != lT(-1));

				//hme1.print(); hme2.print();

				// NOTE: this overwrites `bitset1` and `bitset2`
				if (valid_collision<lT, HMElement, k, p, l, iHwidth, bijection, stern>(bitset1, bitset2, hme1, hme2, pdiHT, flavour, iSyndrom)) {
					// after we successfully recomputed the collision and made sure
					// that it is in different functions, we are recomuting the collision
					// on the full n-k coordinates
					compute_fullcollision<T, n, k, w, p, l, Hwidth, stern>(pde, bitset1, bitset2, pdHT, (const T *) pds, compare);
				}
			}

			// In this case no element is currently at this position in the hashmap.
			// Therefore, we can insert something into the hashmap.
			hm->insert(hme1, hm_index);

		    // in this case we did not find a collision, in other words this run was "useless"
			iters -= 1;
		}
	//}
}

/// \tparam T
/// \tparam config
/// \param par
/// \return
template<typename T, const ConfigPCS<T> &config>
uint64_t ThreadPCS(void *par) {
	DecodingInstance *params     = (DecodingInstance *)par;
	// TODO this is needed for the HM which can nly work with this LogTypeTemplate<config.l>;
	using lT              = uint32_t;

	// CPU stuff, allocate all needed variables/matrices.
	mzd_t *e            = mzd_init(1, config.n);
	mzd_t *wH           = matrix_init(config.n-config.k, config.n+1);
	mzd_t *tmp          = matrix_concat(nullptr, params->A, params->syndromT);
	mzd_copy(wH, tmp);
	mzd_free(tmp);
	mzd_t *wHT          = mzd_init(wH->ncols, wH->nrows);
	mzd_t *H           =  mzd_init(config.n-config.k, config.kl_length+1);  // plus 1 for the syndrome
	mzd_t *HT           = mzd_init(H->ncols, H->nrows);
	mzd_t *iHT          = mzd_init(config.k, config.l);             // k \times l matrix containing only the `l` part of the transposed parity check matrix

	// current permutation of the parity check matrix
	mzp_t *permutation  = mzp_init(config.n);

	// Allocate and int the `exit` flag. meaning: if this value is somehow set
	// to one, we found a solution
	uint32_t Compare = 0, *dCompare = nullptr;
	checkCudaErrors(cudaMalloc(&dCompare, sizeof(uint32_t)));
	checkCudaErrors(cudaMemcpy(dCompare, &Compare, sizeof(uint32_t), cudaMemcpyHostToDevice));

	// parity check matrix in the gpu.
	// NOTE: The parity check matrix on the gpu is probably based on `uint32_t` limbs, whereas the M4RI library
	//      is based on `uint64_t` limbs. We need do take care of this by copy only row by row to the
	//      graphic card.
	// NOTE: The parity check matrix consists of n+1 columns. We need the last columns to append the syndrome,
	//      to apply the gaussian elimination to it.
	// BUG: I waste some limbs by simply allocating 2*limb_in_m4ri.
	T* pdHT;
	// only the last `l` bit of the parity check matrix, extracted to fastly compute matches on `l` bits.
	lT *pdiHT;  //
	checkCudaErrors(cudaMalloc(&pdHT,  config.gpu_matrix_width * config.cpu_matrix_rows * sizeof(T)));
	checkCudaErrors(cudaMalloc(&pdiHT, config.iHwidth * config.cpu_matrix_rows * sizeof(lT)));

	// just set the syndrome to point to the last column (row in this case, because the matrix is transposed)
	// of the parity check matrix.
	T* pdsT                 = &pdHT[config.cpu_gpu_matrix_factor * (config.cpu_matrix_rows-1) * config.cpu_matrix_width];

	// Allocate the output error.
	thrust::device_vector<T> de(config.cpu_gpu_matrix_factor *params->A->width);
	T* pde                  = thrust::raw_pointer_cast(de.data());

	// Allocate the Hashmap for the PCS search
	// TODO managed?
	using HMElement = cuHashMapElement<lT>;
	using HM = cuHashMap<HMElement, config.HashMapSize, PCSHashFkt<HMElement>>;
	HM *hm = nullptr;
	checkCudaErrors(cudaMallocManaged(&hm, sizeof(HM)));

	// timing stuff
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float kernel_runtime = 0., total_kernel_runtime = 0.;
	double time = (double)clock();
	uint64_t loops = 0;

	// internal matrix stuff
	customMatrixData *matrix_data = init_matrix_data(wH->ncols);

	// cuda rng stuff
	curandState * dState; const uint64_t seed = rand();
	checkCudaErrors(cudaMalloc(&dState, params->blocks * params->tpb * sizeof(curandState)));
	InitRandomNumberGenerator<<<params->blocks, params->tpb>>>(dState, seed);

	// as long, we didn't find anything lets run it.
	while (!Compare) {
		loops += 1;

		// apply the gaussian elimination
		matrix_create_random_permutation(wH, wHT, permutation);
		matrix_echelonize_partial_plusfix(wH, config.m4ri_k, config.n-config.k, matrix_data, 0, config.n-config.k, 0, permutation);
		mzd_submatrix(H, wH, 0, config.n-config.k, config.n-config.k, config.n+1);
		matrix_transpose(HT, H);
		mzd_submatrix(iHT, HT, 0, config.n-config.k-config.l, config.k, config.n-config.k);
		// end of the gaussian elimination phase

		// copy the transposed partial parity check matrix to the gpu
		MatrixToCuda<T>(pdHT, HT);
		DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError()));
		DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize()));

		// same as above only for the `l` part
		MatrixToCudaReduce<lT>(pdiHT, iHT, config.iHwidth);
		DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError()));
		DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize()));

		// also extract the syndrome
		lT iSyndrom = exctract_l_bit<lT, config.n, config.k, config.l>(wH);

		//mzd_print(wH);
		//mzd_print(HT);
		//print_matrix<T><<<1,1>>>(pdHT, k+1, config.gpu_matrix_width);
		//print_matrix<T><<<1,1>>>(pdiHT, k, config.iHwidth);
		//print_matrix<T><<<1,1>>>(pdsT, 1, config.gpu_matrix_width);
		//std::cout << "\n" << std::flush;
		//std::cout << iSyndrom << " iSyndrom\n";
		//break;

		kernel_runtime = 0.;
		cudaEventRecord(start);

		for (uint32_t flav = 0; flav < config.nr_flavors; flav++) {
			lT flavor = fastrandombits<lT, config.l>();

			// start the actual kernel
			KernelPCS<T, lT, HM,
			          config.n, config.k, config.w, config.p, config.l,
			          config.gpu_matrix_width, config.iHwidth, config.llimbpos,
			          config.bijection, config.stern, config.nr_iters,
			          config.nr_flavors, config.walk_length, config.D>
			        <<<params->blocks, params->tpb>>>(pde, pdHT, pdiHT, pdsT, iSyndrom, flavor, hm, dState, dCompare);

			// simple error checking.
			DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError()));
			DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize()));
		}
		// printing hashmap
		//print_hashmap<HM><<<1,1>>>(hm);
		//DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError());)
		//DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize());)
		//return 0;

		// this is needed to check how much time was spend on the gpu compared
		// to the gpu.
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&kernel_runtime, start, stop);
		total_kernel_runtime += kernel_runtime;

		// copy back the `is a solution found flag`
		// OPTIMIZE: Instead of checking this every loop, let the algorithm run for a few loops,
		// check then if we found the solution. This ca be done, because once set. Wow it made it slower
		checkCudaErrors(cudaMemcpy(&Compare, dCompare, sizeof(bool), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaDeviceSynchronize());

		// Check if we found the solution
		if (Compare) {
			// TODO put everything into custom function to reduce i cache misses
			// apply back perm
			mzd_t *tmpe = mzd_init(1, config.n);
			CudaToMatrix<T>(tmpe, de);
			checkCudaErrors(cudaDeviceSynchronize());

			for (uint32_t i = 0; i < config.n; i++) {
				mzd_write_bit(e, 0, permutation->values[i], mzd_read_bit(tmpe, 0, i));
			}
			mzd_free(tmpe);
		}

		// print every X loops a status update of the algorithm
		if ((loops%config.print_loops) == 0) {
			config.print_loop(loops, ((double)clock() -  time)/CLOCKS_PER_SEC, total_kernel_runtime/1e3);
		}
	}

	config.print_loop(loops, ((double)clock() -  time)/CLOCKS_PER_SEC, total_kernel_runtime/1e3);
#ifdef EXPECTED_PERMUTATIONS
	const uint64_t frac = uint64_t(EXPECTED_PERMUTATIONS)/uint64_t(loops);
	const uint64_t frac_time128 = frac * time;
	const uint64_t frac_time = uint64_t(frac_time128);
	std::cout << ", exptime: " << frac_time << "s";
#endif

	std::cout << "\n";
	bool ret = checkCorrectness(e, params->A, params->syndromT, config.w);
	// CPU
	mzd_free(wH);
	mzd_free(wHT);
	mzd_free(HT);
	mzd_free(iHT);
	mzp_free(permutation);
	mzd_free(e);

	checkCudaErrors(cudaFree(dCompare));
	checkCudaErrors(cudaFree(pdHT));
	checkCudaErrors(cudaFree(pdiHT));

	return loops;
}

template<typename T, const ConfigPCS<T> &config>
int CudaPCS() {
	DecodingInstance params{h, s, n, k, config.cuda_threads};
	config.print();
	params.print();

	const uint64_t loops = ThreadPCS<T, config>(&params);
	cudaDeviceReset();
	return loops;
}
#endif
