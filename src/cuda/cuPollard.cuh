#ifndef DECODING_CUDAPOLLARD
#define DECODING_CUDAPOLLARD

#include <stdint.h>
#include <curand_kernel.h>

#include "cuhelper.cuh"
#include "cuMisc.cuh"
#include "cuHashMap.cuh"
#include "helper.h"


/// configuration struct containing all important flags.
/// \tparam T	base limb type of everything on the gpu.
template<typename T>
class ConfigPollard {
private:
	/// Construction of the algorithm
	///        n-k-l
	///  ◄───────────►   l   ◄───         k       ───────────►
	///
	/// ┌─────────────┬──────┬──────────────────────────────────┐  ┌─────┐  ▲
	/// │             │      │                                  │  │     │  │
	/// │     I       │   $  │           H1                     │  │     │  │
	/// │             │      │                                  │  │ s1  │  │n-k-l
	/// │             │      │                                  │  │     │  ▼
	/// ├─────────────┼──────┼──────────────────────────────────┤  ├─────┤
	/// │    0        │  $   │           H2                     │  │     │  ▲
	/// │             │      │                                  │  │ s2  │  │l
	/// └─────────────┴──────┴──────────────────────────────────┘  └─────┘  ▼
	///
	/// ┌─────────────┬──────┬──────────────────────────────────┐
	/// │             │   0  │           e2                     │
	/// │     e1      │      │                                  │
	/// └─────────────┴──────┴──────────────────────────────────┘
	///                                        ▲
	///                                        │
	///                                        │
	///                                        │
	///                                        │
	///                        ┌─────────┐     │     ┌─────────┐
	///                        │         │     │     │         │
	///                  t1 -> │ f_1(x)  │ ◄───┴───► │ f_2(x)  │ <- t1
	///                        │         │           │         │
	///                  t2 -> │         │           │         │ <- t2
	///                        │         │           │         │
	///                  t3 -> │         │           │         │ <- t3
	///                        │         │           │         │
	///                  tx -> │ f_1(z)  │           │ f_2(z)  │ <- tx
	///                        │         │           │         │
	///                        └─────────┘           └─────────┘
	// disable the default constructor
	constexpr ConfigPollard() : n(0), k(0), w(0), l(0), p(0) {}
public:
	// instance parameters
	const uint32_t n;   // code length
	const uint32_t k;   // code size
	const uint32_t w;   // code weight
	const uint32_t p;   // weight in the base lists. In this case the weight which is
						// mapped to by the collision functions
	const uint32_t l;   // bits to match on by the two collision functions
	
	// opt m4ri c
	const uint32_t gaus_c = 50;

	// if set to `true` the algorithm uses a perfect bijection to calculate the
	// hashing function.
	const bool bijection            = false;

	// if set to true: stern is used
	// if set to false: dumer is used
	const bool stern                = true;

	// length of the k part
	const uint32_t kl_length        = stern ? k : k+l;

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
	const uint32_t m4ri_k = matrix_opt_k(n - k, MATRIX_AVX_PADDING(n));

	// if this value is set to a different value then -1 the program will end the
	// computation after `loops` permutation, regardless if it found the solution or not
#ifdef USE_LOOPS
	const uint64_t loops        = USE_LOOPS;
#else
	const uint64_t loops        = uint64_t(-1);
#endif

	// print every `print_loops` iterations a status update of the program
#ifdef PRINT_LOOPS
	const uint64_t print_loops  = PRINT_LOOPS;
#else
	const uint64_t print_loops  = 1000;
#endif

	// Nr of stream per thread.
	// currently unused
	const uint32_t nr_streams   = 1;
	// number of different cpu threads working on different permutations.
	// currently unused
	const uint32_t nr_threads   = 1;

	// nr of concurrent random walk we start
	const uint32_t nr_walks     = 1u << l;

	const uint32_t nr_iters     = 1;

	/// standard constructor
	constexpr ConfigPollard(const uint32_t n,
	                        const uint32_t k,
	                        const uint32_t l,
	                        const uint32_t w,
	                        const uint32_t p,
	                        const bool bijection=true,
	                        const bool stern=true,
	                        const uint32_t nr_iters=1,
							const uint32_t nr_walks=-1) :
			n(n), k(k), l(l), w(w), p(p), bijection(bijection), stern(stern),
			nr_iters(uint32_t(nr_iters)), nr_walks(uint32_t(nr_walks == -1 ? 1u << l : nr_walks)) {};

	constexpr ConfigPollard(ConfigPollard &c) = default;

	// returns the expected number of iterations
	constexpr uint64_t Loops() const {
#ifdef EXPECTED_PERMUTATIONS
		return EXPECTED_PERMUTATIONS;
#else
		//return bc(n, w)/( bc(n-k-l, w-2*p) * bc(k, 2*p));
		return bc(n, w)/( bc(n-k-l, w-2*p) * bc(k/2, p) * bc(k/2, p));
#endif
	}

	// print some information about the configuration
	void print() const {
		std::cout << "cuPollard: n: " << unsigned(n)
		          << ", k: " << unsigned (k)
		          << ", l: " << unsigned (l)
		          << ", n-k:" << unsigned (n-k)
		          << ", p: " << unsigned (p)
		          << ", cpu_gpu_matrix_factor: " << cpu_gpu_matrix_factor
		          << ", gpu_matrix_width: " << gpu_matrix_width
		          << ", cpu_matrix_width: " << cpu_matrix_width
		          << ", cpu_matrix_rows: " << cpu_matrix_rows
		          << ", m4ri_k: " << m4ri_k
		          << ", llimbpos: " << llimbpos
		          << ", iHwidth: " << iHwidth
				  << ", nr_walks: " << nr_walks
		          << ", nr_iters: " << nr_iters
		          << ", bijection: " << bijection
				  << ", stern: " << stern
		          << ", n-k-l: " << n-k-l 
				  << ", loops: " << loops
				  << ", print_loops: " << print_loops
				  << "\n";
	}

	/// \param loops
	/// \param time
	/// \param total_kernel_runtime
	void print_loop(const uint64_t loops, const double time, const float total_kernel_runtime) const {
		const double expteded_time = Loops()/loops*time;

		std::cout << "runtime: "  << time << "s"
		          << ", loops: " << loops << "(" << Loops() <<")=" << loops / Loops() << "x"
		          << ", total kernel runtime: " << total_kernel_runtime<< "s (" << total_kernel_runtime/time*100. << "%)"
		          << ", loops/kernel s: " << loops/total_kernel_runtime
		          << ", loops/s: "<< loops/time
		          << ", E[RT]=" << expteded_time << "s(" << time/expteded_time << "x)"
		          << "\r" << std::flush;//"\n";
	}
};



// TODO remove?
///
/// \tparam T
/// \tparam n
/// \tparam k
/// \tparam l
/// \param wH
/// \return
template<typename T, const uint32_t n, const uint32_t k, const uint32_t l>
T exctract_l_bit(const mzd_t *__restrict__ wH) {
	T ret = 0;
	//mzd_print((mzd_t *)wH);
	for (uint32_t i = n-k-l; i < n-k; i++) {
		uint32_t bit = mzd_read_bit(wH, i, n);
		ret ^= (bit << (i-(n-k-l)));
	}
	return ret;
}


/// just computes n over 3
/// \tparam T
/// \param n
/// \return
template<typename T>
__device__
inline T max_bc_helper3(const T n) {
	constexpr T re = 6; // Assumes k = 3

	assert(n >= 3);

	const T a = n*(n-1)*(n-2);

	return a/re;
}

/// final opt version
/// \param a input number
/// \param n max bits
/// \param k max weight
/// \return maximal ctr < n, s.t. k+ctr over k <= a
template<typename T, const uint32_t k>
__device__
inline T opt_max_bc(const T a, const uint32_t n) {
	if constexpr(k == 1) {
		return n - k - a;
	}

	if constexpr (k == 2) {
#ifdef __CUDA_ARCH__
		//const uint32_t t = (sqrtf((a<<3) + 1u) + 1.)/2.;
		const uint32_t t = ((uint32_t)(sqrtf((a<<3) + 1u) + 1u)) >> 1u;
		//const uint32_t t =  uint32_t(__builtin_sqrt((a<<3) + 1) + 1) >> 1;
#else
		const uint32_t t =  uint32_t(__builtin_sqrt(8 * a + 1) + 1) >> 1;
#endif
		return n - t - 1;
	}

	if constexpr (k == 3) {
		constexpr uint32_t 	bitsize = sizeof(T) * 8;
		// const uint32_t k	1 = k-1;

#ifdef __CUDA_ARCH__
		int ctr = bitsize -	 __clz(a|1);
#else
		int ctr = bitsize -	 __builtin_clz(a|1);
#endif
		T tmp = max_bc_helper3(k+ctr);
		T c = k+ctr, d = ctr;
		tmp = (tmp*d) / c;
		while ((tmp > a) && (ctr > 0)) {
			ctr -= 1;
			d -= 1;
			c -= 1;
			tmp = (tmp*d) / c;
		}

		return  n - ctr - k;
	}

	// will never happen
	return 0;
}

/// optimized unrolled version
/// same as the function above, but returning the indicies of the set bits.
/// \tparam T base type
/// \tparam n max bit length 
/// \param a input number
/// \return the a`th element in lexicographical order with weight w
template<typename T, const uint32_t n, const uint32_t w>
__device__
uint16_t opt_bijection_fkt(uint16_t ret[w], const T label) {
	uint64_t wn = n;

	// please compiler make this fast.
	constexpr T binom = bc(n, w);
	T a = label % binom;

	if constexpr (w == 1) {
		wn -= opt_max_bc<T, 1>(a, wn);

		// decrement the counter and set the bit accordingly
		wn -= 1;
		ret[0] = wn;

		ASSERT(wn < n);
		// printf("%d %d %d %d %d\n", wn, n, w, a, label);
	}

	if constexpr (w == 2) {
		// w == 2
		wn -= opt_max_bc<T, 2>(a, wn);
		a -= ((wn-1) *(wn-2)) / 2;

		// decrement the counter and set the bit accordingly
		wn -= 1;
		ret[0] = wn;
		ASSERT(wn < n);

		wn -= opt_max_bc<T, 1>(a, wn);

		// decrement the counter and set the bit accordingly
		wn -= 1;
		ret[1] = wn;
		ASSERT(wn < n);
	}

	if constexpr (w == 3) {
		// w == 3
		const T tmp = opt_max_bc<T, 3>(a, wn);
		wn -= tmp;
		a -= ((wn-1) * (wn-2) * (wn-3)) / 6;

		//printf("idx: %d: %d %d %d\n", threadIdx.x, wn, tmp, a);
		// decrement the counter and set the bit accordingly
		wn -= 1;
		ret[0] = wn;
		ASSERT(wn < n);
		ASSERT(a < binom);

		// w == 2
		wn -= opt_max_bc<T, 2>(a, wn);
		a -= ((wn-1) * (wn-2)) / 2;

		//printf("idx: %d: %d\n", threadIdx.x, wn);
		// decrement the counter and set the bit accordingly
		wn -= 1;
		ret[1] = wn;
		ASSERT(wn < n);

		wn -= opt_max_bc<T, 1>(a, wn);

		// decrement the counter and set the bit accordingly
		wn -= 1;
		ret[2] = wn;
		ASSERT(wn < n);
	}

	// return a&1;
	return 0;
}

/// \tparam T minimal data type to hold `l` bits
/// \tparam p number of bit positions (column position) to extract form the label
/// \tparam l number of bits used in `label`. Or the number of bits extracted from
/// 			the parity check matrix (sum of columns).
/// \param bitpos output: the bit positions which correspond to a sum of rows (columns if the matrix is not transposed)
/// \param label input: random label
/// \return the parity of the next function:
///             0: just sum of row
///             1: sum of rows + syndrome
template<typename T,
         const uint32_t k,
         const uint32_t p,
         const uint32_t l>
__device__ __forceinline__ uint16_t hash1(uint16_t bitpos[p], const T label) noexcept {
	// only usable if p = 1
	static_assert(l < 32);
	static_assert(p == 1);

	//constexpr T mask1 = ((T(1) << l) - 1);
	bitpos[0] = label % k;

	// use the first bit as a selector for the next functions
	return label & 1;
}

///
/// \tparam T
/// \tparam k
/// \tparam p
/// \tparam l
/// \param bitpos
/// \param label
/// \return
template<typename T,
         const uint32_t k,
         const uint32_t p,
         const uint32_t l>
__device__ __forceinline__ 
T hash22(uint16_t bitpos[p], const T label) noexcept {
	// only usable if p = 2
	static_assert(l < 32);
	static_assert(p==2);

	//constexpr uint32_t lprime = (l-1)/2;
	//constexpr uint32_t lprime2 = l+1;
	constexpr uint32_t k1 = k-1;
	bitpos[0] = label % k1;
	const T label2 = (label-bitpos[0]) / k1;
	bitpos[1] = (label2 % (k-bitpos[0])) + bitpos[0];

	return label & 1;
}

/// same as `hash1` but specially optimised for p == 2
/// \tparam T minimal data type to hold `l` bitsl
/// \tparam p must be 2, ist statically tested
/// \tparam l number of bits available in `label`
/// \param bitpos output: random positions of the rows of the parity check matrix to sum up
/// \param label input: random label
/// \return the parity of the next function:
///             0: just sum of row
///             1: sum of rows + syndrome
template<typename T,
         const uint32_t k,
         const uint32_t p,
         const uint32_t l>
__device__ __forceinline__ 
uint16_t hash2(uint16_t bitpos[p], const T label) noexcept {
	static_assert(l < 64);
	static_assert(p == 2);

	// TODO explain
	constexpr static uint32_t lprime= (l-1u)/2u;

	// only usable if p = 2
	constexpr T mask1 = (T(1) << (1+lprime)) - 1;
	constexpr T mask2 = (T(1) << (  lprime)) - 1;

	//constexpr T mod = const_log(bc(k+l, p));
	bitpos[0] = ((label & mask1) >> 1) % k;
	bitpos[1] = ( label >> (l-lprime)) % k;

	//if (bitpos[0] == bitpos[1])
	//	bitpos[1] = (bitpos[1] + 1) % mod;
	const bool t = bitpos[0] == bitpos[1];
	bitpos[1] += t;

	// use the first bit as a selector for the next function, if it's
	// either f1 or f2
	return label & 1;
}

/// same as `hash1` and `hash2` but only specially optimised for
/// weight `p` == 3
/// \tparam T minimal data type to hold `l` bits.
/// \tparam p must be == 3. Ist statically checked
/// \tparam l number of bits available in the `label`
/// \param bitpos output: random position of rows in the parity check matrix to sum up
/// \param label input: randomly calculated label
/// \return the parity of the next function:
///             0: just sum of row
///             1: sum of rows + syndrome
template<typename T,
         const uint32_t k ,
         const uint32_t p,
         const uint32_t l>
__device__ __forceinline__
uint16_t hash3(uint16_t bitpos[p], const T label) noexcept {
	// only usable if p = 3
	static_assert(l < 64);
	static_assert(p == 3);

	constexpr uint32_t ws = (l-1)/3;
	constexpr T mask1 = (T(1) << ws) - 1;
	//	constexpr T mask2 = (T(1) << (1+2*ws)) - 1;
	//	constexpr T mod = k;// bc(k, p);
	//	bitpos[0] = ((label & mask1) >> 1) % mod;
	//	bitpos[1] = ((label & mask2) >> ws) % mod;
	//	bitpos[2] = (label >> (l-ws)) % mod;

	//	if (bitpos[0] == bitpos[1])
	//		bitpos[1] = (bitpos[1] + 1) % mod;
	//
	//	if (bitpos[1] == bitpos[2])
	//		bitpos[2] = (bitpos[2] + 1) % mod;

	constexpr T off = k+l- (1u << ws);
	bitpos[0] = (label >>     1 ) & mask1;
	bitpos[1] = (label >> (ws+1))& mask1 + off/2;
	bitpos[2] = (label >> (l-ws))& mask1 + off;

	const bool t = bitpos[0] == bitpos[1];
	bitpos[1] += t;
	const bool u = bitpos[1] == bitpos[2];
	bitpos[2] += u;

	// use the first bit as an selector for the next function.
	return label & 1;
}

/// Calculates the matrix vector multiplication `H` \times `e`, but only
/// on the last `l` columns of the matrix `iHT`.
/// NOTE: `iHT` must be given in transposed form. So the matrix-vector multiplication
/// is simply a addition of rows.
/// \tparam T minimal datatype to hold `l` bits
/// \tparam p number of rows to sum up
/// \tparam width == 1, or == 2 if `T` is `uint32_t`
/// \param label output: He
/// \param bitpos input: positions of the rows to sum up
/// \param iHT intput: last `l` bits of the `k` rows if the transposed parity check matrix.
template<typename T,
         const uint32_t k,
         const uint32_t p,
         const uint32_t width>
__device__ __forceinline__ 
void calcHe(T &label, const uint16_t bitpos[p], const T *__restrict__ iHT) noexcept {
	assert(bitpos[0] < k);
	if constexpr(p == 2) { assert(bitpos[1] < k); }
	if constexpr(p == 3) { assert(bitpos[2] < k); }

	#pragma unroll
	for (uint32_t i = 0; i < p; ++i) {
		label ^= iHT[bitpos[i]*width];
	}
}


/// \tparam T minimal data type to hold `l` bits
/// \tparam p number of rows to sum up.
/// \tparam l number of bits to match on.
/// \tparam width number of limbs needed to hold the `l` part. Most likely to be 1
/// \param label output:
/// \param bitpos output:
/// \param iHT input:
/// \param flavour input: additional randomization value
/// \param iSyndrom input: last `l` coordinated of the syndrome.
template<typename T,
         const uint32_t k,
         const uint32_t p,
         const uint32_t l,
         const uint32_t width,
         const bool bijection,
         const uint32_t stern>
__device__  void step(T &label,
		uint16_t bitpos[p],
		const T *__restrict__ iHT,
		const T flavour,
		const T iSyndrom) noexcept {
	// NOTE: this value is set by one of the hash functions to either
	// zero or one. If it's set to zero, it means that the syndrome is not calculated
	// in.
	uint16_t next = 0;

	constexpr uint32_t collision_bits = stern ? k/2 : (k+l)/2;
	if constexpr (bijection) {
		// next = bijection_fkt<T, collision_bits, p>(bitpos, label);
		opt_bijection_fkt<T, collision_bits, p>(bitpos, label);
	} else {
		if constexpr (p == 1u) {
			hash1<T, collision_bits, p, l>(bitpos, label);
		} else if constexpr (p == 2u) {
			hash22<T, collision_bits, p, l>(bitpos, label);
		} else if constexpr (p == 3u) {
			hash3<T, collision_bits, p, l>(bitpos, label);
		}
		// bigger p is not supported.
	}
	//	assert(next <= 1);
	//	if (next == 0) {
	//		f1<T, width>(label, bitpos, iHT, flavour);
	//	} else if (next == 1) {
	//		f2<T, width>(label, bitpos, iHT, flavour, iSyndrom);
	//	}

	// same code as above but without an if divergence
	next = __popc(flavour&label)&1u;


	assert(next <= 1);
	const uint32_t off = next*(collision_bits);

	#pragma unroll
	for (uint32_t i = 0; i < p; i++) {
		bitpos[i] += off;
	}

	label = next*iSyndrom;  // NOTE: if next is zero f1 is called
	label ^= flavour;       // randomize the trail even further
	calcHe<T, 2*collision_bits, p, width>(label, bitpos, iHT);
}

/// Currently unused
/// regenerate a new random flavor
/// \tparam T minimal data type to represent `l` bits
/// \tparam l
/// \param state rng
/// \return random `l` bits aligned to zero
template<typename T, const uint32_t l>
__device__ __forceinline__ T flavour(curandState *__restrict__ state) noexcept {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	T r = T(curand(&state[idx])) & ((1 << (l-1)) -1);
	r <<= 1;
	return r;
}

/// Main kernel.
/// \tparam T base limb type of the parity check matrix
/// \tparam lT minimal data type to represent `l` bit
/// \tparam n code length
/// \tparam k code dimension
/// \tparam w code weight
/// \tparam p number of bit positions or number row to add up
/// \tparam l number of bits to match on
/// \tparam Hwidth number of `T` limbs in the parity check matrix HT (must be passed as transposed)
/// \tparam iHwidth number `lT` limb needed to hold the `l` part of the parity check matrix of one row. Most likely == 1
/// \tparam llimbpos limb position of the `l` part in each row of the parity check matrix `HT`
/// \param pde output: reconstructed error vector
/// \param pdHT input: transposed parity check matrix of size n-k \times k
/// \param pdiHT2 input: same as `pdHT` but only the `l` part of each row (aligend to zero)
/// \param pds2 input: syndrome
/// \param iSyndrom input: `l` part of the syndrome
/// \param compare output: set to 1 if the solution was found, 0 otherwise
template<typename T, typename lT,
         const uint64_t n,
         const uint32_t k,
         const uint32_t w,
         const uint32_t p,
         const uint32_t l,
         const uint32_t Hwidth,
         const uint32_t iHwidth,
         const uint32_t llimbpos,
         const bool bijection,
         const bool stern,
         const uint32_t nr_iters>
__global__ void KernelPollard(T *__restrict__ pde,          //
                            const T  *__restrict__ pdHT,    //
                            const lT *__restrict__ pdiHT2,  //
                            const T  *__restrict__ pds2,    //
                            const lT iSyndrom,              //
                            curandState *__restrict__ state,//
                            uint32_t *__restrict__ compare) //
{
	// going to be the number of rows in the matrix
	constexpr uint32_t kl_length  = stern ? k : k+l;
	constexpr uint32_t T_bitwidth = sizeof(T) * 8;
	// int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// first copy everything into the shared memory to speed up things
	__shared__ lT pdiHT[iHwidth*kl_length];
	__shared__ T pds[Hwidth];

	// make sure that the shared mem is actually filled completely
	assert(blockDim.x >= kl_length);

	// copy the syndrome and the parity check matrix into the shared memory
	if (threadIdx.x < kl_length) {
		// NOTE: see `cuLeeBrickellNoMem` to understand what im doing
		#pragma unroll
		for (uint32_t i = 0; i < iHwidth; i++) {
			pdiHT[threadIdx.x + threadIdx.x*i] = pdiHT2[threadIdx.x + threadIdx.x*i];
		}
	}

	if (threadIdx.x < Hwidth) {
		pds[threadIdx.x] = pds2[threadIdx.x];
	}

	// the shared memory is allocated for every block. So we have to make sure that
	// all threads in a block a synchronized again
	__syncthreads();

	// NOMENCLATURE: every variable contain `1` belongs to the slow
	// kangaroo and every variable containing a `2` belongs to the
	// fast kangaroo
	lT      sp, r,          // start point and flavor
	        l1,l2,          // slow and fast kangaroo
	        l1_tmp, l2_tmp; // helper variable to recompute the collision, once found
	// data container to save the `randomly` selected rows to add up and check on a
	// `l` bit window if we found a partial solution.
	uint16_t bitset1[p], bitset2[p];
	for (uint32_t iters = 0; iters < nr_iters; iters++) {
		constexpr lT lMask = (T(1) << l) - 1;
		sp = lT(curand(&state[threadIdx.x])) & lMask;
		r = lT(curand(&state[threadIdx.x])) & lMask;
		//flavour<T, l>(state);// & lMask;
		//r = (blockIdx.x + idx) & lMask;     // choose a random start point
		//sp = (idx<<1) & lMask;              // choose a random flavor

		l1 = sp;
		l2 = sp;
		//printf("sp=%u, r=%u\n", sp, r);

		// start floyd cycle finding algorithm
		do {
			//l1_tmp = l1;
			step<lT, k, p, l, iHwidth, bijection, stern>(l1, bitset1, pdiHT, r, iSyndrom);
			//printf("bitset1[0]=%d, \tl1=%u, \tl1_old=%u\n", bitset1[0], l1, l1_tmp);

			//l2_tmp = l2;
			step<lT, k, p, l, iHwidth, bijection, stern>(l2, bitset2, pdiHT, r, iSyndrom);
			//printf("bitset2[0]=%d, \tl2=%u, \tl2_old=%u\n", bitset2[0], l2, l2_tmp);

			//l2_tmp = l2;
			step<lT, k, p, l, iHwidth, bijection, stern>(l2, bitset2, pdiHT, r, iSyndrom);
			//printf("bitset2[0]=%d, \tl2=%u, \tl2_old=%u\n", bitset2[0], l2, l2_tmp);
		} while (l1 != l2);
		__syncthreads();

		// recompute the collision
		l1 = sp;
		while (l1 != l2) {
			l1_tmp = l1;
			l2_tmp = l2;
			step<lT, k, p, l, iHwidth, bijection, stern>(l1, bitset1, pdiHT, r, iSyndrom);
			step<lT, k, p, l, iHwidth, bijection, stern>(l2, bitset2, pdiHT, r, iSyndrom);
		}
		
		__syncthreads();
		// break if we found a collision in the same function
		//if ((l1_tmp & 1) == (l2_tmp & 1)) {
		if ((__popc(r&l1_tmp)&1) == (__popc(r&l2_tmp)&1)) {
			return;
		}

		// break if it's a useless collision
		//if (bitset1[0] == bitset2[0]) { return; }


		// check if we found the trivial collisions
		//if constexpr (p == 2) {
		//	if ((bitset1[0] == bitset2[0]) &&
		//	    (bitset1[1] == bitset2[1])) {
		//		printf("coll: bitset1[0]=%d bitset2[0]=%d\n", bitset1[0], bitset2[0]);
		//		printf("coll: bitset1[1]=%d bitset2[1]=%d\n", bitset1[1], bitset2[1]);
		//		printf("l1=%u, \tl2=%u, \tl1_old=%u, \tl2_old=%u\n", l1, l2, l1_tmp, l2_tmp);
		//		return;
		//	}
		//}

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
		if ((weight <= w-2*p)) {
			// make the atomic check if we really found a solution.
			if (0 != atomicCAS(compare, 0, 1))
				return;

			// copy the final n-k-l coordinates of the label into the
			// error vector.
			for (uint32_t i = 0; i < Hwidth; i++) {
				pde[i] = x[i];
			}

			// and set the last k+l coordinates to the bit positions
			// which result into the label with low weight.
			for (uint32_t i = 0; i < p; i++) {
				const uint32_t offset = stern ? n-k : n-k-l;
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
	}
}

/// TODO move to misc
/// \param state
/// \param seed
__global__ 
void InitRandomNumberGenerator(curandState *state, 
		const uint64_t seed) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}


/// \tparam T
/// \tparam config
/// \param par
/// \return
template<typename T, const ConfigPollard<T> &config>
uint64_t ThreadPollard(void *par) {
	DecodingInstance *params = (DecodingInstance *)par;
	using lT              = LogTypeTemplate<config.l>;

	// CPU stuff, allocate all needed variables/matrices.
	mzd_t *e            = mzd_init(1, config.n);
	mzd_t *wH           = matrix_init(config.n-config.k, config.n+1);
	mzd_t *tmp          = matrix_concat(nullptr, params->A, params->syndromT);
	mzd_copy(wH, tmp);
	mzd_free(tmp);
	mzd_t *wHT          = mzd_init(wH->ncols, wH->nrows);
	mzd_t *H            = mzd_init(config.n-config.k, config.kl_length+1);  // plus 1 for the syndrome
	mzd_t *HT           = mzd_init(H->ncols, H->nrows);
	mzd_t *iHT          = mzd_init(config.k, config.l);
	// k \times l matrix containing only the `l` part of the transposed parity check matrix

	// current permutation of the parity check matrix
	mzp_t *permutation  = mzp_init(config.n);

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

	// timing stuff
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	float kernel_runtime = 0., total_kernel_runtime = 0.;
	double time = (double)clock();
	uint64_t loops = 0;

	customMatrixData *matrix_data = init_matrix_data(config.n);

	// cuda rng stuff
	curandState * dState;
	const uint64_t seed = rand();
	
	// init rngs.
	checkCudaErrors(cudaMalloc(&dState, params->tpb * sizeof(curandState)));
	InitRandomNumberGenerator<<<1, params->tpb>>>(dState, seed);

	DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError()));
	DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize()));

	// as long, we didn't find anything lets run it.
	while (!Compare && loops < config.loops) {
		loops += 1;

		// apply the gaussian elimination
		//matrix_create_random_permutation(wH, wHT, permutation);
		//matrix_echelonize_partial_plusfix(wH, config.m4ri_k, config.n-config.k,
		//		matrix_data, 0, config.n-config.k, 0, permutation);

		matrix_echelonize_partial_plusfix_opt_onlyc
			<config.n, config.k, config.l, config.gaus_c>
			(wH, wHT, config.m4ri_k, config.n - config.k - config.l,
			 matrix_data, permutation);

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

		// start the actual kernel
		KernelPollard<T, lT,
		              config.n, config.k, config.w, config.p, config.l,
		              config.gpu_matrix_width, config.iHwidth, config.llimbpos,
		              config.bijection, config.stern, config.nr_iters>
		        <<<params->blocks, params->tpb>>>
		        (pde, pdHT, pdiHT, pdsT, iSyndrom, dState, dCompare);

		// simple error checking.
		DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError()));
		DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize()));

		// this is needed to check how much time was spend on the gpu compared
		// to the gpu.
		checkCudaErrors(cudaEventRecord(stop));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&kernel_runtime, start, stop));
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
			mzd_t *tmpe = mzd_init(1, n);
			CudaToMatrix<T>(tmpe, de);
			checkCudaErrors(cudaDeviceSynchronize());

			for (uint32_t i = 0; i < n; i++) {
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


template<typename T, const ConfigPollard<T> &config>
uint64_t CudaPollard() {
	srand(time(nullptr));
	random_seed(rand());
	
	DecodingInstance params{h, s, n, k, config.nr_walks};
	config.print();
	params.print();
	const uint64_t loops = ThreadPollard<T, config>(&params);

	return loops;
}
#endif
