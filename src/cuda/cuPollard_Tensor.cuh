#ifndef DECODING_CUDAPOLLARD_TENSOR
#define DECODING_CUDAPOLLARD_TENSOR

#include <curand_kernel.h>

#include "cuhelper.cuh"
#include "cuMisc.cuh"


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

	// nr of concurrent random walk we start
	const uint32_t nr_walks     = 1u << l;

	const uint32_t nr_iters     = 1;

	/// standard constructor
	constexpr ConfigPollard(const uint32_t n,
	                        const uint32_t k,
	                        const uint32_t l,
	                        const uint32_t w,
	                        const uint32_t p,
	                        const bool bijection=false,
	                        const bool stern=true,
	                        const uint32_t nr_iters=1) :
			n(n), k(k), l(l), w(w), p(p), bijection(bijection), stern(stern), nr_iters(nr_iters) {};

	constexpr ConfigPollard(ConfigPollard &c) = default;

	// returns the expected number of iterations
	constexpr uint64_t Loops() const {
		//return bc(n, w)/( bc(n-k-l, w-2*p) * bc(k, 2*p));
		return bc(n, w)/( bc(n-k-l, w-2*p) * bc(k/2, p) * bc(k/2, p));
	}

	// print some information about the configuration
	void print() const {
		std::cout << "n: " << unsigned(n)
		          << ", k: " << unsigned (k)
		          << ", l: " << unsigned (l)
		          << ", n-k:" << unsigned (n-k)
		          << ", p: " << unsigned (p)
		          << ", cpu_gpu_matrix_factor: " << cpu_gpu_matrix_factor
		          << ", gpu_matrix_width: " << gpu_matrix_width
		          << ", cpu_matrix_width: " << cpu_matrix_width
		          << ", cpu_matrix_rows: " << cpu_matrix_rows
		          << ", llimbpos: " << llimbpos
		          << ", iHwidth: " << iHwidth
				  << ", nr_walks: " << nr_walks
		          << ", nr_iters: " << nr_iters
		          << ", bijection: " << bijection
				  << ", stern: " << stern
		          << ", n-k-l: " << n-k-l << "\n";
	}

	/// \param loops
	/// \param time
	/// \param total_kernel_runtime
	void print_loop(const uint64_t loops, const double time, const float total_kernel_runtime) {
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



// TODO remove?
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

// global constant. only used of `bijection` is set to `true`. If so
__constant__ uint64_t dBinom[4096];

/// perfect bijection between the space of random labels = {0,1}^l and
/// the space of random error vector {0,1}^k of weight `p`.
/// \tparam T minimal data type to hold `l` bits
/// \tparam p number of random bit position
/// \param bitpos output: the bit positions of the rows to add up.
/// \param label input: random label to extract the random bit positions
/// \return the parity of the next function:
///             0: just sum of row
///             1: sum of rows + syndrome
template<typename T, const uint32_t k, const uint32_t p>
__device__ uint16_t bijection_fkt(uint16_t bitpos[p], const T label) {
	uint64_t a = label % dBinom[k*(p+1) +p]; //bc(k, p);
	// *next_bit = label & 1;
	//std::cout<<a<<"\n";
	uint32_t wn = k;
	uint32_t wk = p;
	//uint64_t v = 0;
	int set = 0;
	while (wn != 0) {
		if (set == p)
			break;
		else if (wn + set == p) {
			//v += (1ULL << (wn - 1));
			bitpos[set] = (wn - 1);
			wn -= 1;
			set += 1;
		}
		else if (a < dBinom[(wn - 1)*(p+1) + wk])
		//else if (a < bc(wn-1, wk))
			wn -= 1;
		else {
			a -= dBinom[(wn - 1)*(p+1) + wk];
			//a -= bc(wn-1, wk);

			//v += (1ULL << (wn - 1));
			bitpos[set] = (wn - 1);

			wn -= 1;
			wk -= 1;
			set += 1;
		}
	}

	//return v;
	return label & 1;
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
template<typename T, const uint32_t k, const uint32_t p, const uint32_t l>
__device__ __forceinline__ uint16_t hash1(uint16_t bitpos[p], const T label) noexcept {
	// only usable if p = 1
	static_assert(l < 32);
	static_assert(p == 1);

	//constexpr T mask1 = ((T(1) << l) - 1);
	bitpos[0] = label % k;

	// use the first bit as a selector for the next functions
	return label & 1;
}

//
template<typename T, const uint32_t k, const uint32_t p, const uint32_t l>
__device__ __forceinline__ T hash22(uint16_t bitpos[p], const T label) noexcept {
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
template<typename T, const uint32_t k, const uint32_t p, const uint32_t l>
__device__ __forceinline__ uint16_t hash2(uint16_t bitpos[p], const T label) noexcept {
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
template<typename T, const uint32_t k ,const uint32_t p, const uint32_t l>
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
template<typename T, const uint32_t k, const uint32_t p, const uint32_t width>
__device__ __forceinline__ void calcHe(T &label,
                                       const uint16_t bitpos[p],
                                       const T *__restrict__ iHT) noexcept {
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
template<typename T, const uint32_t k, const uint32_t p, const uint32_t l, const uint32_t width, const bool bijection, const uint32_t stern>
__device__  void step(T &label, uint16_t bitpos[p], const T *__restrict__ iHT, const T flavour, const T iSyndrom) noexcept {
	// NOTE: this value is set by one of the hash functions to either
	// zero or one. If it's set to zero, it means that the syndrome is not calculated
	// in.
	uint16_t next;

	constexpr uint32_t collision_bits = stern ? k/2 : (k+l)/2;
	if constexpr (bijection) {
		next = bijection_fkt<T, collision_bits, p>(bitpos, label);
	} else {
		if constexpr (p == 1u) {
			next = hash1<T, collision_bits, p, l>(bitpos, label);
		} else if constexpr (p == 2u) {
			next = hash22<T, collision_bits, p, l>(bitpos, label);
		} else if constexpr (p == 3u) {
			next = hash3<T, collision_bits, p, l>(bitpos, label);
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


	ASSERT(next <= 1);
	const uint32_t off = next*(collision_bits);
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
__device__ __forceinline__ T flavour(curandState *__restrict__ state) {
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
         const uint64_t n, const uint32_t k, const uint32_t w, const uint32_t p, const uint32_t l,
         const uint32_t Hwidth, const uint32_t iHwidth, const uint32_t llimbpos,
         const bool bijection, const bool stern, const uint32_t nr_iters>
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
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

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
		sp = lT(curand(&state[idx])) & lMask;
		r = lT(curand(&state[idx])) & lMask;  //flavour<T, l>(state);// & lMask;
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

		// recompute the collision
		l1 = sp;
		while (l1 != l2) {
			l1_tmp = l1;
			l2_tmp = l2;
			step<lT, k, p, l, iHwidth, bijection, stern>(l1, bitset1, pdiHT, r, iSyndrom);
			step<lT, k, p, l, iHwidth, bijection, stern>(l2, bitset2, pdiHT, r, iSyndrom);
		}

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
		if (unlikely(weight <= w-2*p)) {
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


__global__ void InitRandomNumberGenerator(curandState *state, const uint64_t seed){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed,idx,0, &state[idx]);
}

// problem instance for the algorithm.
struct ParamsPollard {
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
		std::cout << "Blocks: " << blocks << ", threads per block: " << tpb << ", total: " << blocks*tpb << "\n";
	}
};


/// \tparam T
/// \tparam config
/// \param par
/// \return
template<typename T, const ConfigPollard<T> &config>
bool ThreadPollard(void *par) {
	ParamsPollard *params = (ParamsPollard *)par;
	using lT              = LogTypeTemplate<config.l>;

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

	customMatrixData *matrix_data = init_matrix_data(wH->ncols);


	// cuda rng stuff
	curandState * dState; const uint64_t seed = rand();
	if constexpr(config.p != 3) {
		checkCudaErrors(cudaMalloc(&dState, params->blocks * params->tpb * sizeof(curandState)));
		InitRandomNumberGenerator<<<params->blocks, params->tpb>>>(dState, seed);
	}

	// precompute the binomial coefficients for the bijeciton
	if constexpr (config.bijection) {
		static_assert((config.k+1) * (config.p+1) < 4096);
		// create a lookup table for binomial coefficients for every (i, j) in [k, p]
		uint64_t *binom = new uint64_t[(config.k+1) * (config.p+1)];
		for (uint32_t i = 0; i < config.k+1; ++i) {
			for (uint32_t j = 0; j < config.p+1; ++j) {
				binom[i*(config.p+1) + j] = bc(i, j);
			}
		}

		checkCudaErrors(cudaMemcpyToSymbol(dBinom, binom, sizeof(uint64_t)*(config.p+1)*(config.k+1), 0, cudaMemcpyHostToDevice));
		// free up everything
		delete[] binom;
	}

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
		DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError());)
		DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize());)

		// same as above only for the `l` part
		MatrixToCudaReduce<lT>(pdiHT, iHT, config.iHwidth);
		DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError());)
		DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize());)

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
		DEBUG_MACRO(checkCudaErrors(cudaPeekAtLastError());)
		DEBUG_MACRO(checkCudaErrors(cudaDeviceSynchronize());)

		// this is needed to check how much time was spend on the gpu compared
		// to the gpu.
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&kernel_runtime, start, stop);
		total_kernel_runtime += kernel_runtime;

		// copy back the `is a solution found flag`
		// OPTIMIZE: Instead of checking this every loop, let the algorithm run for a few loops,
		// check then if we found the solution. This ca be done, because once set. Wow it made it slower
		//if (loops % config.exit_loops) {
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
		//}
		// print every X loops a status update of the algorithm
		if ((loops%config.print_loops) == 0) {
			const double time2 = ((double)clock() -  time)/CLOCKS_PER_SEC;
			const double ktime = total_kernel_runtime/1e3;
			std::cout << "\rloops: " <<  loops << "(" << double(loops)/double(config.Loops()) * 100. << "%), loops/s:" << loops/time2 << ", kernel runtime: " << ktime << "s (" << ktime/time2*100. << "%) , runtime: " << time2 << "s\n" << std::flush;
		}
	}

	time = ((double)clock() - time)/CLOCKS_PER_SEC;
	total_kernel_runtime /= 1e3;
	std::cout << "\nruntime: "  << time << "s, loops: " << loops << ", E[loops]: " << config.Loops() << " (" << double(loops)/double(config.Loops())*100. << "%), total kernel runtime: " << total_kernel_runtime << "s (" << total_kernel_runtime/time*100. << "%), loops/kernel s: " << loops/total_kernel_runtime << ", loops/s: " << loops/time << "\n";
	bool ret = checkCorrectness(e, params->A, params->syndromT, w);

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

	return ret;
}


template<typename T, const ConfigPollard<T> &config>
int CudaPollard() {
	srand(time(nullptr));
	random_seed(rand());
	config.print();

	cudaSetDevice(0);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	const uint32_t ncores = getSPcores(prop);

	const uint64_t tpb      = MIN(1024, config.nr_walks);
	const uint64_t blocks   = MAX(1, config.nr_walks / tpb);

	ParamsPollard params{
			.blocks         = blocks,
			.tpb            = tpb,
			.AT             = mzd_from_str(n, n - k, h),
			.syndrom        = mzd_from_str(1, n - k, s),
	};

	params.A = mzd_transpose(nullptr, params.AT);
	params.syndromT = mzd_transpose(nullptr, params.syndrom);
	params.print();

	ThreadPollard<T, config>(&params);

	//std::vector<std::thread> threads;
	//for (int i = 0; i < config.nr_threads; i++) {threads.emplace_back(ThreadPollard<T, config>, &params);
	//	std::cout << "thread: " << i << " started\n"  << std::flush;
	//}

	//std::cout << config.nr_threads << " threads started" << std::endl;
	//for (int i = 0; i < config.nr_threads; i++) {
	//	threads[i].join();
	//}

	//cudaDeviceReset();

	// free everything we allocated
	mzd_free(params.syndrom);
	mzd_free(params.syndromT);
	mzd_free(params.A);
	mzd_free(params.AT);
	return 1;
}
#endif