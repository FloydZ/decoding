#ifndef DECODING_CUDAPOLLARD
#define DECODING_CUDAPOLLARD

#include <stdint.h>
#include <curand_kernel.h>

#include "helper.cuh"
#include "pollard.h"


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

		assert(wn < n);
		// printf("%d %d %d %d %d\n", wn, n, w, a, label);
	}

	if constexpr (w == 2) {
		// w == 2
		wn -= opt_max_bc<T, 2>(a, wn);
		a -= ((wn-1) *(wn-2)) / 2;

		// decrement the counter and set the bit accordingly
		wn -= 1;
		ret[0] = wn;
		assert(wn < n);

		wn -= opt_max_bc<T, 1>(a, wn);

		// decrement the counter and set the bit accordingly
		wn -= 1;
		ret[1] = wn;
		assert(wn < n);
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
		assert(wn < n);
		assert(a < binom);

		// w == 2
		wn -= opt_max_bc<T, 2>(a, wn);
		a -= ((wn-1) * (wn-2)) / 2;

		//printf("idx: %d: %d\n", threadIdx.x, wn);
		// decrement the counter and set the bit accordingly
		wn -= 1;
		ret[1] = wn;
		assert(wn < n);

		wn -= opt_max_bc<T, 1>(a, wn);

		// decrement the counter and set the bit accordingly
		wn -= 1;
		ret[2] = wn;
		assert(wn < n);
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
/// \param pde output: the 2p rows of `pdHT` of the solution
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
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

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
		r = lT(curand(&state[idx])) & lMask;
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

			// the host reconstructs the error from the rows
			for (uint32_t i = 0; i < p; i++) {
				pde[i]     = bitset1[i];
				pde[p + i] = bitset2[i];
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

/// GPU version of `Pollard`: the gaus runs on the CPU, the walks in `KernelPollard`
/// Usage:
///		cuPollard<isdConfig, config> pollard{};
///		pollard.from_string(h, s);
///		pollard.run();
template<const ConfigISD &isd, const ConfigPollard &config>
class cuPollard : public Pollard<isd, config> {
public:
	using Base   = Pollard<isd, config>;
	using ISD    = typename Base::ISD;
	using l_type = typename Base::l_type;
	using Base::n, Base::k, Base::w, Base::p, Base::l, Base::kl;
	using Base::solutions, Base::ws, Base::syndrome, Base::lHT, Base::not_found, Base::loops;

	// `KernelPollard` counts the weight with `__popc`
	using T = uint32_t;
	constexpr static uint32_t Hwidth = (n-k+31u)/32u;
	constexpr static uint32_t tpb    = 256;
	constexpr static uint32_t blocks = (config.nr_steps + tpb - 1u)/tpb;
	static_assert(kl <= tpb);

	T *dHT = nullptr, *dSol = nullptr;
	l_type *dlHT = nullptr;
	uint32_t *dCompare = nullptr;
	curandState *dState = nullptr;
	std::vector<T> hHT;

	cuPollard() noexcept : Base(), hHT((kl+1u)*Hwidth) {
		checkCudaErrors(cudaMalloc(&dHT, hHT.size()*sizeof(T)));
		checkCudaErrors(cudaMalloc(&dlHT, kl*sizeof(l_type)));
		checkCudaErrors(cudaMalloc(&dSol, 2*p*sizeof(T)));
		checkCudaErrors(cudaMalloc(&dCompare, sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dState, blocks*tpb*sizeof(curandState)));
		InitRandomNumberGenerator<<<blocks, tpb>>>(dState, rng());
		checkCudaErrors(cudaGetLastError());
	}

	~cuPollard() noexcept {
		cudaFree(dHT); cudaFree(dlHT); cudaFree(dSol);
		cudaFree(dCompare); cudaFree(dState);
	}

	// same as `Pollard::run`, but `walk` is replaced by `KernelPollard`
	uint64_t run() noexcept {
		ISD::reset();
		uint32_t compare = 0;
		checkCudaErrors(cudaMemcpy(dCompare, &compare, sizeof(uint32_t), cudaMemcpyHostToDevice));

		while (not_found && loops < isd.loops) {
			ISD::step();
			ISD::template extract_lHT<l_type>(lHT);
			ISD::template extract_pHT<T>(hHT.data());
			// syndrome as last row
			cryptanalysislib::memcpy(hHT.data() + kl*Hwidth, (const T *)ws.ptr(), Hwidth);

			checkCudaErrors(cudaMemcpy(dHT, hHT.data(), hHT.size()*sizeof(T), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(dlHT, lHT, kl*sizeof(l_type), cudaMemcpyHostToDevice));
			KernelPollard<T, l_type, n, k, w, p, l, Hwidth, 1, 0, config.isomorhism, false, 1>
			        <<<blocks, tpb>>>(dSol, dHT, dlHT, dHT + kl*Hwidth, l_type(syndrome), dState, dCompare);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaMemcpy(&compare, dCompare, sizeof(uint32_t), cudaMemcpyDeviceToHost));

			if (compare) {
				T sol[2*p];
				checkCudaErrors(cudaMemcpy(sol, dSol, 2*p*sizeof(T), cudaMemcpyDeviceToHost));
				for (uint32_t i = 0; i < 2*p; ++i) {
					solutions[i] = sol[i];
				}
				not_found = false;
			}
		}

		if (!not_found) {
			Base::reconstruct();
		}
		return loops;
	}
};


#endif
