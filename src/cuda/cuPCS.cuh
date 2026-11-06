#ifndef DECODING_CUDAPCS
#define DECODING_CUDAPCS

#include <stdint.h>
#include <curand_kernel.h>

#include "helper.cuh"
#include "pollard.h"
#include "cuPollard.cuh"

/// van Oorschot-Wiener parallel collision search for the collision function of `cuPollard`:
/// every thread walks from a random start point until a distinguished point (`D` low bits
/// zero) and inserts (start, length, point) into a table indexed by the point. If the slot
/// already holds a trail ending in the same point, the two trails are re-walked to the
/// collision, which is checked on the full length like in `KernelPollard`.
/// \tparam T base limb type of the parity check matrix
/// \tparam lT minimal data type to represent `l` bit
/// \tparam n code length
/// \tparam k code dimension
/// \tparam w code weight
/// \tparam p number of rows to add up
/// \tparam l number of bits to match on
/// \tparam Hwidth number of `T` limbs in the parity check matrix HT (must be passed as transposed)
/// \tparam iHwidth number `lT` limb needed to hold the `l` part of one row. Most likely == 1
/// \tparam llimbpos limb position of the `l` part in each row of the parity check matrix `HT`
/// \tparam bijection enforce the internal hash function to be a perfect (but slow) bijection
/// \tparam stern match on k bits. If set to `false` (k+l)
/// \tparam D number of zero low bits of a distinguished point
/// \tparam log_table log2 of the number of table entries
/// \param pde output: the 2p rows of `pdHT` of the solution
/// \param pdHT input: transposed parity check matrix of size n-k \times k
/// \param pdiHT2 input: same as `pdHT` but only the `l` part of each row (aligned to zero)
/// \param pds2 input: syndrome
/// \param iSyndrom input: `l` part of the syndrome
/// \param flavour input: randomizes the function, one value per table
/// \param table input/output: 2^log_table entries, must be zero at the start
/// \param compare output: set to 1 if the solution was found, 0 otherwise
/// \param walks_per_thread number of trails each thread computes
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
         const uint32_t D,
         const uint32_t log_table>
__global__ void KernelPCS(T *__restrict__ pde,                     //
                          const T  *__restrict__ pdHT,             //
                          const lT *__restrict__ pdiHT2,           //
                          const T  *__restrict__ pds2,             //
                          const lT iSyndrom,                       //
                          const lT flavour,                        //
                          unsigned long long *__restrict__ table,  //
                          curandState *__restrict__ state,         //
                          uint32_t *__restrict__ compare,          //
                          const uint32_t walks_per_thread)         //
{
	// table entry: start | length << l | point << (l + 16), 0 = empty
	static_assert(2u*l + 16u <= 64u);
	static_assert(D < l);
	constexpr uint32_t kl_length = stern ? k : k+l;
	constexpr lT lMask = (lT(1) << l) - 1;
	constexpr lT dMask = (lT(1) << D) - 1;
	constexpr uint32_t max_len = 20u << D;
	constexpr uint32_t tMask = (1u << log_table) - 1u;
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ lT pdiHT[iHwidth*kl_length];
	__shared__ T pds[Hwidth];
	for (uint32_t i = threadIdx.x; i < iHwidth*kl_length; i += blockDim.x) {
		pdiHT[i] = pdiHT2[i];
	}
	if (threadIdx.x < Hwidth) {
		pds[threadIdx.x] = pds2[threadIdx.x];
	}
	__syncthreads();

	uint16_t bitset1[p], bitset2[p];
	for (uint32_t walk = 0; walk < walks_per_thread; walk++) {
		if (*(volatile uint32_t *)compare) {
			return;
		}

		// walk to the next distinguished point
		const lT sp = lT(curand(&state[idx])) & lMask;
		lT x = sp;
		uint32_t len = 0;
		do {
			step<lT, k, p, l, iHwidth, bijection, stern>(x, bitset1, pdiHT, flavour, iSyndrom);
			len += 1;
		} while (((x & dMask) != 0) && (len < max_len));
		if ((x & dMask) != 0) {
			continue;
		}

		const uint32_t slot = (uint32_t)((x >> D) * 2654435761u) & tMask;
		const unsigned long long me = (unsigned long long)sp |
		                              ((unsigned long long)len << l) |
		                              ((unsigned long long)x << (l + 16u));
		const unsigned long long old = atomicExch(&table[slot], me);
		if (old == 0) {
			continue;
		}
		const lT sp2 = lT(old) & lMask;
		const lT x2 = lT(old >> (l + 16u)) & lMask;
		const uint32_t len2 = uint32_t(old >> l) & 0xffffu;
		if ((x2 != x) || (sp2 == sp)) {
			continue;
		}

		// locate the collision: align the lengths, then walk both trails together
		lT a = sp, b = sp2;
		uint32_t la = len, lb = len2;
		while (la > lb) { step<lT, k, p, l, iHwidth, bijection, stern>(a, bitset1, pdiHT, flavour, iSyndrom); la -= 1; }
		while (lb > la) { step<lT, k, p, l, iHwidth, bijection, stern>(b, bitset2, pdiHT, flavour, iSyndrom); lb -= 1; }
		// one start point lies on the other trail
		if (a == b) {
			continue;
		}
		lT a_prev = a, b_prev = b;
		while (a != b) {
			a_prev = a;
			b_prev = b;
			step<lT, k, p, l, iHwidth, bijection, stern>(a, bitset1, pdiHT, flavour, iSyndrom);
			step<lT, k, p, l, iHwidth, bijection, stern>(b, bitset2, pdiHT, flavour, iSyndrom);
		}

		// useless collision within the same function
		if ((__popc(flavour & a_prev) & 1u) == (__popc(flavour & b_prev) & 1u)) {
			continue;
		}

		uint32_t weight = 0;
		#pragma unroll
		for (uint32_t i = 0; i < Hwidth; i++) {
			T v = pds[i];
			#pragma unroll
			for (uint32_t j = 0; j < p; j++) {
				v ^= pdHT[bitset1[j] * Hwidth + i];
				v ^= pdHT[bitset2[j] * Hwidth + i];
			}
			weight += (uint32_t) __popc(v);// TODO generalize to T
		}

		if (weight <= w-2*p) {
			if (0 != atomicCAS(compare, 0, 1))
				return;

			// the host reconstructs the error from the rows
			for (uint32_t i = 0; i < p; i++) {
				pde[i]     = bitset1[i];
				pde[p + i] = bitset2[i];
			}
			return;
		}
	}
}

/// GPU version of `Pollard` with distinguished points: the gaus runs on the CPU, the
/// collision search in `KernelPCS`. `config.nr_steps` walks per table, `config.versions`
/// tables (flavours) per permutation.
/// Usage:
///		cuPCS<isdConfig, config> pcs{};
///		pcs.from_string(h, s);
///		pcs.run();
template<const ConfigISD &isd, const ConfigPollard &config>
class cuPCS : public Pollard<isd, config> {
public:
	using Base   = Pollard<isd, config>;
	using ISD    = typename Base::ISD;
	using l_type = typename Base::l_type;
	using Base::n, Base::k, Base::w, Base::p, Base::l, Base::kl;
	using Base::solutions, Base::ws, Base::syndrome, Base::lHT, Base::not_found, Base::loops;

	// `KernelPCS` counts the weight with `__popc`
	using T = uint32_t;
	constexpr static uint32_t Hwidth = (n-k+31u)/32u;
	constexpr static uint32_t tpb    = 256;
	constexpr static uint32_t blocks = std::min((config.nr_steps + tpb - 1u)/tpb, 256u);
	constexpr static uint32_t walks_per_thread = std::max(config.nr_steps/(blocks*tpb), 1u);
	constexpr static size_t table_size = size_t(1) << config.log_table;
	constexpr static l_type lmask = (l_type(1) << l) - 1u;

	T *dHT = nullptr, *dSol = nullptr;
	l_type *dlHT = nullptr;
	uint32_t *dCompare = nullptr;
	curandState *dState = nullptr;
	unsigned long long *dTable = nullptr;
	std::vector<T> hHT;

	cuPCS() noexcept : Base(), hHT((kl+1u)*Hwidth) {
		checkCudaErrors(cudaMalloc(&dHT, hHT.size()*sizeof(T)));
		checkCudaErrors(cudaMalloc(&dlHT, kl*sizeof(l_type)));
		checkCudaErrors(cudaMalloc(&dSol, 2*p*sizeof(T)));
		checkCudaErrors(cudaMalloc(&dCompare, sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dState, blocks*tpb*sizeof(curandState)));
		checkCudaErrors(cudaMalloc(&dTable, table_size*sizeof(unsigned long long)));
		InitRandomNumberGenerator<<<blocks, tpb>>>(dState, rng());
		checkCudaErrors(cudaGetLastError());
	}

	~cuPCS() noexcept {
		cudaFree(dHT); cudaFree(dlHT); cudaFree(dSol);
		cudaFree(dCompare); cudaFree(dState); cudaFree(dTable);
	}

	// same as `cuPollard::run`, but `KernelPCS` with a fresh table per flavour
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
			for (uint32_t v = 0; (v < config.versions) && !compare; v++) {
				const l_type flavour = l_type(rng()) & lmask;
				checkCudaErrors(cudaMemsetAsync(dTable, 0, table_size*sizeof(unsigned long long)));
				KernelPCS<T, l_type, n, k, w, p, l, Hwidth, 1, 0, config.isomorhism, false, config.D, config.log_table>
				        <<<blocks, tpb>>>(dSol, dHT, dlHT, dHT + kl*Hwidth, l_type(syndrome), flavour, dTable, dState, dCompare, walks_per_thread);
				checkCudaErrors(cudaGetLastError());
				checkCudaErrors(cudaMemcpy(&compare, dCompare, sizeof(uint32_t), cudaMemcpyDeviceToHost));
			}

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
