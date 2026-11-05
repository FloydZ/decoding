#ifndef DECODING_CUDALEEBRICKELLNOMEM
#define DECODING_CUDALEEBRICKELLNOMEM

#include <cstdint>
#include <vector>

#include "helper.cuh"
#include "isd.h"
#include "math/bc.h"

using namespace cryptanalysislib;

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
/// │                    │                                  │  │     │
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

/// actual hashing function. Depending on p a different weight function is choosen.
/// NOTE: the outputs may collide, duplicated rows cancel out.
/// \tparam T		limb type. Most likely `uint32_t`
/// \tparam k		code dimension
/// \tparam p		number of bit positions (or column selection) this function is going to output
/// \tparam logk	(only needed if p==2), logarithmic size of `k`
/// \tparam f
/// \param set		output:
/// \param value	input: thread index we gonna use as the
template<typename T,
         const uint32_t k,
         const uint32_t p,
         const uint32_t logk,
         const uint32_t f>
__device__ __forceinline__
void hash(uint16_t set[p],
          const T value) {
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
		set[1] = (set[1] + t) % mod;
	} else if constexpr(p==3){
		constexpr uint32_t ws = logk;
		// constexpr T mask1 = (T(1) << ws) - 1;
		// constexpr T mask2 = (T(1) << (2*ws)) - 1;

		constexpr T mod = k;
		set[2] = ( value          ) % mod;
		set[1] = ((value >> ws   )) % mod;
		set[0] = ( value >> (f-ws)) % mod;

		const bool t = set[0] == set[1];
		set[1] = (set[1] + t) % mod;

		const bool u = set[1] == set[2];
		set[2] = (set[2] + u) % mod;
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
		set[1] = (set[1] + t) % mod;

		const bool u = set[1] == set[2];
		set[2] = (set[2] + u) % mod;

		const bool v = set[2] == set[3];
		set[3] = (set[3] + v) % mod;
	}
}

/// \tparam T			limp type
/// \tparam k			code dimension, needed for the hash function
/// \tparam w			code weight, needed for the hash function
/// \tparam p			number of bit positions (or column selection) to map on
/// \tparam logk		helper for the hash function, see `cuLeeBrickellNoMem` for explanation
/// \tparam f			same as `logk`
/// \tparam width		number of limbs needed to store one row in `dHH` or `dss`
/// \param de			output: the `p` rows of `dHH` of the solution
/// \param dHH			input: transposed parity check matrix
/// \param dss			input: syndrome
/// \param nr_combinations total number of hash inputs we try
/// \param compare		output: set to 1 if the solution was found, zero otherwise.
template<typename T,
         const uint32_t k,
         const uint32_t w,
         const uint32_t p,
         const uint32_t logk,
         const uint32_t f,
         const uint32_t width>
__global__ void lee_brickellnomem_step(uint32_t *__restrict__ de, // solution rows
                                       const T *__restrict__ dHH, // parity check matrix
                                       const T *__restrict__ dss, // syndrome
                                       const uint32_t nr_combinations, // total number of combinations we try // todo move to template
                                       unsigned int *__restrict__ compare) {
	const uint32_t block_number =  blockDim.x * gridDim.x;
	uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	// copy the parity check matrix and the syndrome into the shared mem.
	__shared__ T ds[width], dH[width*k];
	for (uint32_t i = threadIdx.x; i < width*k; i += blockDim.x) {
		dH[i] = dHH[i];
	}

	if (threadIdx.x < width) {
		ds[threadIdx.x] = dss[threadIdx.x];
	}
	__syncthreads();

	uint16_t dC[p] = {0};
	while(index < nr_combinations) {
		// TODO maybe flavor the run?
		::hash<T, k, p, logk, f>(dC, index);
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

			// the host reconstructs the error from the rows
			for (uint16_t i = 0; i < p; i++) {
				de[i] = dC[i];
			}
			return;
		}

		index += block_number;
	}
}

/// GPU Lee-Brickell without the combinations table: each thread hashes its index into `p` rows
/// Usage:
///		cuLeeBrickellNoMem<isdConfig> lb{};
///		lb.from_string(h, s);
///		lb.run();
template<const ConfigISD &isd>
class cuLeeBrickellNoMem : public ISDInstance<uint64_t, isd> {
public:
	constexpr static uint32_t n = isd.n;
	constexpr static uint32_t k = isd.k;
	constexpr static uint32_t w = isd.w;
	constexpr static uint32_t p = isd.p;

	static_assert(isd.l == 0);
	static_assert((p > 0) && (p <= 4) && (p < w));

	using ISD    = ISDInstance<uint64_t, isd>;
	using Error  = typename ISD::Error;
	using Label  = typename ISD::Label;
	using ISD::HT,ISD::ws,ISD::e,ISD::P,ISD::not_found,ISD::loops,ISD::expected_loops;

	// `lee_brickellnomem_step` counts the weight with `__popc`
	using T = uint32_t;
	constexpr static uint32_t Hwidth = (n-k+31u)/32u;

	// number of bits needed to select one row
	constexpr static uint32_t logk = ceil_log2(k);
	// number of bits needed to select `p` rows from the thread index
	constexpr static uint32_t f    = p * logk;
	static_assert(f < 32);

	// every hash input, s.t. every combination is reached
	constexpr static uint32_t nr_combinations = 1u << f;
	constexpr static uint32_t tpb    = 256;
	constexpr static uint32_t blocks = std::min((nr_combinations + tpb - 1u)/tpb, 1024u);
	static_assert((Hwidth*(k+1u))*sizeof(T) <= 48u*1024u, "HT does not fit into the shared memory");

	// rows of `HT` of the solution
	uint32_t solutions[p] = {0};

	T *dHT = nullptr;
	uint32_t *dSol = nullptr, *dCompare = nullptr;
	std::vector<T> hHT;

	cuLeeBrickellNoMem() noexcept : hHT((k+1u)*Hwidth) {
		expected_loops = bc(n, w)/(bc(n-k, w-p) * bc(k, p));

		checkCudaErrors(cudaMalloc(&dHT, hHT.size()*sizeof(T)));
		checkCudaErrors(cudaMalloc(&dSol, p*sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dCompare, sizeof(uint32_t)));
	}

	~cuLeeBrickellNoMem() noexcept {
		cudaFree(dHT); cudaFree(dSol); cudaFree(dCompare);
	}

	/// reconstruct the final solution
	void __attribute__ ((noinline))
	reconstruct() noexcept {
		Error tmpe;
		Label tmp;
		// copy in the syndrome
		tmp = ws;
		for (uint32_t i = 0; i < p; ++i) {
			assert(solutions[i] < k);
			Label::add(tmp.ptr(), tmp.ptr(), HT.row(solutions[i]));
		}

		for (uint32_t i = 0; i < n-k; ++i) {
			const auto bit = tmp.get(i);
			tmpe.set(bit, 0, n-k-1-i);
		}

		// NOTE: flip, the hash may output the same row twice
		for (uint32_t i = 0; i < p; ++i) {
			const uint32_t pos = n-k + solutions[i];
			tmpe.set(tmpe.get(0, pos) ^ 1u, 0, pos);
		}

		// apply back permutation.
		for (uint32_t i = 0; i < n; ++i) {
			e.set(tmpe.get(0, i), 0, P.values[i]);
		}
	}

	// execute the program and return the number of loops needed to find the solution
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();
		uint32_t compare = 0;
		checkCudaErrors(cudaMemcpy(dCompare, &compare, sizeof(uint32_t), cudaMemcpyHostToDevice));

		while (not_found && loops < isd.loops) {
			ISD::step();
			ISD::template extract_pHT<T>(hHT.data());
			// syndrome as last row
			cryptanalysislib::memcpy(hHT.data() + k*Hwidth, (const T *)ws.ptr(), Hwidth);

			checkCudaErrors(cudaMemcpy(dHT, hHT.data(), hHT.size()*sizeof(T), cudaMemcpyHostToDevice));
			lee_brickellnomem_step<T, k, w, p, logk, f, Hwidth>
			        <<<blocks, tpb>>>(dSol, dHT, dHT + k*Hwidth, nr_combinations, dCompare);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaMemcpy(&compare, dCompare, sizeof(uint32_t), cudaMemcpyDeviceToHost));

			if (compare) {
				checkCudaErrors(cudaMemcpy(solutions, dSol, p*sizeof(uint32_t), cudaMemcpyDeviceToHost));
				not_found = false;
			}
		}

		if (!not_found) {
			reconstruct();
		}
		return loops;
	}

	/// important. Dont rename it
	void info() const noexcept {
		std::cout << "cuLeeBrickellNoMem" << std::endl;
	}
};

#endif // DECODING_CUDALEEBRICKELLNOMEM
