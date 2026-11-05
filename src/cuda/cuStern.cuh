#pragma once

#include <cstdint>
#include <vector>

#include "helper.cuh"
#include "stern.h"

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

/// calculates H1*e1 + s on the `l` bits and inserts it into the buckets
/// NOTE: elements of full buckets are dropped
/// \tparam T limb type
/// \tparam k instance parameter
/// \tparam l instance parameter
/// \tparam p instance parameter
/// \tparam bucket_size number of elements per bucket
/// \param diH `l` part of each row of the parity check matrix
/// \param dis `l` part of the syndrome
/// \param dC change list, contains the
/// \param pdBucket output: list indices per bucket
/// \param pdCounter output: load per bucket, must be zero
/// \param nr_combinations size of the list to enumerate: (k+l)/2 over p = |dC| = size of the change list.
template<typename T,
         const uint32_t k,
         const uint32_t l,
         const uint32_t p,
         const uint32_t bucket_size>
__global__ void dumer_step1(const T *__restrict__ diH,                   // parity check matrix
							const T dis,
							const uint32_t *__restrict__ dC,    // array containing bit pos of "1" in error vector
                            uint32_t* pdBucket,
                            uint32_t* pdCounter,
                            const uint64_t nr_combinations){
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

	while(idx < nr_combinations) {
		assert(dC[p*idx] < ((k+l)/2));
		// left column + syndrome and extract lower l bit
		T a = dis;

		#pragma unroll
		for (uint32_t i = 0; i < p; i++) {
			const uint32_t j = dC[idx*p + i];
			a ^= diH[j];
		}

    	assert(a < (1u << l));

		// NOTE: elements of full buckets are dropped
    	const uint32_t pos = atomicAdd((uint32_t*) &pdCounter[a], 1);
		if (pos < bucket_size) {
			pdBucket[a*bucket_size + pos] = idx;
		}

		idx += blockDim.x * gridDim.x;
	}
}


/// \tparam T limb type of the rows/columns
/// \tparam k code dimension
/// \tparam l number of bits to match on in the first lvl
/// \tparam w code weight
/// \tparam p number of bits to enumerate in the base lists
/// \tparam width number of limbs each row (or column in the not transposed matrix) consists of.
/// \tparam bucket_size number of elements per bucket
/// \param de output: the `p` left and `p` right list rows, interleaved
/// \param dH input: parity check matrix
/// \param ds input: syndrome
/// \param diH input: `l` part of each row of the parity check matrix
/// \param dC input: combinations of the error vector we are enumerating in the base lists
/// \param nr_combinations number of of elements in the second list (right list)
/// \param compare output: set to 1 if the solution was found, 0 otherwise
template<typename T,
         const uint32_t k,
         const uint32_t l,
         const uint32_t w,
         const uint32_t p,
		 const uint32_t width,
         const uint32_t bucket_size>
__global__ void dumer_step3(uint32_t *__restrict__ de,                  // output solution rows
                            const T *__restrict__ dH,                   // parity check matrix
                            const T *__restrict__ ds,                   // syndrome
                            const T *__restrict__ diH,                  // parity check matrix
                            const uint32_t *__restrict__ dC,            // table
                            const uint32_t* __restrict__ pdBucket,
                            const uint32_t* __restrict__ pdCounter,
                            const uint32_t nr_combinations,             // size of table
                            uint32_t *compare) {  						// TODO reduce to only one parameter
	constexpr uint32_t kl_half      = (k+l)/2;
	constexpr uint32_t weightt      = w - (2u*p);

	const uint32_t block_number     = blockDim.x * gridDim.x;
	uint32_t index_right            = threadIdx.x + blockIdx.x * blockDim.x;

	while(index_right < nr_combinations) {
    	// no width needed, only 1 int per column
		// NOTE same as in `dumer_step1` but for the right list
		T a_right = 0;

		#pragma unroll
		for (uint32_t i = 0; i < p; i++) {
			const uint32_t j = dC[index_right*p + i] + kl_half;
			a_right ^= diH[j];
		}

		// some security check
		assert(a_right < (1u << l));

		const uint32_t load = min(bucket_size, pdCounter[a_right]);
		for (uint32_t j = 0; j < load; j++) {
			// get the pdC index for the collision
			const uint32_t index_left = pdBucket[a_right * bucket_size + j];
			assert(index_left < nr_combinations);

			// compute the weight on the full length, the `l` part is zero
			uint32_t weight = 0;

			#pragma unroll
			for (uint32_t i = 0; i < width; i++) {
				T x = ds[i];
				#pragma unroll
				for (uint32_t r = 0; r < p; r++) {
					x ^= dH[(           dC[index_left  * p + r]  * width) + i] ^
					     dH[((kl_half + dC[index_right * p + r]) * width) + i];
				}
				weight += __popc(x);
			}

			// if we pass this check we found a solution
			if (weight <= weightt) {
				if (0 != atomicCAS(compare, 0, 1)) {
					return;
				}

				// copy back the solution
				for (uint32_t i = 0; i < p; i++) {
					de[2*i + 0] = dC[index_left *p + i];
					de[2*i + 1] = dC[index_right*p + i];
				}
				return;
			}
		}

		index_right += block_number;
	}
}

/// GPU version of `Stern` (Dumer): the gaus runs on the CPU, the
/// list construction and matching in `dumer_step1` and `dumer_step3`
/// Usage:
///		cuStern<isdConfig, config> stern{};
///		stern.from_string(h, s);
///		stern.run();
template<const ConfigISD &isd, const ConfigStern &config>
class cuStern : public Stern<isd, config> {
public:
	using Base   = Stern<isd, config>;
	using ISD    = typename Base::ISD;
	using l_type = typename Base::l_type;
	using Base::n, Base::k, Base::w, Base::p, Base::l, Base::kl_half;
	using Base::solutions, Base::ws, Base::syndrome, Base::lHT, Base::not_found, Base::loops;

	// the kernels count the weight with `__popc`
	using T = uint32_t;
	static_assert(sizeof(l_type) == sizeof(T));
	constexpr static uint32_t Hwidth = (n-k+31u)/32u;
	constexpr static uint32_t kl     = k+l;
	constexpr static uint32_t nr_combinations = bc(kl_half, p);
	constexpr static uint32_t bucket_size = config.HM_bucketsize;
	constexpr static uint32_t tpb    = 256;
	constexpr static uint32_t blocks = std::min((nr_combinations + tpb - 1u)/tpb, 1024u);

	T *dHT = nullptr, *dlHT = nullptr;
	uint32_t *dC = nullptr, *dBucket = nullptr, *dCounter = nullptr;
	uint32_t *dSol = nullptr, *dCompare = nullptr;
	std::vector<T> hHT;

	cuStern() noexcept : Base(), hHT((kl+1u)*Hwidth) {
		// all `p` out of `kl_half` combinations in lexicographical order
		std::vector<uint32_t> hC(size_t(nr_combinations)*p);
		uint32_t c[p];
		for (uint32_t j = 0; j < p; ++j) { c[j] = j; }
		for (uint32_t i = 0; i < nr_combinations; ++i) {
			for (uint32_t j = 0; j < p; ++j) { hC[size_t(i)*p + j] = c[j]; }

			int32_t j = p-1;
			while ((j >= 0) && (c[j] == kl_half-p+j)) { j -= 1; }
			if (j < 0) { break; }
			c[j] += 1;
			for (uint32_t t = j+1; t < p; ++t) { c[t] = c[t-1] + 1; }
		}

		checkCudaErrors(cudaMalloc(&dHT, hHT.size()*sizeof(T)));
		checkCudaErrors(cudaMalloc(&dlHT, kl*sizeof(T)));
		checkCudaErrors(cudaMalloc(&dC, hC.size()*sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dBucket, (size_t(1) << l)*bucket_size*sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dCounter, (size_t(1) << l)*sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dSol, 2*p*sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dCompare, sizeof(uint32_t)));
		checkCudaErrors(cudaMemcpy(dC, hC.data(), hC.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
	}

	~cuStern() noexcept {
		cudaFree(dHT); cudaFree(dlHT); cudaFree(dC); cudaFree(dBucket);
		cudaFree(dCounter); cudaFree(dSol); cudaFree(dCompare);
	}

	// same as `Stern::run`, but the list construction and matching runs on the gpu
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
			checkCudaErrors(cudaMemcpy(dlHT, lHT, kl*sizeof(T), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemset(dCounter, 0, (size_t(1) << l)*sizeof(uint32_t)));
			dumer_step1<T, k, l, p, bucket_size>
			        <<<blocks, tpb>>>(dlHT, T(syndrome), dC, dBucket, dCounter, nr_combinations);
			dumer_step3<T, k, l, w, p, Hwidth, bucket_size>
			        <<<blocks, tpb>>>(dSol, dHT, dHT + kl*Hwidth, dlHT, dC, dBucket, dCounter, nr_combinations, dCompare);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaMemcpy(&compare, dCompare, sizeof(uint32_t), cudaMemcpyDeviceToHost));

			if (compare) {
				uint32_t sol[2*p];
				checkCudaErrors(cudaMemcpy(sol, dSol, 2*p*sizeof(uint32_t), cudaMemcpyDeviceToHost));
				for (uint32_t i = 0; i < p; ++i) {
					solutions[i]     = sol[2*i + 0];
					solutions[p + i] = kl_half + sol[2*i + 1];
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
