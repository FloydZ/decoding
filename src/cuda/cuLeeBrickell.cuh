#ifndef DECODING_CUDALEEBRICKELL_H
#define DECODING_CUDALEEBRICKELL_H

#include <cstdint>
#include <vector>

#include "helper.cuh"
#include "isd.h"
#include "math/bc.h"

using namespace cryptanalysislib;

/// Important Notes:
///		grid stride loop over all `nr_combinations` combinations of `p` rows of `dH`
/// \tparam width	number of `T` limbs of a row of `dH`
/// \tparam n[in]: code length
/// \tparam k[in]: code dimension
/// \tparam w[in]: max. weight of the error
/// \tparam p[in]: weight to numerate
/// \tparam width[in]: number of limbs of the syndrome
/// \param de 				output: the `p` rows of `dH` of the solution
/// \param dH 				input: transposed working submatrix of size k x n-k
/// \param dss 				input: working syndrome
/// \param dC 				input: combinations of `p` rows, the i-th is dC[i*p], ..., dC[i*p+p-1]
/// \param nr_combinations	number of combinations in `dC`
/// \param compare			output: set to 1 if the solution was found
template<typename T,
         const uint32_t n, 
         const uint32_t k,
         const uint32_t w,
         const uint32_t p,
         const uint32_t width=4>
__global__ void lee_brickell_step(uint32_t *__restrict__ de,
                                  const T *__restrict__ dH,
                                  const T *__restrict__ dss,
                                  const uint16_t *__restrict__ dC,
                                  const uint32_t nr_combinations,
                                  uint32_t *compare) {
	const uint32_t thread_number = blockDim.x * gridDim.x;
	uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ T ds[width];

	if (threadIdx.x < width)
		ds[threadIdx.x] = dss[threadIdx.x];

	__syncthreads();

	while(index < nr_combinations) {
		uint32_t weight = 0;
		for (uint32_t i = 0; (i < width) && (weight <= (w-p)); i++) {
			T t = ds[i];
			for (uint32_t j = 0; j < p; j++) {
				t ^= dH[dC[p*index + j] * width + i];
			}
			weight += (uint32_t) __popc(t);
		}

		if(weight <= (w-p)) {
			if( 0 != atomicCAS(compare,0,1))
				return;
			for (uint32_t j = 0; j < p; j++) {
				de[j] = dC[p*index + j];
			}
			return;
		}

		if (*(volatile uint32_t *)compare)
			return;

		index += thread_number;
	}
}

/// GPU Lee-Brickell: the gaus runs on the CPU, the enumeration in `lee_brickell_step`
/// Usage:
///		cuLeeBrickell<isdConfig> lb{};
///		lb.from_string(h, s);
///		lb.run();
template<const ConfigISD &isd>
class cuLeeBrickell : public ISDInstance<uint64_t, isd> {
public:
	constexpr static uint32_t n = isd.n;
	constexpr static uint32_t k = isd.k;
	constexpr static uint32_t w = isd.w;
	constexpr static uint32_t p = isd.p;

	static_assert(isd.l == 0);
	static_assert((p > 0) && (p < w));

	using ISD    = ISDInstance<uint64_t, isd>;
	using Error  = typename ISD::Error;
	using Label  = typename ISD::Label;
	using ISD::HT,ISD::ws,ISD::e,ISD::P,ISD::not_found,ISD::loops,ISD::expected_loops;

	// `lee_brickell_step` counts the weight with `__popc`
	using T = uint32_t;
	constexpr static uint32_t Hwidth = (n-k+31u)/32u;
	constexpr static uint32_t nr_combinations = bc(k, p);
	constexpr static uint32_t tpb    = 256;
	constexpr static uint32_t blocks = std::min((nr_combinations + tpb - 1u)/tpb, 1024u);

	// rows of `HT` of the solution
	uint32_t solutions[p] = {0};

	T *dHT = nullptr;
	uint16_t *dC = nullptr;
	uint32_t *dSol = nullptr, *dCompare = nullptr;
	std::vector<T> hHT;

	cuLeeBrickell() noexcept : hHT((k+1u)*Hwidth) {
		expected_loops = bc(n, w)/(bc(n-k, w-p) * bc(k, p));

		// all `p` out of `k` combinations in lexicographical order
		std::vector<uint16_t> hC(size_t(nr_combinations)*p);
		uint16_t c[p];
		for (uint32_t j = 0; j < p; ++j) { c[j] = j; }
		for (uint32_t i = 0; i < nr_combinations; ++i) {
			for (uint32_t j = 0; j < p; ++j) { hC[size_t(i)*p + j] = c[j]; }

			int32_t j = p-1;
			while ((j >= 0) && (c[j] == k-p+j)) { j -= 1; }
			if (j < 0) { break; }
			c[j] += 1;
			for (uint32_t t = j+1; t < p; ++t) { c[t] = c[t-1] + 1; }
		}

		checkCudaErrors(cudaMalloc(&dHT, hHT.size()*sizeof(T)));
		checkCudaErrors(cudaMalloc(&dC, hC.size()*sizeof(uint16_t)));
		checkCudaErrors(cudaMalloc(&dSol, p*sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dCompare, sizeof(uint32_t)));
		checkCudaErrors(cudaMemcpy(dC, hC.data(), hC.size()*sizeof(uint16_t), cudaMemcpyHostToDevice));
	}

	~cuLeeBrickell() noexcept {
		cudaFree(dHT); cudaFree(dC); cudaFree(dSol); cudaFree(dCompare);
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

		for (uint32_t i = 0; i < p; ++i) {
			tmpe.set(1u, 0, n-k + solutions[i]);
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
			lee_brickell_step<T, n, k, w, p, Hwidth>
			        <<<blocks, tpb>>>(dSol, dHT, dHT + k*Hwidth, dC, nr_combinations, dCompare);
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
        std::cout << "cuLeeBrickell" << std::endl;
    }
};

#endif // DECODING_CUDALEEBRICKELL_H
