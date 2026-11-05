#pragma once

#include <cstdint>
#include <mma.h>

#include "cuStern.cuh"

using namespace nvcuda;

// compile constants
constexpr int WMMA_M = 8;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 128;

// sm_80+ only supports AND-popc, older ones XOR-popc
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define CUSTERN_TENSOR_AND 1
#else
#define CUSTERN_TENSOR_AND 0
#endif

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

/// one warp computes c[i][j] = popc(op(a_i, b_j)), op = AND (sm_80+) or XOR,
/// for 8 rows a_i and 8 rows b_j of K128*128 bits.
/// Layout of `a` and `b`: 128 bit chunk major, e.g. chunk kk of row i is at
/// (kk*rows + i)*4. `a` and `b` must be 256 bit aligned.
/// \tparam K128 number of 128 bit chunks per row
/// \tparam rows number of rows per chunk in `a` and `b`, also the leading dimension of `c`
/// \param a input: first row of the 8 left rows
/// \param b input: first row of the 8 right rows
/// \param c output: 8x8 row major
template<const uint32_t K128, const uint32_t rows>
__device__ __forceinline__ void tensor_8x8x128(const uint32_t *a, const uint32_t *b, int *c) {
	// Declare the fragments
	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::b1, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::b1, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

	wmma::fill_fragment(c_frag, 0);

	#pragma unroll
	for (uint32_t kk = 0; kk < K128; kk++) {
		wmma::load_matrix_sync(a_frag, a + kk*rows*4, WMMA_K);
		wmma::load_matrix_sync(b_frag, b + kk*rows*4, WMMA_K);

		// Perform the matrix multiplication
#if CUSTERN_TENSOR_AND
		wmma::bmma_sync(c_frag, a_frag, b_frag, c_frag, wmma::experimental::bmmaBitOpAND);
#else
		wmma::bmma_sync(c_frag, a_frag, b_frag, c_frag);
#endif
	}

	// Store the output
	wmma::store_matrix_sync(c, c_frag, rows, wmma::mem_row_major);
}

/// one warp per bucket: computes the full weight of all left x right pairs in the
/// bucket with the tensor cores.
/// \tparam T limb type of the rows/columns
/// \tparam k code dimension
/// \tparam l number of bits to match on in the first lvl
/// \tparam w code weight
/// \tparam p number of bits to enumerate in the base lists
/// \tparam width number of limbs each row consists of.
/// \tparam bucket_size number of elements per bucket, multiple of 8
/// \tparam warps number of warps per block
/// \param de output: the `p` left and `p` right list rows, interleaved
/// \param dH input: parity check matrix
/// \param ds input: syndrome
/// \param dC input: combinations of the error vector we are enumerating in the base lists
/// \param pdBucketL input: left list (with syndrome) indices per bucket
/// \param pdCounterL input: left list load per bucket
/// \param pdBucketR input: right list indices per bucket
/// \param pdCounterR input: right list load per bucket
/// \param compare output: set to 1 if the solution was found, 0 otherwise
template<typename T,
         const uint32_t k,
         const uint32_t l,
         const uint32_t w,
         const uint32_t p,
         const uint32_t width,
         const uint32_t bucket_size,
         const uint32_t warps>
__global__ void KernelDumer(uint32_t *__restrict__ de,
                            const T *__restrict__ dH,
                            const T *__restrict__ ds,
                            const uint32_t *__restrict__ dC,
                            const uint32_t *__restrict__ pdBucketL,
                            const uint32_t *__restrict__ pdCounterL,
                            const uint32_t *__restrict__ pdBucketR,
                            const uint32_t *__restrict__ pdCounterR,
                            uint32_t *compare) {
	static_assert(sizeof(T) == sizeof(uint32_t));
	static_assert((bucket_size % WMMA_M) == 0);
	constexpr uint32_t kl_half = (k+l)/2;
	constexpr uint32_t weightt = w - (2u*p);
	constexpr uint32_t K128    = (width + 3u)/4u;           // 128 bit chunks per row
	constexpr uint32_t K32     = K128*4u;                   // 32 bit words per row
	constexpr uint32_t tiles   = bucket_size / WMMA_M;

	__shared__ __align__(32) uint32_t sA[warps][K32*bucket_size], sB[warps][K32*bucket_size];
	__shared__ __align__(32) int sC[warps][bucket_size*bucket_size];
	__shared__ uint32_t sWL[warps][bucket_size], sWR[warps][bucket_size];

	const uint32_t warp     = threadIdx.x / 32u;
	const uint32_t lane     = threadIdx.x % 32u;
	const uint32_t nr_warps = gridDim.x * warps;

	for (uint32_t b = blockIdx.x*warps + warp; b < (1u << l); b += nr_warps) {
		// stop the whole warp, if any thread found the solution
		const uint32_t done = __shfl_sync(0xffffffffu, lane == 0 ? *(volatile uint32_t *)compare : 0u, 0);
		if (done) { break; }

		const uint32_t nl = min(bucket_size, pdCounterL[b]);
		const uint32_t nr = min(bucket_size, pdCounterR[b]);
		if ((nl == 0) || (nr == 0)) { continue; }

		// load s + H*e1 (left) and H*e2 (right), zero padded
		for (uint32_t i = lane; i < K32*bucket_size; i += 32u) {
			const uint32_t row = i / K32, word = i % K32;
			const uint32_t pos = ((word/4u)*bucket_size + row)*4u + (word%4u);
			T a = 0, c = 0;
			if (word < width) {
				if (row < nl) {
					const uint32_t idx = pdBucketL[b*bucket_size + row];
					a = ds[word];
					for (uint32_t r = 0; r < p; r++) {
						a ^= dH[dC[idx*p + r]*width + word];
					}
				}
				if (row < nr) {
					const uint32_t idx = pdBucketR[b*bucket_size + row];
					for (uint32_t r = 0; r < p; r++) {
						c ^= dH[(kl_half + dC[idx*p + r])*width + word];
					}
				}
			}
			sA[warp][pos] = a;
			sB[warp][pos] = c;
		}
		__syncwarp();

		// weight of each element, needed for the AND trick
		if (lane < bucket_size) {
			uint32_t wl = 0, wr = 0;
			for (uint32_t word = 0; word < K32; word++) {
				const uint32_t pos = ((word/4u)*bucket_size + lane)*4u + (word%4u);
				wl += __popc(sA[warp][pos]);
				wr += __popc(sB[warp][pos]);
			}
			sWL[warp][lane] = wl;
			sWR[warp][lane] = wr;
		}

		// all pairs with the tensor cores
		#pragma unroll
		for (uint32_t ti = 0; ti < tiles; ti++) {
			#pragma unroll
			for (uint32_t tj = 0; tj < tiles; tj++) {
				tensor_8x8x128<K128, bucket_size>(&sA[warp][ti*WMMA_M*4u], &sB[warp][tj*WMMA_N*4u],
				                                  &sC[warp][ti*WMMA_M*bucket_size + tj*WMMA_N]);
			}
		}
		__syncwarp();

		for (uint32_t i = lane; i < bucket_size*bucket_size; i += 32u) {
			const uint32_t il = i / bucket_size, ir = i % bucket_size;
			if ((il >= nl) || (ir >= nr)) { continue; }

#if CUSTERN_TENSOR_AND
			const uint32_t weight = sWL[warp][il] + sWR[warp][ir] - 2u*uint32_t(sC[warp][i]);
#else
			const uint32_t weight = uint32_t(sC[warp][i]);
#endif
			// if we pass this check we found a solution
			if ((weight <= weightt) && (0 == atomicCAS(compare, 0, 1))) {
				const uint32_t index_left  = pdBucketL[b*bucket_size + il];
				const uint32_t index_right = pdBucketR[b*bucket_size + ir];
				for (uint32_t r = 0; r < p; r++) {
					de[2*r + 0] = dC[index_left *p + r];
					de[2*r + 1] = dC[index_right*p + r];
				}
			}
		}
		__syncwarp();
	}
}

/// same as `cuStern`, but the right list is bucketed too and every bucket is
/// matched on the full length with the tensor cores in `KernelDumer`.
/// Usage:
///		cuSternTensor<isdConfig, config> stern{};
///		stern.from_string(h, s);
///		stern.run();
template<const ConfigISD &isd, const ConfigStern &config>
class cuSternTensor : public cuStern<isd, config> {
public:
	using Base   = cuStern<isd, config>;
	using ISD    = typename Base::ISD;
	using l_type = typename Base::l_type;
	using T      = typename Base::T;
	using Base::n, Base::k, Base::w, Base::p, Base::l, Base::kl_half, Base::kl, Base::Hwidth;
	using Base::nr_combinations, Base::bucket_size, Base::tpb, Base::blocks;
	using Base::solutions, Base::ws, Base::syndrome, Base::lHT, Base::not_found, Base::loops;
	using Base::hHT, Base::dHT, Base::dlHT, Base::dC, Base::dBucket, Base::dCounter, Base::dSol, Base::dCompare;

	static_assert((bucket_size % WMMA_M) == 0, "bucket size must be a multiple of 8");

	constexpr static uint32_t warps         = 4;
	constexpr static uint32_t tensor_blocks = std::min(((1u << l) + warps - 1u)/warps, 4096u);

	// buckets of the right list
	uint32_t *dBucketR = nullptr, *dCounterR = nullptr;

	cuSternTensor() noexcept : Base() {
		checkCudaErrors(cudaMalloc(&dBucketR, (size_t(1) << l)*bucket_size*sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dCounterR, (size_t(1) << l)*sizeof(uint32_t)));
	}

	~cuSternTensor() noexcept {
		cudaFree(dBucketR); cudaFree(dCounterR);
	}

	// same as `cuStern::run`, but the matching runs on the tensor cores
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
			checkCudaErrors(cudaMemset(dCounterR, 0, (size_t(1) << l)*sizeof(uint32_t)));

			// left list: s + H*e1, right list: H*e2
			dumer_step1<T, k, l, p, bucket_size>
			        <<<blocks, tpb>>>(dlHT, T(syndrome), dC, dBucket, dCounter, nr_combinations);
			dumer_step1<T, k, l, p, bucket_size>
			        <<<blocks, tpb>>>(dlHT + kl_half, T(0), dC, dBucketR, dCounterR, nr_combinations);
			KernelDumer<T, k, l, w, p, Hwidth, bucket_size, warps>
			        <<<tensor_blocks, warps*32u>>>(dSol, dHT, dHT + kl*Hwidth, dC,
			                                       dBucket, dCounter, dBucketR, dCounterR, dCompare);
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
