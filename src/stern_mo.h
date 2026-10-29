#ifndef DECODING_STERNMO
#define DECODING_STERNMO

#include <cstdint>
#include <utility>

#include "isd.h"
#include "stern.h"

struct ConfigSternMO : public ConfigISD {
public:
	/// Well, ... currently this forces the windowsize `k` = 64.
	const uint32_t r = 3;//(n-k + 63) / 64;
	const uint32_t N = 50; //
	const uint32_t dk = 12;// (w-2*p) + 6;
	const uint32_t nnk = 32;

	// debug mode. in case needed, we can enforce bruteforce
	const bool enforce_bf = false;
	// runs the NN step on the GPU, needs a CUDA compiler
	const bool cuda = false;
	// GPU: flat tree (`r` levels, `N` <= 64) instead of the recursive
	// one, see `nn_flat_level` in nn/nn.cuh. mce1284 p=2: r=2 N=64 dk=4 nnk=16
	// (RTX 5090: 131 us/iter, P = 0.63, vs 339 us/iter, P = 0.31 recursive)
	const bool cuda_flat = false;
	[[nodiscard]] consteval uint64_t compute_loops() const noexcept {
#ifdef EXPECTED_PERMUTATIONS
		return EXPECTED_PERMUTATIONS;
#else
		return bc(n, w)/( bc(n-k, w-2*p) * bc(k/2, p) * bc(k/2, p));
#endif
	}

	void print() const noexcept {
		ConfigISD::print();
		std::cout << "{ "
		          << "\"r\": " << r
				  << ", \"N\": " << N
				  << ", \"dk\": " << dk
				  << ", \"nnk\": " << nnk
				  << ", \"cuda\": " << cuda
				  << ", \"cuda_flat\": " << cuda_flat
				  << " }" << std::endl;
	}
};

#ifdef USE_CUDA
/// builds the two base lists on the GPU, same as
/// `SternMO::construct_nearest_neighbour_lists`: element `i` of `L1` is the
/// sum of the `p` rows `biject(i)` of the left half of `HT`, of `L2` the sum
/// of the same rows of the right half plus the syndrome.
/// \param HT `2*half` rows of `limbs` limbs each, followed by the syndrome
template<typename T, const uint32_t limbs, const uint32_t half, const uint32_t p>
__global__ void sternmo_lists(T *L1, T *L2, const T *HT, const uint32_t size) {
	static_assert((p >= 1) && (p <= 4) && (p <= half));
	const T *ws = HT + 2u * half * limbs;
	const uint32_t stride = gridDim.x * blockDim.x;
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
		// same enumeration as `biject`: rows[j] is the max. c < rows[j-1]
		// with bc(c, p-j) <= x
		uint32_t rows[p];
		uint32_t x = i;
		#pragma unroll
		for (uint32_t j = 0; j < p; j++) {
			const uint32_t pj = p - j;
			uint32_t c;
			if (pj == 1u) {
				rows[j] = x;
				break;
			} else if (pj == 2u) {
				c = (uint32_t(sqrtf(8.f * float(x) + 1.f)) + 1u) / 2u;
			} else if (pj == 3u) {
				// bc(c, 3) ~ (c - 1)^3 / 6
				c = uint32_t(cbrtf(6.f * float(x))) + 1u;
			} else {
				// bc(c, 4) = ((c^2 - 3c + 1)^2 - 1) / 24
				c = (uint32_t(sqrtf(4.f * floorf(sqrtf(24.f * float(x) + 1.f)) + 5.f)) + 3u) / 2u;
			}

			// NOTE: fixes rounding errors of the `float` estimate.
			// bc(c, pj) is exact in each step of the product.
			uint64_t b;
			while (true) {
				b = 1;
				#pragma unroll
				for (uint32_t t = 0; t < pj; t++) { b = b * (c - t) / (t + 1u); }
				if (b <= x) { break; }
				c--;
			}
			while (true) {
				uint64_t b1 = 1;
				#pragma unroll
				for (uint32_t t = 0; t < pj; t++) { b1 = b1 * (c + 1u - t) / (t + 1u); }
				if (b1 > x) { break; }
				b = b1;
				c++;
			}
			rows[j] = c;
			x -= uint32_t(b);
		}

		T l[limbs], r[limbs];
		for (uint32_t j = 0; j < limbs; j++) {
			l[j] = 0;
			r[j] = ws[j];
		}
		for (uint32_t a = 0; a < p; a++) {
			for (uint32_t j = 0; j < limbs; j++) {
				l[j] ^= HT[rows[a] * limbs + j];
				r[j] ^= HT[(rows[a] + half) * limbs + j];
			}
		}
		for (uint32_t j = 0; j < limbs; j++) {
			L1[size_t(i) * limbs + j] = l[j];
			L2[size_t(i) * limbs + j] = r[j];
		}
	}
}
#endif

template<const ConfigISD &isd, const ConfigSternMO &config>
class SternMO : public ISDInstance<uint64_t, isd>{
public:
	constexpr static uint32_t n = isd.n;
	constexpr static uint32_t k = isd.k;
	constexpr static uint32_t w = isd.w;
	constexpr static uint32_t p = isd.p;
	constexpr static uint32_t l = isd.l;

	constexpr static uint32_t r = config.r;

	using ISD = ISDInstance<uint64_t, isd>;
	using PCSubMatrix_T = ISD::PCSubMatrix_T;
	using Error 		= ISD::Error;
	using Label 		= ISD::Label;
	using limb_type 	= ISD::limb_type;
	using ISD::A,ISD::H,ISD::wA,ISD::wAT,ISD::HT,ISD::s,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::ghz,ISD::expected_loops;
	using ISD::cycles, ISD::gaus_cycles, ISD::periodic_print, ISD::packed, ISD::simd;

	// if set to
	constexpr static uint32_t kl_half =  k/2;
	constexpr static uint32_t NR_HT_LIMBS = PCSubMatrix_T::limbs();

	// baselist
	constexpr static size_t baselist_enumeration_length = kl_half;
	constexpr static size_t baselist_enumeration_weigth = p;
	constexpr static size_t baselist_enumeration_size =
			bc(baselist_enumeration_length, baselist_enumeration_weigth);
	constexpr static size_t size_MO_L = roundToAligned<1024>(baselist_enumeration_size*sizeof(Label));
	Label *stern_MO_L1 = nullptr, *stern_MO_L2 = nullptr;


	constexpr static NN_Config nn_config{
	        n-k, config.r, config.N, config.nnk,
	        baselist_enumeration_size, config.dk, w-(2*p), 0, 1024,
	        0, 0, false, config.cuda_flat};
	NN<nn_config> algo {};
	using NNElement = typename NN<nn_config>::Element;

	size_t solutions[2*p] = {0};

#ifdef USE_CUDA
	// GPU: per ISD iteration in flight (2) a pinned copy of `HT` and `ws`
	// (`size_HT` limbs) and of the permutation. `HT` and `ws` on the GPU.
	constexpr static uint32_t nr_limbs = NN<nn_config>::ELEMENT_NR_LIMBS;
	constexpr static size_t size_HT = (2 * kl_half + 1) * nr_limbs;
	limb_type *hHT[2] = {nullptr, nullptr}, *dHT = nullptr;
	uint32_t *hP[2] = {nullptr, nullptr};
	cudaStream_t stream = nullptr;
	// `cuda_flat`: the GPU work of one iteration per slot as a CUDA graph
	cudaGraphExec_t graph[2] = {nullptr, nullptr};

	~SternMO() noexcept {
		// NOTE: no `checkCudaErrors`, it calls `exit`
		for (uint32_t i = 0; i < 2; i++) {
			if (hHT[i]) { cudaFreeHost(hHT[i]); }
			if (hP[i]) { free(hP[i]); }
			if (graph[i]) { cudaGraphExecDestroy(graph[i]); }
		}
		if (dHT) { cudaFree(dHT); }
		if (stream) { cudaStreamDestroy(stream); }
	}
#endif

	constexpr SternMO() noexcept {
		nn_config.print();
		expected_loops = config.compute_loops();

		stern_MO_L1 = (Label *)cryptanalysislib::aligned_alloc(1024, size_MO_L);
		stern_MO_L2 = (Label *)cryptanalysislib::aligned_alloc(1024, size_MO_L);

		algo.L1 = (NNElement *)stern_MO_L1;
		algo.L2 = (NNElement *)stern_MO_L2;
	}

	// NOTE: `stern_MO_L1/L2` are owned by `algo`, which frees them in `~NN`.

	/// reconstruct the solution
	void __attribute__ ((noinline))
	reconstruct() {
		// we need to translate the indices from the NN to stern list indices.
		const size_t soll = solutions[0];
		const size_t solr = solutions[1];
		uint16_t rows[p];
		Label tmpl, tmpr;
		bool foundl = false, foundr = false;

		for (size_t i = 0; i < baselist_enumeration_size; i++) {
			biject<baselist_enumeration_length, p>(i, rows);
			bool correctl = true;
			bool correctr = true;

			tmpl.zero(); tmpr = ws;

			for (uint16_t j = 0; j < p; j++) {
				assert(rows[j] < kl_half);
				Label::add(tmpl.ptr(), tmpl.ptr(), HT[rows[j]]);
				Label::add(tmpr.ptr(), tmpr.ptr(), HT[rows[j] + baselist_enumeration_length]);
			}

			if (!tmpl.is_equal(((Label*)algo.L1)[soll])) {
				correctl = false;
			}
			if (!tmpr.is_equal(((Label*)algo.L2)[solr])) {
				correctr = false;
			}

			if (correctl) {
				assert(!foundl);
				foundl = true;
				for (uint32_t j = 0; j < p; j++) {
					solutions[j] = rows[j];
				}
			}

			if (correctr) {
				assert(!foundr);
				foundr = true;
				for (uint32_t j = 0; j < p; j++) {
					solutions[p + j] = rows[j] + baselist_enumeration_length;
				}
			}
		}

		assert(foundl);
		assert(foundr);
		(void)foundl;
		(void)foundr;

		Error tmpe, tmpe2;
		Label tmp; tmp.zero();
		// copy in the syndrome
		tmp = ws;
		for (uint32_t i = 0; i < 2*p; ++i) {
			assert(solutions[i] < k+l);
			Label::add(tmp.ptr(), tmp.ptr(), HT.row(solutions[i]));
		}

		// reversing and shit
		for (uint32_t i = 0; i < n-k; ++i) {
			const auto bit = tmp.get(i);
			tmpe.set(bit, 0, n-k-1-i);
		}

		for (uint32_t i = 0; i < 2*p; ++i) {
			assert(n-k-l + solutions[i] < n);
			tmpe.set(1, 0, n-k-l + solutions[i]);
		}

		// apply back permutation.
		for (uint32_t i = 0; i < n; ++i) {
			e.set(tmpe.get(0, i), 0, P.values[i]);
		}
	}


	///
	constexpr void construct_nearest_neighbour_lists() noexcept {
		alignas(32) Label tmpl, tmpr;
		alignas(32) uint16_t rows[p];

		for (size_t i = 0; i < baselist_enumeration_size; i++) {
			biject<baselist_enumeration_length, p>(i, rows);
			tmpl.zero(); tmpr = ws;

			#pragma unroll
			for (uint16_t j = 0; j < p; j++) {
				assert(rows[j] < kl_half);
				Label::add(tmpl.ptr(), tmpl.ptr(), HT[rows[j]]);
				Label::add(tmpr.ptr(), tmpr.ptr(), HT[rows[j] + baselist_enumeration_length]);
			}

			stern_MO_L1[i] = tmpl;
			stern_MO_L2[i] = tmpr;
		}
	}

	/// generates the two lists L1 and L2 from HT
	constexpr void apply_nearest_neighbour() noexcept {
		if constexpr (config.enforce_bf) {
			algo.bruteforce(baselist_enumeration_size, baselist_enumeration_size);
		} else {
			algo.nn(baselist_enumeration_size, baselist_enumeration_size);
		}
		if (algo.solutions_nr > 0) {
			solutions[0] = algo.solutions[0].first;
			solutions[1] = algo.solutions[0].second;

			not_found = false;
		}
	}

#ifdef USE_CUDA
	/// Monte Carlo estimate of the success probability of one NN iteration
	/// (`cuda` only), see `NN::cuda_success_probability`. Overwrites the GPU lists.
	double nn_success_probability(const uint32_t instances, const uint32_t per) noexcept {
		if constexpr (config.cuda) {
			if (stream == nullptr) { checkCudaErrors(cudaStreamCreate(&stream)); }
			return algo.cuda_success_probability(baselist_enumeration_size, n - k, instances, per, stream);
		} else {
			return -1.;
		}
	}
#endif

	/// runs the May-Ozerov Stern versions
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();

#ifdef USE_CUDA
		if constexpr (config.cuda) {
			static_assert(isd.q == 2);
			static_assert(sizeof(Label) == sizeof(NNElement));
			algo.cuda_alloc();
			if (dHT == nullptr) {
				checkCudaErrors(cudaStreamCreate(&stream));
				for (uint32_t i = 0; i < 2; i++) {
					checkCudaErrors(cudaHostAlloc(&hHT[i], size_HT * sizeof(limb_type), cudaHostAllocDefault));
					hP[i] = (uint32_t *)malloc(n * sizeof(uint32_t));
				}
				checkCudaErrors(cudaMalloc(&dHT, size_HT * sizeof(limb_type)));
			}

			// one ISD iteration on the CPU, `HT`, `ws` and `P` are saved in slot `s`
			auto step = [&](const uint32_t s) {
				ISD::step();
				// NOTE: `std::`, `cryptanalysislib::memcpy` counts elements, not bytes
				for (uint32_t i = 0; i < 2 * kl_half; i++) {
					std::memcpy(hHT[s] + i * nr_limbs, HT[i], nr_limbs * sizeof(limb_type));
				}
				std::memcpy(hHT[s] + 2 * kl_half * nr_limbs, ws.ptr(), nr_limbs * sizeof(limb_type));
				std::memcpy(hP[s], P.values, n * sizeof(uint32_t));
			};

			// builds the lists of slot `s` and runs the NN on them, async on the GPU
			auto enqueue = [&](const uint32_t s) {
				constexpr uint32_t threads = 256;
				constexpr uint32_t blocks = (baselist_enumeration_size + threads - 1) / threads;
				checkCudaErrors(cudaMemcpyAsync(dHT, hHT[s], size_HT * sizeof(limb_type), cudaMemcpyHostToDevice, stream));
				sternmo_lists<limb_type, nr_limbs, kl_half, p><<<blocks, threads, 0, stream>>>
					((limb_type *)algo.dL1, (limb_type *)algo.dL2, dHT, baselist_enumeration_size);
				checkCudaErrors(cudaGetLastError());
				algo.cuda_launch(baselist_enumeration_size, baselist_enumeration_size, stream);
			};
			// NOTE: `cuda_flat`: the ~10 launches per iteration cost ~40 us of
			// 	CPU time, as one graph per slot ~5 us. The recursive tree draws
			// 	its seed on the host per launch, hence it is not captured.
			auto launch = [&](const uint32_t s) {
				if constexpr (config.cuda_flat) {
					if (graph[s] == nullptr) {
						cudaGraph_t g;
						checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
						enqueue(s);
						checkCudaErrors(cudaStreamEndCapture(stream, &g));
						checkCudaErrors(cudaGraphInstantiate(&graph[s], g, 0));
						checkCudaErrors(cudaGraphDestroy(g));
					}
					checkCudaErrors(cudaGraphLaunch(graph[s], stream));
				} else {
					enqueue(s);
				}
			};

			uint32_t s = 0;
			step(s);
			launch(s);
			while (true) {
				// NOTE: the next iteration runs on the CPU, while the GPU computes the NN
				const bool next = loops < isd.loops;
				if (next) { step(s ^ 1u); }

				algo.cuda_finish(stream);
				if (algo.solutions_nr > 0) {
					// restore the iteration of the solution
					for (uint32_t i = 0; i < 2 * kl_half; i++) {
						std::memcpy(HT[i], hHT[s] + i * nr_limbs, nr_limbs * sizeof(limb_type));
					}
					std::memcpy(ws.ptr(), hHT[s] + 2 * kl_half * nr_limbs, nr_limbs * sizeof(limb_type));
					std::memcpy(P.values, hP[s], n * sizeof(uint32_t));
					loops -= next;

					solutions[0] = algo.solutions[0].first;
					solutions[1] = algo.solutions[0].second;
					not_found = false;
					break;
				}

				if (!next) { break; }
				s ^= 1u;
				launch(s);
			}
		} else
#else
		static_assert(!config.cuda, "`cuda` needs `USE_CUDA`");
#endif
		while (not_found && loops < isd.loops) {
			ISD::step();
			construct_nearest_neighbour_lists();
			apply_nearest_neighbour();
		}

		// `reconstruct` asserts that a solution exists
		if (!not_found) {
			reconstruct();
		}
		return loops;
	}

	/// important. Dont rename it
	constexpr void info() const noexcept {
		std::cout << "{"
				  << " \"size_MO_L\": " << size_MO_L
				  << " }" << std::endl;
	}
};
#endif
