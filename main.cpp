#include <chrono>
#include <cstdint>
#include <iostream>

#include "prange.h"
#include "stern.h"
#include "stern_im.h"
#include "stern_mo.h"
#include "bjmm.h"
#include "may_ozerov.h"

// NOTE: written by `decoding/optimize.py` into the build directory. Defines the
// 	instance (`h`, `s`), `algorithm_type` and the Monte Carlo size
// 	`mc_instances`, `mc_per` (0: none) of the NN success probability.
#include "main.h"

#ifndef MC_INSTANCES
#define MC_INSTANCES 0
#define MC_PER 0
#endif

/// the GPU NN algorithms offer a Monte Carlo of their NN success probability
template<typename A>
concept HasNNSuccessProbability = requires(A &a) { a.nn_success_probability(1u, 1u); };

/// NN success probability per iteration (GPU algorithms, see `optimize.py`)
template<typename A>
void print_nn_success_probability(A *algo) {
	if constexpr (HasNNSuccessProbability<A> && (MC_INSTANCES > 0)) {
		std::cout << "{ \"P_nn\": " << algo->nn_success_probability(MC_INSTANCES, MC_PER) << " }" << std::endl;
	}
}

int main() {
	// NOTE: on the heap, the algorithms are too large for the stack
	auto *algo = new algorithm_type{};
	algo->from_string(h, s);
	// NOTE: the first run contains the one time setup (GPU allocations, graph
	// 	capture), the timed second run only the iterations
	auto t0 = std::chrono::steady_clock::now();
	uint64_t loops = algo->run();
	auto t1 = std::chrono::steady_clock::now();
	if (!algo->correct()) {
		t0 = std::chrono::steady_clock::now();
		loops = algo->run();
		t1 = std::chrono::steady_clock::now();
	}
	const double seconds = std::chrono::duration<double>(t1 - t0).count();
	// NOTE: the `lps` of the final status line includes the Monte Carlo below
	std::cout << "{ \"iterations\": " << loops << ", \"seconds\": " << seconds
	          << ", \"iterations_per_second\": " << (seconds > 0. ? double(loops) / seconds : 0.) << " }" << std::endl;
	algo->print();
	const bool ok = algo->correct();

	print_nn_success_probability(algo);

	delete algo;
	return ok ? 0 : 1;
}
