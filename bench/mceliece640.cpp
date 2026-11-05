#include <benchmark/benchmark.h>

#include "mce640.h"
#include "prange.h"
#include "stern.h"
#include "stern_im.h"
#include "bjmm.h"

// NOTE: every benchmark iteration runs (at most) `bench_loops` ISD iterations.
// 	The algorithms stop early if they find the solution, hence the number of
// 	iterations is reported as `items_per_second`.
constexpr uint64_t bench_loops = 1024;

static void BM_Prange(benchmark::State &state) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=0,.l=0,.c=20,.threads=1,.loops=bench_loops};

	// NOTE: on the heap, the algorithms are too large for the stack
	auto *algo = new Prange<isdConfig>{};
	algo->from_string(h, s);
	uint64_t loops = 0;
	for (auto _ : state) {
		loops += algo->run();
	}

	state.SetItemsProcessed(loops);
	delete algo;
}

static void BM_Stern(benchmark::State &state) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=16,.c=20,.threads=1,.loops=bench_loops};
	static constexpr ConfigStern config{isdConfig, 16};

	auto *algo = new Stern<isdConfig, config>{};
	algo->from_string(h, s);
	uint64_t loops = 0;
	for (auto _ : state) {
		loops += algo->run();
	}

	state.SetItemsProcessed(loops);
	delete algo;
}

static void BM_SternIM(benchmark::State &state) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=19,.c=20,.threads=1,.loops=bench_loops};
	static constexpr ConfigStern configStern{isdConfig, 16};
	static constexpr ConfigSternIM config{isdConfig, 2};

	auto *algo = new SternIM<isdConfig, configStern, config>{};
	algo->from_string(h, s);
	uint64_t loops = 0;
	for (auto _ : state) {
		loops += algo->run();
	}

	state.SetItemsProcessed(loops);
	delete algo;
}

static void BM_BJMM(benchmark::State &state) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=16,.c=20,.threads=1,.loops=bench_loops};
	static constexpr ConfigBJMM config{isdConfig, 2, 1<<6, 20};

	auto *algo = new BJMM<isdConfig, config>{};
	algo->from_string(h, s);
	uint64_t loops = 0;
	for (auto _ : state) {
		loops += algo->run();
	}

	state.SetItemsProcessed(loops);
	delete algo;
}

BENCHMARK(BM_Prange)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Stern)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SternIM)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BJMM)->Unit(benchmark::kMillisecond);
