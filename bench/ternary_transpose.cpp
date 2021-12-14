#include "b63.h"
#include "counters/perf_events.h"

#define CUSTOM_ALIGNMENT 4096
#define BINARY_CONTAINER_ALIGNMENT
#define USE_AVX2_SPECIAL_ALIGNMENT 4096
#define NUMBER_THREADS 1

#include <helper.h>
#include <ternary.h>

constexpr uint64_t n = 200;
using Container = kAryPackedContainer_T<uint64_t , 3, n>;
using TerMatrix = TernaryMatrix<Container>;

B63_BASELINE(ternary_transpose, nn) {
	TerMatrix A(n, n), B(n, n);
	B63_SUSPEND {
		A.random();
		B.random();
	}

	for (int i = 0; i < nn; ++i) {
			TerMatrix::transpose(B, A);
	}

	auto data = uint64_t (B.get(0, 0));
	B63_KEEP(data);
}

B63_BENCHMARK(ternary_transpose2, nn) {
	TerMatrix A(n, n), B(n, n);
	B63_SUSPEND {
		A.random();
		B.random();
	}

	for (int i = 0; i < nn; ++i) {
		TerMatrix::transpose2(B, A);
	}

	auto data = uint64_t (B.get(0, 0));
	B63_KEEP(data);

}


int main(int argc, char **argv) {
	B63_RUN_WITH("time,lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cycles,lpe:instructions,lpe:ref-cycles", argc, argv);
	return 0;
}