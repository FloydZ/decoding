#include "b63.h"
#include "counters/perf_events.h"
#include <helper.h>
#include <custom_matrix.h>


#define CUSTOM_ALIGNMENT 4096
#define BINARY_CONTAINER_ALIGNMENT
#define USE_AVX2_SPECIAL_ALIGNMENT 4096

#include "m4ri/m4ri.h"
#include "../test/mceliece/challenges/mce1284.h"
#include "container.h"

using Container = BinaryContainer<n>;

constexpr uint64_t ulimb = 5;
constexpr uint64_t umask = 9283749;
constexpr uint32_t early_exit = 20;


B63_BASELINE(add_only_upper_weight_partly, nn) {
	Container A, B, C;
	uint64_t weight = 0;
	B63_SUSPEND {
		A.random();
		B.random();
	}

	for (int i = 0; i < 10*nn; ++i) {
		weight += Container::add_only_upper_weight_partly<ulimb, umask>(C, A, B);
		B63_SUSPEND {
			A.random();
			B.random();
		}
	}

	B63_KEEP(weight);
}

B63_BENCHMARK(add_only_upper_weight_partly_withoutasm, nn) {
	Container A, B, C;
	uint64_t weight = 0;
	B63_SUSPEND {
		A.random();
		B.random();
	}

	for (int i = 0; i < 10*nn; ++i) {
		weight += Container::add_only_upper_weight_partly_withoutasm<ulimb, umask>(C, A, B);
		B63_SUSPEND {
			A.random();
			B.random();
		}
	}

	B63_KEEP(weight);
}

B63_BENCHMARK(add_only_upper_weight_partly_withoutasm_earlyexit, nn) {
	Container A, B, C;
	uint64_t weight = 0;
	B63_SUSPEND {
		A.random();
		B.random();
	}

	for (int i = 0; i < 10*nn; ++i) {
		weight += Container::add_only_upper_weight_partly_withoutasm_earlyexit<ulimb, umask, early_exit>(C, A, B);
		B63_SUSPEND {
			A.random();
			B.random();
		}
	}

	B63_KEEP(weight);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("time,lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cycles,lpe:instructions,lpe:ref-cycles", argc, argv);
	return 0;
}