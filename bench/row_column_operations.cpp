#include "b63.h"
#include "counters/perf_events.h"

#define BaseList_p 1


#include <decoding.h>
#include <helper.h>
#include <label.h>

#define NC 3200
#define NR 1600
#define WEIGHT 32


B63_BASELINE(bench_mpz_row_swap, n) {
	const rci_t nc = NC;
	const rci_t nr = NR;
	static mzd_t *A;
	unsigned int i = 0;
	unsigned int j = 0;
	int32_t res = 0;

	B63_SUSPEND {
		i = rand() % NR;
		j = rand() % NR;

		A = mzd_init(nr, nc);
		if (A == NULL)
			return;

		matrix_generate_random_weighted(A, WEIGHT);
	}

	for (int k = 0; k < n; ++k) {
		mzd_row_swap(A, i, j);
	}

	B63_SUSPEND {
		res += A->rows[0][0];
		mzd_free(A);

	}

	// this is to prevent compiler from optimizing res out
	B63_KEEP(res);
}

B63_BENCHMARK(bench_mzd_col_swap, n) {
	const rci_t nc = NC;
	const rci_t nr = NR;
	static mzd_t *A;
	unsigned int i = 0;
	unsigned int j = 0;
	int32_t res = 0;

	B63_SUSPEND {
		i = rand() % NR;
		j = rand() % NR;

		A = mzd_init(nr, nc);
		matrix_generate_random_weighted(A, WEIGHT);
	}

	for (int k = 0; k < n; ++k) {
		mzd_col_swap(A, i, j);
	}

	B63_SUSPEND {
		res += A->rows[0][0];
		mzd_free(A);
	}

	// this is to prevent compiler from optimizing res out
	B63_KEEP(res);
}


int main(int argc, char **argv) {
	B63_RUN_WITH("time,lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	return 0;
}