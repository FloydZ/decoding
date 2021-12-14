#include "b63.h"
#include "counters/perf_events.h"

#define BaseList_p 1

#include <helper.h>
#include <glue_m4ri.h>
#include <decoding.h>

#include "../test/quasicyclic/challenges/qc54.h"

#define NC n
#define NR (n-k)
#define WEIGHT w

constexpr static bool ss = true;
constexpr static uint64_t mult = 10;

// andres idee
void matrix_apply_random_permutation_andre(mzd_t *A, mzp_t *P_C , rci_t *perm, const int nkl) {
	mzd_t *A_sub = mzd_init_window(A, 0, nkl, A->ncols, A->ncols);
	mzd_t *AT = mzd_init(A_sub->ncols, A_sub->nrows);

	mzd_apply_p_right_trans_even_capped(A, P_C, 0, nkl);
	mzd_transpose(AT, A_sub);

	for (int i = 0; i < AT->nrows; ++i) {
		mzd_row_swap(AT, i, P_C->values[i+nkl]-nkl);
	}
}

// apply full permute on columns
void matrix_apply_random_permutation2(mzd_t *A, mzp_t *P_C) {
	mzd_apply_p_right(A, P_C);
}


// andres idee
B63_BASELINE(bench_matrix_apply_random_permutation, nn) {
	const rci_t nc = NC;
	const rci_t nr = NR;
	mzd_t *A, *AT;
	mzp_t *P;
	static const unsigned int nkl = nc/10;

	rci_t perm[nc];
	int32_t res = 0;

	B63_SUSPEND {
		if (ss) {
			AT = mzd_from_str(n, n - k, h);
			A = mzd_transpose(NULL, AT);
		} else {
			A = mzd_init(nr, nc);
			matrix_generate_random_weighted(A, WEIGHT);
		}
		P = mzp_init(A->ncols+nkl);

	}

	for (int j = 0; j < mult*nn; ++j) {
		matrix_apply_random_permutation_andre(A, P, perm, nkl);
	}

	B63_SUSPEND {
		res += perm[0];
		mzd_free(A);
		mzd_free(AT);
		mzp_free(P);
	}

	// this is to prevent compiler from optimizing res out
	B63_KEEP(res);
}

B63_BENCHMARK(bench_matrix_apply_random_permutation2, nn) {
	const rci_t nc = NC;
	const rci_t nr = NR;
	mzd_t *A, *AT;
	mzp_t *P;
	int32_t res = 0;

	B63_SUSPEND {
		if (ss) {
			AT = mzd_from_str(n, n - k, h);
			A = mzd_transpose(NULL, AT);
		} else {
			A = mzd_init(nr, nc);
			matrix_generate_random_weighted(A, WEIGHT);
		}

		P = mzp_init(A->ncols);
	}

	for (int j = 0; j < mult*nn; ++j) {
		for (int i = 0; i < P->length-1; ++i) {
			word pos = fastrandombytes_uint64() % (P->length - i);

			auto tmp = P->values[i];
			P->values[i] = P->values[i+pos];
			P->values[pos+i] = tmp;
		}

		matrix_apply_random_permutation2(A, P);
	}

	B63_SUSPEND {
		res += P->values[0];
		mzd_free(A);
		mzd_free(AT);
		mzp_free(P);
	}

	// this is to prevent compiler from optimizing res out
	B63_KEEP(res);
}

// heißt jetzt ohne die 3. war aber mal das letzte
B63_BENCHMARK(bench_matrix_apply_random_permutation3, nn) {
	const rci_t nc = NC;
	const rci_t nr = NR;
	mzd_t *A, *AT;
	mzp_t *P;

	int32_t res = 0;

	B63_SUSPEND {
		if (ss) {
			AT = mzd_from_str(n, n - k, h);
			A = mzd_transpose(NULL, AT);
		} else {
			A = mzd_init(nr, nc);
			matrix_generate_random_weighted(A, WEIGHT);
		}
		P = mzp_init(A->ncols);
	}

	for (int i = 0; i < mult*nn; ++i) {
		matrix_create_random_permutation(A, P);
	}

	B63_SUSPEND {
		res += P->values[0];
		mzd_free(A);
		mzd_free(AT);
		mzp_free(P);
	}

	// this is to prevent compiler from optimizing res out
	B63_KEEP(res);
}

B63_BENCHMARK(bench_matrix_apply_random_permutation4, nn) {
	const rci_t nc = NC;
	const rci_t nr = NR;
	mzd_t *A, *AT;
	mzp_t *P;

	int32_t res = 0;

	B63_SUSPEND {
		if (ss) {
			AT = mzd_from_str(n, n - k, h);
			A = mzd_transpose(NULL, AT);
		} else {
			A = mzd_init(nr, nc);
			matrix_generate_random_weighted(A, WEIGHT);
		}
		P = mzp_init(A->ncols);
	}

	for (int i = 0; i < mult*nn; ++i) {
		matrix_create_random_permutation(A, AT, P);
	}

	B63_SUSPEND {
		res += P->values[0];
		mzd_free(A);
		mzd_free(AT);
		mzp_free(P);
	}

	// this is to prevent compiler from optimizing res out
	B63_KEEP(res);
}

B63_BENCHMARK(bench_matrix_apply_random_permutation5, nn) {
	const rci_t nc = NC;
	const rci_t nr = NR;
	mzd_t *A, *AT;
	mzp_t *P;

	int32_t res = 0;

	B63_SUSPEND {
		if (ss) {
			AT = mzd_from_str(n, n - k, h);
			A = mzd_transpose(NULL, AT);
		} else {
			A = mzd_init(nr, nc);
			matrix_generate_random_weighted(A, WEIGHT);
		}
		P = mzp_init(A->ncols);
	}

	for (int j = 0; j < mult*nn; ++j) {
		mzd_transpose(AT, A);
		// dont permute the last column since it is the syndrome
		for (int i = 0; i < P->length-1; ++i) {
			word pos = fastrandombytes_int(nc-i);

			ASSERT(i+pos < P->length);

			auto tmp = P->values[i];
			P->values[i] = P->values[i+pos];
			P->values[pos+i] = tmp;

			mzd_row_swap(AT, i, i+pos);
		}
		mzd_transpose(A, AT);

	}

	B63_SUSPEND {
		res += P->values[0];
		mzd_free(A);
		mzd_free(AT);
		mzp_free(P);
	}

	// this is to prevent compiler from optimizing res out
	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("time,lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cycles,lpe:instructions,lpe:ref-cycles", argc, argv);
	return 0;
}