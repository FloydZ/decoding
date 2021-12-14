#include "b63.h"
#include "counters/perf_events.h"


#include <helper.h>
#include <glue_m4ri.h>
#include <custom_matrix.h>

#include "m4ri/m4ri.h"
#include "../test/quasicyclic/challenges/qc46.h"
//#include "../test/quasicyclic/challenges/qc10.h"

// andres idee
B63_BASELINE(M4ri, nn) {
	mzd_t *A, *AT, *B;
	uint64_t ret = 0;

	B63_SUSPEND {
		AT = mzd_from_str(n, n-k, h);
		A = mzd_transpose(NULL, AT);
		B = mzd_init(A->nrows, A->ncols);
	}

	for (int j = 0; j < 100*nn; ++j) {
		mzd_copy(B, A);
		//ret += mzd_echelonize_m4ri(B, 0, 4);
		ret += mzd_echelonize(B, 1);
	}

	B63_SUSPEND {
		mzd_free(A);
		mzd_free(B);
	}
	B63_KEEP(ret);
}

B63_BENCHMARK(matrix, nn) {
	mzd_t *A, *AT, *B;
	customMatrixData *matrix_data;
	uint64_t ret = 0;

	B63_SUSPEND {
		AT = mzd_from_str(n, n-k, h);
		A = mzd_transpose(NULL, AT);
		B = mzd_init(A->nrows, A->ncols);
		matrix_data = init_matrix_data(A->ncols);
	}

	for (int j = 0; j < 100*nn; ++j) {
		mzd_copy(B, A);
		ret += matrix_echelonize_partial(B, 5, n - k - 18, matrix_data, 0);
	}

	B63_SUSPEND {
		mzd_free(A);
		mzd_free(B);
		free_matrix_data(matrix_data);
	}
	B63_KEEP(ret);
}
int main(int argc, char **argv) {
	B63_RUN_WITH("time,lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cycles,lpe:instructions,lpe:ref-cycles", argc, argv);
	return 0;
}