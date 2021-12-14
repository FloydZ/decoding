#ifndef SMALLSECRETLWE_MCELIECEV2_H
#define SMALLSECRETLWE_MCELIECEV2_H

#include <sys/mman.h>
#include <cstdlib>
#include <strings.h>
#include <sys/random.h>
#include <random>
#include <algorithm>


#include "m4ri/m4ri.h"
#include "m4ri/brilliantrussian.h"

#include "helper.h"
#include "random.h"
#include "combinations.h"
#include "custom_matrix.h"
#include "glue_m4ri.h"
#include "tree.h"

#include "list_generator.h"
#include "list_fill.h"
#include "emz.h"
#include "prange.h"

/// finds the row which is the nearest (hamming weight) to the row `row` on the `t` coordinates
template<const uint32_t n, const uint32_t t, const uint32_t k>
__INLINE__ unsigned int findNN(const mzd_t *A, const uint32_t row) {
	ASSERT(k == A->nrows);

	// naive search
	unsigned int index = -1;
	unsigned int min_dist = hamming_weight_row<n, n+t>(A, row);
	for (unsigned int j = row+1; j < k; ++j) {
		unsigned int weight = hamming_weight_sum_row<n, n+t>(A, row, j);
		if (weight < min_dist) {
			min_dist = weight;
			index = j;
		}
	}
	return index;
}

unsigned int findNN(const mzd_t *A, const uint32_t row, const uint32_t n, const uint32_t t, const uint32_t k) {
	ASSERT(k == A->nrows);

	// naive search
	unsigned int index = -1;
	unsigned int min_dist = hamming_weight_row(A, row, n, n+t);
	for (unsigned int j = row+1; j < k; ++j) {
		unsigned int weight = hamming_weight_sum_row(A, row, j, n, n+t);
		if (weight < min_dist) {
			min_dist = weight;
			index = j;
		}
	}
	return index;
}

/// IMPORTANT: k = #rows
struct ConfigSparsityNN {
public:
	const uint32_t n, k, t, X, thresh1, thresh2;
	const int m4ri_k;
	constexpr ConfigSparsityNN(uint32_t n, uint32_t k, uint32_t t, uint32_t X, uint32_t thresh1, uint32_t thresh2) :
			n(n), k(k), t(t), X(X), thresh1(thresh1), thresh2(thresh2), m4ri_k(matrix_opt_k(k, n)) {}
};

// prints some info about the weight distribution of a sparse matrix
void print_some_info(mzd_t *A, const uint32_t n, const uint32_t t) {
	double w_n = 0.0, w_t = 0.0;
	for (int i = 0; i < n; ++i)   { w_n += hamming_weight_column(A, i); }
	for (int i = n; i < n+t; ++i) { w_t += hamming_weight_column(A, i); }

	w_n /= n; w_t /= t;
	std::cout << "Weight Dist: npart: " << w_n << ", tpart: " << w_t << "\n";
}

void print_some_more_info(mzd_t *A, const uint32_t n, const uint32_t t) {
	double w_n = 0.0, w_t = 0.0;
	for (int i = 0; i < n; ++i)   { w_n += hamming_weight_column(A, i); }
	for (int i = n; i < n+t; ++i) { w_t += hamming_weight_column(A, i); }

	//w_n /= n; w_t /= t;
	std::cout << "Weight Dist: npart: " << w_n/n << ", tpart: " << w_t/t << " whole: " << (w_n + w_t)/(n+t) << "\n";
	std::cout << "[";
	for (int i = 0; i < n+t; ++i) {
		std::cout << hamming_weight_column(A, i);
		if (i != n+t-1)
			std::cout << ", ";
	}
	std::cout << "]\n";
}


/// INPUT MUST BE FULL Matrix
/// \tparam config
/// \param A
/// \return
template<const ConfigSparsityNN &config>
unsigned int sparsityNN(mzd_t *A, customMatrixData *matrix_data) {
	constexpr uint32_t n = config.n;    // #cols
	constexpr uint32_t k = config.k;    // #rows
	constexpr uint32_t t = config.t;    // #of additional cols

	const unsigned int m4ri_k = m4ri_opt_k(k, A->ncols, 0);
	matrix_echelonize_partial(A, m4ri_k, k, matrix_data, 0);
	unsigned int j;

	// for each row
	for (int i = 0; i < k; ++i) {
		j = findNN<n,t,k>(A, i);
//		j = findNN(A, i, n, t, k);

		if (j == -1){ continue; }
		mzd_row_xor(A, i, j);
	}

	for (int i = 0; i < k; ++i) {
		j = findNN<n,t,k>(A, i);
//		j = findNN(A, i, n, t, k);

		if (j == -1){ continue; }
		mzd_row_xor(A, i, j);
	}

	// print_some_more_info(A, n, t);
	return 0;
}

// Indyk Motwani Approach.
template<const ConfigSparsityNN &config>
unsigned int sparsityNN2(mzd_t *A, customMatrixData *matrix_data) {
	constexpr uint32_t n = config.n;    // #cols
	constexpr uint32_t k = config.k;    // #rows
	constexpr uint32_t t = config.t;    // #of additional cols

	const unsigned int m4ri_k = m4ri_opt_k(k, A->ncols, 0);
	matrix_echelonize_partial(A, m4ri_k, k, matrix_data, 0);
	unsigned int j;

	// static const uint32_t iters = 1;
	static uint32_t weights[k];
	static uint64_t random_vector[n/64];

	// for each row
	for (int i = 0; i < k; ++i) {
		mzd_row_random_with_weight(random_vector, (n+t)/64, 4, n, n+t);
		weights[i] = mzd_row_xor_weight(A, i, random_vector);
	}
	for (int i = 0; i < k; ++i) {
		// not finished, but i think this is slower.
	}

	print_some_more_info(A, n, t);
	return 0;
}

/// find the 3NN
template<const ConfigSparsityNN &config>
unsigned int sparsityNN3(mzd_t *A, customMatrixData *matrix_data) {
	// Extract parameters.
	constexpr uint32_t n = config.n;    // #cols
	constexpr uint32_t k = config.k;    // #rows
	constexpr uint32_t t = config.t;    // #of additional cols
	matrix_echelonize_partial(A, config.m4ri_k, k, matrix_data, 0);

	for (int i = 0; i < k; ++i) {
		unsigned int min_j = -1, min_m = -1;
		unsigned int min_w = -1;

		for (int j = 0; j < k; ++j) {
			if (i == j) continue;

			for (int m = 0; m < k; ++m) {
				if (m == j || m == i) continue;

				unsigned int w = hamming_weight_sum_row<n, n + t>(A, i, j, m);
				if (w < min_w) {
					min_w = w;
					min_m = m;
					min_j = j;
				}
			}
		}

		mzd_row_xor(A, i, min_j, min_m);
	}

	int j;
	for (int i = 0; i < k; ++i) {
		j = findNN<n, t, k, 0>(A, i);
		if (j == -1){ continue; }
		mzd_row_xor(A, i, A, j);
	}

//	for (int i = 0; i < k; ++i) {
//		unsigned int min_j = -1, min_m = -1;
//		unsigned int min_w = -1;
//
//		for (int j = 0; j < k; ++j) {
//			if (i == j) continue;
//
//			for (int m = 0; m < k; ++m) {
//				if (m == j || m == i) continue;
//
//				unsigned int w = hamming_weight_sum_row<n, n + t>(A, i, j, m);
//				if (w < min_w) {
//					min_w = w;
//					min_m = m;
//					min_j = j;
//				}
//			}
//		}
//
//		mzd_row_xor(A, i, min_j, min_m);
//	}

	//print_some_info(A, n, t);
	//std::cout << "nothing dound in " << ctr << "\n";
	return 0;
}

/// Our Algorithm
/// +-->
/// |   1) Apply Random permutation
/// |   2) Gauß Elimination (Well actually part of the sparsity algo)
/// |         <--------------+   n   +----------------->
/// |
/// |         <----+n-k+t+-----><-------+ k-t +-------->
/// |                         t
/// |         +--------------+--+----------------------+  +---+
/// |         |              |  |                      |  |   |
/// |         |              |  |                      |  |   |
/// |         |     I_{n-k}  |$$|          H           |  |   |
/// |         |              |$$|                      |  | s | n-k
/// |         |              |  |                      |  |   |
/// |         |              |  |                      |  |   |
/// |         +----------------------------------------+  +---+
/// |         +----------------------------------------+
/// |         |     e1 wt=w     |       e2 wt=p        |
/// |         +-----------------+----------------------+
/// |
/// |   3) Apply Sparsity Algorithm
/// |
/// |         +-----------------+-----------------------+ +---+
/// |         |                 |                       | |   |
/// |		  |                 |                       | |   |
/// |         |       M         |          H`           | |   |
/// |         |                 |                       | | s`|
/// |         |                 |                       | |   |
/// |         |                 |                       | |   |
/// |         +-----------------------------------------+ +---+
/// |         +-----------------------------------------+
/// |         |      e1 wt=w    |         e2  wt=0      |
/// |         +-----------------+-----------------------+
/// |
/// |   4) It holds: Me1 + He2 = s`
/// |   5) if wt(s`) = wt(Me1) < threshold                   Prange Input:
/// |   5.1) Apply Prange Subroutine                               v-----------------+n-k+t+-----------------v
/// |                                                              +-----------------------------------------+
/// |                                                              |                                         |               Only Repeat
/// |                                                              |                                         |
/// |                                                              |                                         |               as long as
/// |                                                              |                    M                    |
/// |                                                              |                                         |               the runtime is
/// |                                                              |                                         |
/// |                                                              |                                         |               below the expected
/// |                                                              +-----------------------------------------+
/// |
/// |                                                           1) apply random permutation                       <------+
/// |                                                           2) Gauß elimination                                      |
/// |                                                                                          v----+ t+-----v           |
/// |                                                              v-----------------n-k+t+------------------>           |
/// |                                                              +-----------------------------------------+ +---+     |
/// |                                                              |                           |             | |   |     |
/// |                                                              |                           |             | |   |     |
/// |                                                              |  I_{n-k}                  |      H``    | |   |     |
/// |                                                              |                           |             | |s``| n-k |
/// |                                                              |                           |             | |   |     |
/// |                                                              |                           |             | |   |     |
/// |                                                              +-----------------------------------------+ +---+     |
/// |                                                              +-----------------------------------------+           |
/// |                                                              |    e11                    |      e12    |           |
/// |                                                              +---------------------------+-------------+           |
/// |                                                                                                                    |
/// |                                                           3) It holds: H`` e12 + s`` = e11                         |
/// |                                                           4) If wt(s``) = w = > e = s``||0^t||0^k-t                |
/// |                                                           5) Repeat for different permutation              +-------+
/// +-------------------------------------------------------+   6) Exit if runtime exceeded


struct PerformanceMcElieceV2 {
public:
	uint64_t loops = 0;
	uint64_t false_positives = 0;

	void print() {
		std::cout << "Sparsity Loops: " << loops << ", false_positives: " <<  false_positives << "\n" << std::flush;
	}

	void reset() {
		loops = 0;
		false_positives = 0;
	}
};

struct ConfigSparsity {
public:
	const uint32_t n, k, w, p, t;   // Instance parameters
	const uint32_t thresh;          // prange threshold
	const uint64_t max_iters;
	constexpr ConfigSparsity(uint32_t n, uint32_t k, uint32_t w, uint32_t p, uint32_t t, uint32_t thresh, uint64_t max_iters=-1) :
		n(n), k(k), w(w), p(p), t(t), thresh(thresh), max_iters(max_iters) {}
};

template<const ConfigSparsity &config>
int Sparsity_thread(mzd_t *e, const mzd_t *const s, const mzd_t *const A, PerformanceMcElieceV2 *perf=nullptr) {
	constexpr uint32_t n = config.n;
	constexpr uint32_t k = config.k;
	constexpr uint32_t w = config.w;
	constexpr uint32_t t = config.t;

	// variables needed for the outer/non-prange loop
	// every thread needs its own copy of the workingmatrix
	mzd_t *work_matrix_H = mzd_init(n - k, n + 1);
	mzd_t *work_matrix_H_T= mzd_init(n + 1, n - k);
	mzd_t *columnTransposed = mzd_transpose(NULL, s);
	mzd_concat(work_matrix_H, A, columnTransposed);
	mzd_transpose(work_matrix_H_T, work_matrix_H);
	mzp_t *P_C = mzp_init(n);
	customMatrixData *matrix_data = init_matrix_data(A->ncols);

	// variables needed for the internal prange loop
	customMatrixData *prange_matrix_data = init_matrix_data(n-k+t);
	prange_matrix_data->working_nr_cols = n;
	mzp_t *prange_C = mzp_init_window(P_C, 0, n-k+t);
	mzd_t *prange_M2 = mzd_init_window(work_matrix_H, 0, 0, n-k,n-k+t);
	mzd_t *prange_M2_T = mzd_init_window(work_matrix_H_T, 0, 0, n-k+t, n-k);

	static constexpr uint64_t num_prange_iters = bc(n-k+t, w)/bc(n-k, w);
	static constexpr ConfigSparsityNN configSparsityNN(n-k, n-k, t, 0, 0,0);
	static constexpr ConfigPrange configPrange(n-k+t, n-k, w, n, num_prange_iters);

	uint64_t loops = 0, false_positives = 0;
	int ret;

	while (true MULTITHREADED_WRITE(&& !finished.load())) {
		loops += 1;

		// create and apply a random permutation to the whole working matrix
		matrix_create_random_permutation(work_matrix_H, work_matrix_H_T, P_C);
		sparsityNN<configSparsityNN>(work_matrix_H, matrix_data);

		// if syndrom is below threshold
		const unsigned int weight = hamming_weight_column(work_matrix_H, config.n);
		if (weight > config.thresh) {
			continue;
		}

        // Automatically unroll a small amount of loops.
        if constexpr (num_prange_iters <= 3) {
            #define SPARSITY_HELPER                                         \
            ret = prange_thread_single<configPrange>::template func<0>      \
                (e, prange_M2, prange_M2_T, prange_C, prange_matrix_data);  \
            if (unlikely((ret) MULTITHREADED_WRITE(&& !finished.load()))) { \
                break;                                                      \
            }                                                               \

            CRYPTANALYSELIB_REPEAT(SPARSITY_HELPER, 3)
            // old approach
            //ret = static_for<num_prange_iters, int, prange_thread_single<configPrange>>(
            //        e, prange_M2, prange_M2_T, prange_C, prange_matrix_data);
            //if (unlikely((ret) MULTITHREADED_WRITE(&& !finished.load()))) {
            //    break;
            //}
        } else {
            ret = prange_thread<configPrange>(e, prange_M2, prange_M2_T, prange_C, prange_matrix_data);
            if (unlikely((ret) MULTITHREADED_WRITE(&& !finished.load()))) {
                break;
            }
        }

		false_positives += 1;
	}

	mzd_free_window(prange_M2);
	mzd_free_window(prange_M2_T);
	mzp_free_window(prange_C);

	mzd_free(work_matrix_H);
	mzd_free(work_matrix_H_T);
	mzd_free(columnTransposed);
	mzp_free(P_C);

	free_matrix_data(matrix_data);
	free_matrix_data(prange_matrix_data);

	//PERFORMANE_WRITE(perf->loops += loops);
	//PERFORMANE_WRITE(perf->false_positives += false_positives);

	// std::cout << "McElieceV2 Loops: " << loops << " * prange loops: " << num_prange_iters << " false_positive: " << false_positive << " " << double(false_positive)/double(loops) * 100.0 << "\n";
	return ret;
}

template<const ConfigSparsity &config>
int Sparsity(mzd_t *e, const mzd_t *const s, const mzd_t *const A, PerformanceMcElieceV2 *perf=nullptr) {
#if (defined(NUMBER_THREADS) && NUMBER_THREADS > 1)
    // reset the mutlithreading stuff
	MULTITHREADED_WRITE(finished.store(false);)
	MULTITHREADED_WRITE(outerloops_all.store(0);)

	int threads_finished = 0;
	std::vector<std::thread> threads{NUMBER_THREADS};
	for( auto & t : threads ) {
		t = std::thread(mc_eliece_v2_thread<config>, e, s, A);
	}
	for (unsigned int i = 0; i < NUMBER_THREADS; i++) {
		threads[i].join();
	}
	std::cout << "number of all perm: " << outerloops_all.load();
	return 1;
#else
	return Sparsity_thread<config>(e, s, A, perf);
#endif
}
#endif //SMALLSECRETLWE_DECODING_H
