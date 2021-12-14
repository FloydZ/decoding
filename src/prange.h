#ifndef SMALLSECRETLWE_PRANGE_H
#define SMALLSECRETLWE_PRANGE_H

#include "decoding.h"

// IMPORTANT: Due to implementation difficulties i have to rename the input parameters
//  n=#cols
//  k=#rows AND NOT n-k=#rows
struct ConfigPrange {
public:
	const uint32_t n, k, w;
	const uint64_t max_iters;
	const int m4ri_k;
	const uint32_t syndrom_col;
	constexpr ConfigPrange(uint32_t n, uint32_t k, uint32_t w, uint32_t syndrom_col, uint64_t max_iters=0) :
			n(n), k(k), w(w), syndrom_col(syndrom_col), max_iters(max_iters == 0 ? bc(n, w)/bc(k, w) : max_iters),
			m4ri_k(matrix_opt_k(k, n)) {}
};

/// TODO describe
struct PerformancePrange {
public:
	uint64_t loops = 0;

	// Simple gives a little bit of output
	void print() {
		std::cout << "Prange Loops: " << loops << "\n" << std::flush;
	}

	void reset() {
		loops = 0;
	}

	void expected(const ConfigPrange &config) {
		uint64_t expected_loops = bc(config.n, config.w)/bc(config.k, config.w);
		std::cout << "Prange Loops/Expected: " << loops << "/" << expected_loops << "\n" << std::flush;
	}
};

/// Metaprogramming test:
template<const ConfigPrange &config>
struct prange_thread_single {
	template<int index>
	static int func(mzd_t *e, mzd_t *work_matrix_H, mzd_t *work_matrix_H_T, mzp_t *P_C,
				    customMatrixData *matrix_data) {
		int ret = 0;
		matrix_create_random_permutation(work_matrix_H, work_matrix_H_T, P_C);
		int ra = matrix_echelonize_partial(work_matrix_H, config.m4ri_k, config.k, matrix_data, 0);
        // update the Gauß algorithm, if it was not possible to fully solve it.
        for (int b = ra; b < config.k; ++b) {
            bool found = false;
            // find a column where in the last row there is a one
            for (int i = b; i < config.n; ++i) {
                if (mzd_read_bit(work_matrix_H, b, i)) {
                    found = true;
                    if (i == b)
                        break;

                    std::swap(P_C->values[i], P_C->values[b]);
                    mzd_col_swap(work_matrix_H, b, i);
                    break;
                }
            }
            if (found) {
                for (int i = 0; i < b; ++i) {
                    if (i == b) {
                        continue;
                    }

                    if (mzd_read_bit(work_matrix_H, i, b)) {
                        // config.syndrom_col+1 descibes the end col.
                        mzd_row_xor_unroll<config.syndrom_col+1>(work_matrix_H, i, b);
                    }
                }
            } else {
                continue;
            }
        }

        const unsigned int weight = hamming_weight_column(work_matrix_H, config.syndrom_col);
		if (weight <= config.w) {
            for(rci_t j = 0; j < config.k; ++j){
                mzd_write_bit(e, 0, P_C->values[j], mzd_read_bit(work_matrix_H, j, config.syndrom_col));
            }

			ret = 1;
		}
		return ret;
	}
};

/// IMPORTANT: Does not alloc nor free any memory
/// \tparam config
/// \param e 					return parameter: Is set to the error if a solution is found.
/// \param working_s			tmp parameter will be overwritten
/// \param working_s_T			tmp parameter will be overwritten
/// \param work_matrix_H		syndrom is the las column
/// \param work_matrix_H_T
/// \param P_C
/// \return
template<const ConfigPrange &config>
uint64_t prange_thread(mzd_t *e,
                  mzd_t *work_matrix_H, mzd_t *work_matrix_H_T,  mzp_t *P_C,
                  customMatrixData *matrix_data, PerformancePrange *perf) {
	// Extract parameters.
	constexpr uint32_t n = config.n;    // #cols
	constexpr uint32_t k = config.k;    // #nrows IMPORTANT #rows IS NOT n-k
	constexpr uint32_t w = config.w;    // #weight
	constexpr uint64_t max_iters = config.max_iters;
	const unsigned int m4ri_k = config.m4ri_k;
	//constexpr uint64_t expected_loops = 0;

	// Make sure we do not make any crazy things
	ASSERT(n > k
				 && work_matrix_H->nrows == k && work_matrix_H->ncols >= n
				 && P_C->length == n);

	uint64_t loops = 0;
	while (max_iters - loops > 0) {
		loops += 1;

		matrix_create_random_permutation(work_matrix_H, work_matrix_H_T, P_C);
		int ra = matrix_echelonize_partial(work_matrix_H, m4ri_k, k, matrix_data, 0);

		// update the Gauß algorithm, if it was not possible to fully solve it.
		for (int b = ra; b < k; ++b) {
			bool found = false;
			// find a column where in the last row there is a one
			for (int i = b; i < n; ++i) {
				if (mzd_read_bit(work_matrix_H, b, i)) {
					found = true;
					if (i == b)
						break;

					std::swap(P_C->values[i], P_C->values[b]);
					mzd_col_swap(work_matrix_H, b, i);
					break;
				}
			}
			if (found) {
				for (int i = 0; i < b; ++i) {
					if (i == b) {
						continue;
					}

					if (mzd_read_bit(work_matrix_H, i, b)) {
						// config.syndrom_col+1 descibes the end col.
						mzd_row_xor_unroll<config.syndrom_col+1>(work_matrix_H, i, b);
					}
				}
			} else {
				continue;
			}
		}

		const unsigned int weight = hamming_weight_column(work_matrix_H, config.syndrom_col);
		if (weight <= w) {
			for(rci_t j = 0; j < k; ++j){
				mzd_write_bit(e, 0, P_C->values[j], mzd_read_bit(work_matrix_H, j, config.syndrom_col));
			}

			PERFORMANE_WRITE(perf->loops = loops);
			return 1;
		}

		if ((loops % 100000) == 0) {
			std::cout << "Loops: " << loops << "\n";
		}
	}

	PERFORMANE_WRITE(perf->loops += loops);
	return 0;
}

///
/// \tparam config
/// \param e
/// \param s
/// \param A
/// \param perf
/// \return
template<const ConfigPrange &config>
uint64_t prange(mzd_t *e, const mzd_t *const s, const mzd_t *const A, PerformancePrange *perf=nullptr) {
	// Extract parameters.
	constexpr uint32_t n = config.n;    // #cols
	constexpr uint32_t k = config.k;    // #nrows IMPORTANT #rows IS NOT n-k

	mzd_t *s_T ;
	mzd_t *work_matrix_H;
	mzd_t *work_matrix_H_T;
	customMatrixData *matrix_data;
	mzp_t *P_C;

	#pragma omp critical
	{
		s_T = mzd_transpose(NULL, s);
		work_matrix_H = matrix_concat(nullptr, A, s_T); // creates a new matrix with avx padding.
		work_matrix_H_T = mzd_init(n + 1, k);
		matrix_data = init_matrix_data(n);
		P_C = mzp_init(n);
	}
	uint64_t ret = prange_thread<config>(e, work_matrix_H, work_matrix_H_T, P_C, matrix_data, perf);

#pragma omp critical
	{
		mzd_free(work_matrix_H);
		mzd_free(work_matrix_H_T);
		mzd_free(s_T);
		mzp_free(P_C);
		free_matrix_data(matrix_data);
	}
    return ret;
}


#endif //SMALLSECRETLWE_DUMER_H
