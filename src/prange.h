#ifndef SMALLSECRETLWE_PRANGE_H
#define SMALLSECRETLWE_PRANGE_H

#include "leebrickell.h"

// IMPORTANT: Due to implementation difficulties I have to rename the input parameters
//  n=#cols
//  k=#rows AND NOT n-k=#rows
struct ConfigPrange {
private:
	// Disable the empty constructor.
	constexpr ConfigPrange() : n(0), k(0), w(0), max_iters(0), m4ri_k(0), syndrom_col(0) {}

public:
	// instance parameter:
	const uint32_t  n, // code length
	                k, // code dimension
	                w; // weight
	// stop the algorithm after this amount of iterations, even the solution was not found.
	// Useful if you want to use Prange as a subroutine in some other algorithm
	const uint64_t max_iters;

	// do not change. optimal r parameter for the `method of the 4 russians` algorithm
	const int m4ri_k;

	// exact column within the parity check matrix where the syndrome is written int. Useful if you want to exploit alignment stuff.
	const uint32_t syndrom_col;

	// print every X iteration informations about the current state of the programm
	const uint64_t print_loops = 10000;

	// Append the He=w mod 2
	const bool TrivialAppendRow= true;

	constexpr ConfigPrange(const uint32_t n,
	                       const uint32_t k,
	                       const uint32_t w,
	                       const uint32_t syndrom_col,
	                       const uint64_t max_iters=0) :
			n(n),
	        k(k),
	        w(w),
	        max_iters(max_iters == 0 ? bc(n, w)/bc(k, w) : max_iters),
	        m4ri_k(matrix_opt_k(k, n)),
			syndrom_col(syndrom_col)
		{}
};

/// pass the struct to the algorithm to collect runtime performance information.
struct PerformancePrange {
public:
	uint64_t loops = 0;

	// prints some runtime/performance informations
	void print() noexcept {
		std::cout << "Prange Loops: " << loops << "\n" << std::flush;
	}

	// resets all runtime/performance counters
	constexpr void reset() noexcept {
		loops = 0;
	}

	// prints the expected number of loops
	void expected(const ConfigPrange &config) {
		uint64_t expected_loops = bc(config.n, config.w)/bc(config.k, config.w);
		std::cout << "Prange Loops/Expected: " << loops << "/" << expected_loops << "\n" << std::flush;
	}
};

/// Metaprogramming test:
template<const ConfigPrange &config>
struct prange_thread_single {
	template<uint32_t index>
	static int func(mzd_t *e, mzd_t *work_matrix_H, mzd_t *work_matrix_H_T, mzp_t *P_C,
				    customMatrixData *matrix_data) {
		int ret = 0;
		matrix_create_random_permutation(work_matrix_H, work_matrix_H_T, P_C);
		matrix_echelonize_partial_plusfix(work_matrix_H, config.m4ri_k, k, matrix_data, 0, k, 0, P_C);

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

template<const ConfigPrange &config>
class Prange {
public:
	constexpr static uint32_t n = config.n;// #cols
	constexpr static uint32_t k = config.k + config.TrivialAppendRow;
	constexpr static uint32_t w = config.w;// #weight
	constexpr static uint64_t max_iters = config.max_iters;
	constexpr static uint32_t m4ri_k = config.m4ri_k;

	//constexpr uint64_t expected_loops = 0;
	mzd_t *e;      // output: vector
	const mzd_t *s;// input: syndrome
	const mzd_t *A;// input: parity check matrix.
	mzd_t *AT;

	Prange(mzd_t *e, const mzd_t *const s, const mzd_t *const A,
			const uint32_t ext_tid = 0) noexcept
	: e(e), s(s), A(A){}

	/// IMPORTANT: Does not alloc nor free any memory
	/// \param e 					return parameter: Is set to the error if a solution is found.
	/// \param working_s			tmp parameter will be overwritten: Syndrome
	/// \param working_s_T			tmp parameter will be overwritten: Syndrome transposed
	/// \param work_matrix_H		syndrome is the last column
	/// \param work_matrix_H_T		tmp parameter, will be overwritten. Used to store the transposed of: `work_matrix_H`
	/// \param P_C					tmp parameter, will be overwritten. Used to store the new permutation.
	/// \return 0: nothing found
	/// 		1: solution found
	uint64_t prange_thread(mzd_t *e,
	                       mzd_t *work_matrix_H, mzd_t *work_matrix_H_T, mzp_t *P_C,
	                       customMatrixData *matrix_data, PerformancePrange *perf) noexcept {
		// Make sure we do not make any crazy things
		ASSERT(n > k);
		ASSERT(work_matrix_H->ncols >= n);
		ASSERT(P_C->length == n);

		uint64_t loops = 0;
		while (max_iters - loops > 0) {
			loops += 1;

			matrix_create_random_permutation(work_matrix_H, work_matrix_H_T, P_C);
			matrix_echelonize_partial_plusfix(work_matrix_H, m4ri_k, k, matrix_data, 0, k, 0, P_C);

			const unsigned int weight = hamming_weight_column(work_matrix_H, config.syndrom_col);
			if (unlikely(weight <= w)) {
				for (rci_t j = 0; j < k; ++j) {
					mzd_write_bit(e, 0, P_C->values[j], mzd_read_bit(work_matrix_H, j, config.syndrom_col));
				}
				
				// write the solution
				mzd_print(e);
				PERFORMANE_WRITE(perf->loops = loops);
				return 1;
			}

#if !defined(NO_LOGGING)
			// periodically prints some information
			if ((loops % config.print_loops) == 0) {
				std::cout << "Loops: " << loops << "\n";
			}
#endif
		}

		PERFORMANE_WRITE(perf->loops += loops);
		return 0;
	}

	/// The input matrix will be appended by a few rows.
	/// Note that the input matrix must have in its last colum the syndrom
	/// Append the following trivial rows
	///	[ 1^n 						| 0]	row1
	/// \param in input matrix.
	/// \return newly created matrix with a few additional rows.
	static mzd_t* append_trivial_rows(mzd_t *in) noexcept {
		const uint32_t cols = in->ncols;
		const uint32_t rows = in->nrows;

		// Note the use of TrivialAppendRows
		mzd_t *ret  = in;
		mzd_t *row1 = mzd_init(1, cols);

		// write the rows
		for (uint32_t i = 0; i < n; ++i) {
			mzd_write_bit(row1, 0, i, 1);
		}

		// write the syndrom
		mzd_write_bit(row1, 0, n, w%2);
		mzd_copy_row(ret, rows-config.TrivialAppendRow, row1, 0);
		mzd_free(row1);
		return ret;
	}

	uint64_t __attribute__((noinline))
	run() noexcept {
		return this->prange(e, s, A);
	}

	/// \param e	output: error vector = solution
	/// \param s	input the syndrome to match on
	/// \param A	input: parity check matrix
	/// \param perf	input: 	nullptr: nothing happens
	///						struct: on return, this struct contains runtime/performance information about the algorithm.
	/// \return #loops needed to find the solution
	uint64_t prange(mzd_t *e, const mzd_t *const s, const mzd_t *const A, PerformancePrange *perf = nullptr) noexcept {
		mzd_t *s_T;
		mzd_t *work_matrix_H;
		mzd_t *work_matrix_H_T;
		customMatrixData *matrix_data;
		mzp_t *P_C;

		#pragma omp critical
		{
			s_T = mzd_transpose(NULL, s);

			work_matrix_H = matrix_init(n+1, n-k);
			mzd_t *tmp = matrix_concat(nullptr, A, s_T); // creates a new matrix with avx padding.
			matrix_copy(work_matrix_H, tmp);
			mzd_free(tmp);

			work_matrix_H_T = mzd_init(work_matrix_H->ncols, work_matrix_H->nrows);
			matrix_data = init_matrix_data(n);
			P_C = mzp_init(n);

			if constexpr (config.TrivialAppendRow) {
				work_matrix_H = append_trivial_rows(work_matrix_H);
			}
		}
		uint64_t ret = this->prange_thread(e, work_matrix_H, work_matrix_H_T, P_C, matrix_data, perf);

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
};

#endif //SMALLSECRETLWE_DUMER_H
