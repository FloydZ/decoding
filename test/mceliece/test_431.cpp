#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "challenges/mce431.h"

#define SSLWE_CONFIG_SET
#define G_n                     n
#define G_k                     k
#define G_w                     w
#define G_l                     18u
#define G_d                     0u                  // Depth of the search Tree
#define LOG_Q                   1u                  // unused
#define G_q                     1u                  // unused
#define G_p                     3u
#define BaseList_p              2u

#define G_t                     10u
#define G_Thresh                37u
#define G_epsilon               0u
#define NUMBER_THREADS          1u

#define SORT_INCREASING_ORDER
#define VALUE_BINARY
//#define SORT_PARALLEL
#define PERFORMANCE_LOGGING


static  std::vector<uint64_t>                     __level_translation_array{{G_n-G_k-G_l, G_n-G_k}};
constexpr std::array<std::array<uint8_t, 1>, 1>   __level_filter_array{{ {{0}} }};

#include "helper.h"
#include "matrix.h"
#include "decoding.h"
#include "emz.h"
#include "sparsity.h"
#include "../test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(Sparsity, t431) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A  = mzd_init(n-k, n);

	customMatrixData *matrix_data = init_matrix_data(n);
	constexpr uint32_t m4ri_k = matrix_opt_k(n-k, n);

	constexpr uint64_t tsize = 100000;

	double gaus_time = 0.0;
	for (int i = 0; i < tsize; ++i) {
		//mzd_transpose(A, AT);
		m4ri_random_full_rank(A);

		double t0 = ((double)clock()/CLOCKS_PER_SEC);
		matrix_echelonize_partial(A, m4ri_k, n - k, matrix_data, 0);
		gaus_time += ((double)clock()/CLOCKS_PER_SEC) - t0;
	}

	static constexpr ConfigSparsityNN config(n, n-k, 20, 0, 0, 0);
	double sparse_time = 0.0;
	for (int i = 0; i < tsize; ++i) {
		// mzd_transpose(A, AT);
		m4ri_random_full_rank(A);

		double t0 = ((double)clock()/CLOCKS_PER_SEC);
		sparsityNN<config>(A, matrix_data);
		sparse_time += ((double)clock()/CLOCKS_PER_SEC) - t0;
	}

	double gt = gaus_time/tsize;
	double st = sparse_time/tsize;
	std::cout << "gaus: " << gt << " sparsity:" << st << " factor: " << st/gt;

	mzd_free(A);
	mzd_free(AT);
}

TEST(MCEliece, t431) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	auto aswd = 0;
	for (int i = 0; i < 1; ++i) {
		std::cout << i << "\n";
		aswd += LeeBrickell<G_n, G_k, G_l, G_w, G_p, G_d>(ee, ss, A);
		EXPECT_EQ(w, hamming_weight(ee));
	}

	print_matrix("FINAL A: ", A);
	print_matrix("FINAL e: ", ee);
	print_matrix("FINAL s: ", ss);

	// check output:
	mzd_transpose(ee_T, ee);
	mzd_mul_naive(ss_tmp, A, ee_T);
	mzd_transpose(ss_tmp_T, ss_tmp);
	print_matrix("FINAL ss:", ss_tmp_T);

	for (int i = 0; i < ss->ncols; ++i) {
		EXPECT_EQ(mzd_read_bit(ss, 0, i), mzd_read_bit(ss_tmp, i, 0));
	}

	mzd_free(A);
	mzd_free(AT);
	mzd_free(ss);
	mzd_free(ss_tmp);
	mzd_free(ss_tmp_T);
	mzd_free(ee);
}


TEST(MCEliece_Prange, t431) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);

	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigPrange config(G_n, G_n-G_k, G_w, G_n, -1);

	auto aswd = 0;
	for (int i = 0; i < 1; ++i) {
		aswd += prange<config>(ee, ss, A);
		EXPECT_EQ(w, hamming_weight(ee));
	}

	print_matrix("FINAL e: ", ee);
	print_matrix("SHOUL s: ", ss);

	// check output:
	mzd_transpose(ee_T, ee);
	mzd_mul_naive(ss_tmp, A, ee_T);
	mzd_transpose(ss_tmp_T, ss_tmp);
	print_matrix("IS    s:", ss_tmp_T);

	for (int i = 0; i < ss->ncols; ++i) {
		EXPECT_EQ(mzd_read_bit(ss, 0, i), mzd_read_bit(ss_tmp, i, 0));
	}

	mzd_free(A);
	mzd_free(AT);
	mzd_free(ss);
	mzd_free(ss_tmp);
	mzd_free(ss_tmp_T);
	mzd_free(ee);
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();

	srand(time(NULL));
	random_seed(rand()*time(NULL));

	return RUN_ALL_TESTS();
}
