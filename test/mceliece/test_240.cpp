#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "challenges/mce240.h"
#define SSLWE_CONFIG_SET
#define G_n                     n
#define G_k                     k
#define G_w                     w
#define G_l                     12
#define G_d                     1u                  // Depth of the search Tree
#define LOG_Q                   1u                  // unused
#define G_q                     1u                  // unused
#define G_p                     2u
#define BaseList_p  1            2u

#define G_l1                    (12)
#define G_l2                    (G_l-G_l1)
#define G_w1                    1u
#define G_w2                    1u

#define G_t                     10u
#define G_Thresh                19u

#define G_epsilon                0u
#define NUMBER_THREADS           1u

// #define PERFORMANCE_LOGGING

#define SORT_INCREASING_ORDER
#define VALUE_BINARY
#define PERFORMANCE_LOGGING

// only for 100 set.
static  std::vector<uint64_t>                     __level_translation_array{{G_n-G_k-G_l, G_n-G_k}};
constexpr std::array<std::array<uint8_t, 1>, 1>   __level_filter_array{{ {{0}} }};

#include "helper.h"
#include "matrix.h"
#include "decoding.h"
#include "emz.h"
#include "sparsity.h"
#include "prange.h"
#include "../test.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

//Decoding_TEST(t240)
//CUSTOM_TEST(Prange, t240_custom, FOO(static constexpr ConfigPrange config(G_n, G_n-G_k, G_w, G_n, 1000);), prange<config>(ee, ss, A))
Prange_Perf_TEST(t240)

TEST(MCEliece_Prange, t240_2) {
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

bool comp(const mzd_t *A, const mzd_t *B){
	for (int i = 0; i < A->ncols; ++i) {
		if (mzd_read_bit(A, 0, i) != mzd_read_bit(B, 0, i) )
			return false;
	}

	return true;
}

TEST(MCEliece_Prange_Error_finder, t240_2) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);

	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigPrange config(G_n, G_n-G_k, G_w, G_n, -1);

	bool c = true;
	while(c) {
		prange<config>(ee, ss, A);
		EXPECT_EQ(w, hamming_weight(ee));
		// check output:
		mzd_transpose(ee_T, ee);
		mzd_mul_naive(ss_tmp, A, ee_T);
		mzd_transpose(ss_tmp_T, ss_tmp);
		c = comp(ss, ss_tmp_T);
	}



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

Prange_TEST(t240)

CUSTOM_TEST(MCElieceV22, t240_custom_thresh19_1, FOO(static constexpr ConfigSparsity config(G_n, G_k, G_w, G_p, 5, 20);), Sparsity<config>(ee, ss, A))
CUSTOM_TEST(MCElieceV22, t240_custom_thresh19_2, FOO(static constexpr ConfigSparsity config(G_n, G_k, G_w, G_p, 10, 20);), Sparsity<config>(ee, ss, A))

CUSTOM_TEST(MCElieceV22, t240_custom_thresh19, FOO(static constexpr ConfigSparsity config(G_n, G_k, G_w, G_p, G_t, 19);), Sparsity<config>(ee, ss, A))
CUSTOM_TEST(MCElieceV22, t240_custom_thresh18, FOO(static constexpr ConfigSparsity config(G_n, G_k, G_w, G_p, G_t + 1, 18);), Sparsity<config>(ee, ss, A))
CUSTOM_TEST(MCElieceV22, t240_custom_thresh17, FOO(static constexpr ConfigSparsity config(G_n, G_k, G_w, G_p, G_t, 17);), Sparsity<config>(ee, ss, A))
CUSTOM_TEST(MCElieceV22, t240_custom_thresh16, FOO(static constexpr ConfigSparsity config(G_n, G_k, G_w, G_p, G_t, 16);), Sparsity<config>(ee, ss, A))
CUSTOM_TEST(MCElieceV22, t240_custom_thresh15, FOO(static constexpr ConfigSparsity config(G_n, G_k, G_w, G_p, G_t, 15);), Sparsity<config>(ee, ss, A))
CUSTOM_TEST(MCElieceV22, t240_custom_thresh14, FOO(static constexpr ConfigSparsity config(G_n, G_k, G_w, G_p, G_t, 14);), Sparsity<config>(ee, ss, A))
CUSTOM_TEST(MCElieceV22, t240_custom_thresh13, FOO(static constexpr ConfigSparsity config(G_n, G_k, G_w, G_p, G_t, 13);), Sparsity<config>(ee, ss, A))

Sparsity_Test(t240);
CUSTOM_COMPARE(t240)

TEST(COMPARE2, t240) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(nullptr, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);

	static constexpr ConfigSparsity mcconfig(G_n, G_k, G_w, G_p, 5, 21);
	static constexpr ConfigPrange config(G_n, G_n-G_k, G_w, G_n,-1);

	PerformancePrange perfPrange;
	PerformanceMcElieceV2 perfSparsity;

	double pr_time = 0.0;
	for (int j = 0; j < TESTSIZE; ++j) {
		double t0 = ((double)clock()/CLOCKS_PER_SEC);
		prange<config>(ee, ss, A, &perfPrange);
		pr_time += ((double)clock()/CLOCKS_PER_SEC) - t0;

		EXPECT_EQ(w, hamming_weight(ee));
		// check output:
		mzd_transpose(ee_T, ee);
		mzd_mul_naive(ss_tmp, A, ee_T);

		for (int i = 0; i < ss->ncols; ++i) {
			EXPECT_EQ(mzd_read_bit(ss, 0, i), mzd_read_bit(ss_tmp, i, 0));
		}

		perfPrange.print();
		perfPrange.reset();
	}

	double mc_time = 0.0;
	double mc2_time = 0.0;
	for (int j = 0; j < TESTSIZE; ++j) {
		double t0 = ((double)clock()/CLOCKS_PER_SEC);
        Sparsity<mcconfig>(ee, ss, A, &perfSparsity);
		mc2_time += ((double)clock()/CLOCKS_PER_SEC) - t0;

		EXPECT_EQ(w, hamming_weight(ee));
		// check output:
		mzd_transpose(ee_T, ee);
		mzd_mul_naive(ss_tmp, A, ee_T);
		for (int i = 0; i < ss->ncols; ++i) {
			EXPECT_EQ(mzd_read_bit(ss, 0, i), mzd_read_bit(ss_tmp, i, 0));
		}

		perfSparsity.print();
		perfSparsity.reset();

	}

	std::cout << pr_time/TESTSIZE << " " <<
	          mc2_time/TESTSIZE << " " << "\n";

	mzd_free(A);
	mzd_free(AT);
	mzd_free(ss);
	mzd_free(ee);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));

	return RUN_ALL_TESTS();
}
