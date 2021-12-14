#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>
#include <chrono>

#include "m4ri/m4ri.h"
#include "challenges/mce1161.h"

#define SSLWE_CONFIG_SET
#define G_n                     n
#define G_k                     k
#define G_w                     w
#define G_l                     0u
#define G_d                     2u                  // Depth of the search Tree
#define LOG_Q                   1u                  // unused
#define G_q                     1u                  // unused
#define G_p                     1u
#define G_l1                    (15)
#define G_l2                    (G_l-G_l1)
#define G_w1                    0u
#define G_w2                    2u
#define G_epsilon               0u
#define G_t                     20u

#define NUMBER_THREADS          1
#define SORT_INCREASING_ORDER
#define VALUE_BINARY
//#define CHECK_PERM

#define NUMBER_THREADS 1

static  std::vector<uint64_t>           __level_translation_array{{G_n-G_k-G_l, G_n-G_k-G_l+G_l2, G_n-G_k}};
constexpr std::array<std::array<uint8_t, 2>, 2>   __level_filter_array{{ {{0,0}}, {{0,0}} }};

#include "helper.h"
#include "matrix.h"
#include "decoding.h"
#include "sparsity.h"
#include "../test.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(MCElieceV2, t1161) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(nullptr, AT);

	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	auto aswd = 0;
	for (int i = 0; i < 1; ++i) {
		std::cout << i << "\n";
		aswd += mc_eliece_d2_v2<G_n, G_k, G_w, G_p, G_t, G_l1, G_w1, G_l2, G_w2, G_epsilon,
				0, 13, 1, 1, 1>(ee, ss, A);
		EXPECT_EQ(w, hamming_weight(ee));
	}

	std::cout << aswd  << "\n";
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

// Quick benchmark comparing M4ri and sparsity algorithm.
TEST(MCElieceV2, QuickBencht1161) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_init(n-k, n);
	mzd_t *M = mzd_init(n-k, n);

	const int RUNS = 1;
	double time = 0.0;
	for (int i = 0; i < RUNS; ++i) {
		mzd_transpose(A, AT);
		double t0 = (double)clock()/CLOCKS_PER_SEC;
		mzd_echelonize(A, 1);
		time += ((double)clock()/CLOCKS_PER_SEC) - t0;
	}
	double echo = time/(double)RUNS;
	std::cout << "Echolonize: " << echo << "s\n";

/*
	time = 0.0;
	for (int i = 0; i < RUNS; ++i) {
		mzd_transpose(A, AT);
		double t0 = (double)clock()/CLOCKS_PER_SEC;
		sparsity<n-k-G_l, G_t, n-k-G_l>(M, A, -1);
		time += ((double)clock()/CLOCKS_PER_SEC) - t0;
	}
	double spars = time/(double)RUNS;
	std::cout << "Sparsity: " << spars << "s\n";
	std::cout << "coeff: " << spars/echo << "\n";
*/

	time = 0.0;
	for (int i = 0; i < RUNS; ++i) {
		mzd_transpose(A, AT);
		double t0 = (double)clock()/CLOCKS_PER_SEC;
		sparsityNN<n-k-G_l, G_t, n-k-G_l, 1>(M, A, -1);
		time += ((double)clock()/CLOCKS_PER_SEC) - t0;
	}
	double sparsNN = time/(double)RUNS;
	std::cout << "Sparsity: " << sparsNN << "s\n";
	std::cout << "coeffNN: " << sparsNN/echo << "\n";

	mzd_free(A);
	mzd_free(AT);
	mzd_free(M);
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();

    random_seed(time(NULL));

	return RUN_ALL_TESTS();
}
