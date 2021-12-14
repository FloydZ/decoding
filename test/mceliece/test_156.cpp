#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "challenges/mce156.h"
#define SSLWE_CONFIG_SET
#define G_n                     n
#define G_k                     k
#define G_w                     w
#define G_l                     12
#define G_d                     1u                  // Depth of the search Tree
#define LOG_Q                   1u                  // unused
#define G_q                     1u                  // unused
#define G_p                     2u
#define BaseList_p              2u

#define G_l1                    (12)
#define G_l2                    (G_l-G_l1)
#define G_w1                    1u
#define G_w2                    1u

#define G_Thresh                10u
#define G_t                     4u
#define G_epsilon               0u
#define NUMBER_THREADS          1u

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
#include "dumer.h"
#include "prange.h"

#include "../test.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(MCElieceV2, hamming_weight_sum_row) {
	constexpr uint32_t n = 10;
	mzd_t *A = mzd_init(n, n);
	m4ri_random_full_rank(A);

	std::cout << hamming_weight_sum_row<0, 10>(A, 1, 2) << "\n";
	print_matrix("A", A);
	mzd_free(A);
}

TEST(MCElieceV2, SparsityNN) {
	constexpr uint32_t n = 232;
	constexpr uint32_t t = 10;

	customMatrixData *matrix_data = init_matrix_data(n+t);
	mzd_t *A = mzd_init(n, n+t);
	mzd_t *B = mzd_init(n, n+t);
	m4ri_random_full_rank(A);
	mzd_copy(B, A);

	//print_matrix("A", A);
	static constexpr ConfigSparsityNN config(n, n, t, 0, 0, 0);
	sparsityNN<config>(A, matrix_data);
	print_matrix("A", A, -1, -1, 0, n+t);

	//sparsityNN2<config>(B, matrix_data);
	//print_matrix("B", B, -1, -1, 0, n+t);

	mzd_free(A);
	free_matrix_data(matrix_data);
}


McEliece_v2_TEST(t156)
Prange_Perf_TEST(t156)
Prange_TEST(t156)

/// Special (not Mceliece form )
TEST(MCEliece_Prange, Specialformt156) {
	constexpr uint32_t n = 31;
	constexpr uint32_t t = 5;

	mzd_t *A  = mzd_init(n, n+t);
	mzd_randomize(A);
	mzd_t *ss = mzd_init(1, n);
	mzd_t *ss_T = mzd_init(n, 1);

	mzd_t *ee = mzd_init(1, n+t);
	ee->rows[0][0] = 15;  // set 4 bits
	mzd_t *ee2 = mzd_init(1, n+t);

	mzd_t *ee_T = mzd_init(n+t, 1);
	mzd_transpose(ee_T, ee);
	mzd_t *ee2_T = mzd_init(n+t, 1);

	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_mul_naive(ss_T, A, ee_T);

	mzd_transpose(ss, ss_T);

	static constexpr ConfigPrange config(n+t, n, 4, n+t, -1);

	auto aswd = 0;
	for (int i = 0; i < 1; ++i) {
		std::cout << i << "\n";
		aswd += prange<config>(ee2, ss, A);
		EXPECT_EQ(w, hamming_weight(ee));
	}

	print_matrix("FINAL e: ", ee2);
	print_matrix("FINAL s: ", ss);

	// check output:
	mzd_transpose(ee2_T, ee2);
	mzd_mul_naive(ss_tmp, A, ee2_T);
	mzd_transpose(ss_tmp_T, ss_tmp);
	print_matrix("FINAL ss:", ss_tmp_T);

	for (int i = 0; i < ss->ncols; ++i) {
		EXPECT_EQ(mzd_read_bit(ss, 0, i), mzd_read_bit(ss_tmp, i, 0));
	}

	mzd_free(A);
	mzd_free(ss);
	mzd_free(ss_tmp);
	mzd_free(ss_tmp_T);
	mzd_free(ee);
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(rand()* time(NULL));

	return RUN_ALL_TESTS();
}
