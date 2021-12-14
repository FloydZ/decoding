#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"


#include "challenges/510.h"

// 100 parameter set
static const unsigned int l = 5;
static const unsigned int p = 3;


// IMPORTANT: Define 'SSLWE_CONFIG_SET' before one include 'helper.h'.
#define SSLWE_CONFIG_SET
#define G_k                     1
#define G_l                     l                   // unused?
#define G_d                     3                   // Depth of the search Tree
#define G_n                     (k+l)
#define LOG_Q                   1u
#define G_q                     (1u << LOG_Q)
#define G_w                     w
#define BaseList_p              2

#define SORT_INCREASING_ORDER
#define VALUE_BINARY

// only for 100 set.
static  std::vector<uint64_t>           __level_translation_array{{0, 20, 40, G_n}};
constexpr std::array<std::array<uint8_t, 3>, 3>   __level_filter_array{{ {{4,0,0}}, {{1,0,0}}, {{0,0,0}} }};

#include "helper.h"
#include "matrix.h"
#include "combinations.h"
#include "decoding.h"
#include "../test.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


const char *ee_str = "000000000010000000000000000010000000010100000000000001000000010100000000010001000000000001010010010010110000001000001010000000000000011000000000000000001000000000010000001000000000000000100000011100001010000001010010010000000000001000010010000000010000000000000000001100101010000001000000000000000000000000000100100000000000000000001000000000000000010000100000000000000000001000001000000011100000000000001000010000110100011000001001000000000000000011000000000000000000010100010000100000100000000000000000000000";
TEST(AlgorithmOnlyM4ri, one ) {
	mzd_t *AT   = mzd_from_str(n, n-k, h);
	mzd_t *A    = mzd_transpose(NULL, AT);
	mzd_t *ss   = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp   = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);
	mzd_t *ee = mzd_from_str(1, n, ee_str);
	mzd_t *ee_T = mzd_init(n, 1);

	print_matrix("ss: ", ss);

	// print_matrix("FINAL A: ", A);
	print_matrix("FINAL e: ", ee);
	print_matrix("FINAL s: ", ss);

	EXPECT_EQ(w, hamming_weight(ee));
	// check output:
	mzd_transpose(ee_T, ee);
	mzd_mul_naive(ss_tmp, A, ee_T);
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
	mzd_free(ee_T);

}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);

	uint64_t r = 0;
    fastrandombytes(&r, 8);
	srandom(r);

	return RUN_ALL_TESTS();
}
