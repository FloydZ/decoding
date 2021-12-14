#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"

#include "challenges/100.h"

// IMPORTANT: Define 'SSLWE_CONFIG_SET' before one include 'helper.h'.
#define SSLWE_CONFIG_SET

#define G_n                     n
#define G_k                     k
#define G_w                     w
#define G_l                     8u
#define G_d                     1u
#define LOG_Q                   1u
#define G_q                     1u
#define G_p                     4u
#define BaseList_p              1u
#define SORT_INCREASING_ORDER
#define VALUE_BINARY
//#define SORT_PARALLEL

static std::vector<uint64_t>                        __level_translation_array{{G_n - G_k - G_l, G_n - G_k}};
constexpr std::array<std::array<uint8_t, 3>, 3>     __level_filter_array{{ {{0,0,0}}, {{0,0,0}}, {{0,0,0}} }};

#include "helper.h"
#include "matrix.h"
#include "decoding.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(DecodingMitM100, Tree) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);

	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	//mzd_t *A = mzd_init(400, 1600);
	auto aswd = 0;
	for (int i = 0; i < 1; ++i) {
		std::cout << i << "\n";
		aswd += LeeBrickell<G_n, G_k, G_l, G_w, G_p, G_d>(ee, ss, A);
		ASSERT_LE(hamming_weight(ee), G_w);
	}

	std::cout << aswd  << "\n";
	print_matrix("FINAL e: ", ee);
	print_matrix("FINAL s: ", ss);


	// check output:
	mzd_transpose(ee_T, ee);
	mzd_mul_naive(ss_tmp, A, ee_T);
	mzd_transpose(ss_tmp_T, ss_tmp);
	print_matrix("calc  s:", ss_tmp_T);

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

	uint64_t r = 0;
    fastrandombytes(&r, 8);
	srandom(r);

	return RUN_ALL_TESTS();
}
