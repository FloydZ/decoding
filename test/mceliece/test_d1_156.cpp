#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "challenges/mce156.h"

#define SSLWE_CONFIG_SET
#define G_n                     n
#define G_k                     k
#define G_w                     w
#define G_l                     10
#define G_d                     1u                  // Depth of the search Tree
#define LOG_Q                   1u                  // unused
#define G_q                     1u                  // unused
#define G_p                     1u
#define BaseList_p              1u

#define NUMBER_THREADS 1
#define SORT_INCREASING_ORDER
#define VALUE_BINARY

// only for 100 set.
static  std::vector<uint64_t>                     __level_translation_array{{G_n-G_k-G_l, G_n-G_k}};
constexpr std::array<std::array<uint8_t, 1>, 1>   __level_filter_array{{ {{0}} }};

#include "helper.h"
#include "matrix.h"
#include "combinations.h"
#include "decoding.h"
#include "emz.h"
#include "../test.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(MCEliece_d1, t156) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);

	print_matrix("A", A);

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
	mzd_free(ss);
	mzd_free(ss_tmp);
	mzd_free(ss_tmp_T);
	mzd_free(ee);

}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();

	random_seed(time(NULL));

	return RUN_ALL_TESTS();
}
