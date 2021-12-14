#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "challenges/lw0.h"

#define SSLWE_CONFIG_SET
#define G_n                     n
#define G_k                     k
#define G_w                     240
#define G_l                     12u
#define G_d                     0u                  // Depth of the search Tree
#define LOG_Q                   1u                  // unused
#define G_q                     1u                  // unused
#define BaseList_p              1u

#define G_l1                    2u
#define G_t                     1u
#define G_epsilon               0u
#define NUMBER_THREADS          1u


#define SORT_INCREASING_ORDER
#define VALUE_BINARY
#define SORT_PARALLEL

// DO NOT DISABLE
#define CUSTOM_ALIGNMENT 4096
#define BINARY_CONTAINER_ALIGNMENT
//#define USE_PREFETCH
//#define USE_AVX2_SPECIAL_ALIGNMENT

static  std::vector<uint64_t>                     __level_translation_array{{G_n-G_k-G_l, G_n-G_k}};
constexpr std::array<std::array<uint8_t, 1>, 1>   __level_filter_array{{ {{0}} }};

#include "matrix.h"
#include "bjmm.h"
#include "../test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

/*
36.149706182387604
{'time': 56.352157671581054,
 'memory': 14.765959420300877,
 'base_p': 1,
 'l1': 2,
 'l': 12
}
24.47221983297076
{'time': 53.28725530396183,
 'memory': 25.529501210987643,
 'base_p': 2,
 'l1': 6,
 'l': 29
}
12.18774001956636
{'time': 51.62448016557784,
 'memory': 36.286621172534495,
 'base_p': 3,
 'l1': 9,
 'l': 43
}
 */
TEST(BJMM, lw0) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_init(1, n-k);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	mzd_clear(ss);

	static constexpr ConfigBJMM config(G_n, G_k, G_w, BaseList_p, G_l, G_l1,
									   100, 45,
									   G_l1, (G_l-G_l1),
									   G_w-4*BaseList_p -1,
									   NUMBER_THREADS, false, false,
									   false, 0, true);
	BJMM<config> bjmm(ee, ss, A);
	bjmm.BJMMF();

	std::cout << "Weight: " << hamming_weight(ee) << "\n";
	EXPECT_LE(hamming_weight(ee), G_w);

	print_matrix("FINAL e: ", ee);
	print_matrix("FINAL s: ", ss);

	// check output:
	mzd_transpose(ee_T, ee);
	mzd_mul_naive(ss_tmp, A, ee_T);
	mzd_transpose(ss_tmp_T, ss_tmp);
	print_matrix("FINAL IS s:", ss_tmp_T);

	for (int i = 0; i < ss->ncols; ++i) {
		EXPECT_EQ(mzd_read_bit(ss, 0, i), 0);
	}

	mzd_free(A);
	mzd_free(AT);
	mzd_free(ss);
	mzd_free(ss_tmp);
	mzd_free(ss_tmp_T);
	mzd_free(ee);
	mzd_free(ee_T);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(rand()*time(NULL));
	return RUN_ALL_TESTS();
}