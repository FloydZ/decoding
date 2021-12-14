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
#define G_d                     2u                  // Depth of the search Tree
#define LOG_Q                   1u                  // unused
#define G_q                     1u                  // unused
#define G_p                     2u
#define BaseList_p              1u

#define G_l1                    (2)
#define G_l2                    (2)
#define G_w1                    1u
#define G_w2                    1u

#define G_Thresh                10u
#define G_t                     4u
#define G_epsilon               0u
#define NUMBER_THREADS          1u

#define SORT_INCREASING_ORDER
#define VALUE_BINARY
#define SORT_PARALLEL

// DO NOT DISABLE
#define CUSTOM_ALIGNMENT 4096
#define BINARY_CONTAINER_ALIGNMENT
//#define USE_PREFETCH


// only for 100 set.
static  std::vector<uint64_t>                     __level_translation_array{{G_n-G_k-G_l, G_n-G_k}};
constexpr std::array<std::array<uint8_t, 1>, 1>   __level_filter_array{{ {{0}} }};

#include "helper.h"
#include "matrix.h"
#include "approximate.h"
#include "../test.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(Approximate, d2) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint32_t ls = 3;
	constexpr std::array<uint32_t , 3> li{G_l1, G_l, 0};
	constexpr std::array<uint64_t , 3> nbi{G_l1, G_l, 0};
	constexpr std::array<uint64_t , 3> sbi{100, 100, 0};


	static constexpr ConfigApproximate config(G_n, G_k, G_w, G_d, BaseList_p, G_l, ls, li, nbi, sbi,
	                                   G_w-4, NUMBER_THREADS, BaseListConstruction::MITM);
	Approximate<config> appr(ee, ss, A);

	appr.run();
	EXPECT_GE(w, hamming_weight(ee));


	print_matrix("FINAL e: ", ee);
	print_matrix("FINAL SHOULD s: ", ss);

	// check output:
	mzd_transpose(ee_T, ee);
	mzd_mul_naive(ss_tmp, A, ee_T);
	mzd_transpose(ss_tmp_T, ss_tmp);
	print_matrix("FINAL IS s:", ss_tmp_T);

	for (int i = 0; i < ss->ncols; ++i) {
		EXPECT_EQ(mzd_read_bit(ss, 0, i), mzd_read_bit(ss_tmp, i, 0));
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
	random_seed(rand()* time(NULL));

	return RUN_ALL_TESTS();
}
