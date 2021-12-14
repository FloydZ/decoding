#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "challenges/mce1284b.h"

constexpr uint32_t l = 19;
constexpr uint32_t p = 1;
constexpr uint32_t l1 = 2;

#define G_epsilon               0u
#define CUSTOM_ALIGNMENT 4096
#define BINARY_CONTAINER_ALIGNMENT
//#define USE_PREFETCH
//#define USE_AVX2
//#define USE_AVX2_SPECIAL_ALIGNMENT

#include "matrix.h"
#include "bjmm.h"
#include "prange.h"
#include "../test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


// Small list instance
TEST(BJMM, t1284backdoor_small) {
//		perms: 38.85968551742974
//		{'time': 57.383243644998316,
//		 'memory': 16.088539819423847
//		 'p1': 1,
//		 'l': 19,
//		 'l1': 2}
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
	                                   160, 4,
	                                   l1, l- l1,
	                                   w,
	                                   1);
	BJMM<config> bjmm(ee, ss, A);

	for (int i = 0; i < 10; ++i) {
		bjmm.BJMMF();
		EXPECT_GE(w, hamming_weight(ee));
	}

//	// check output:
//	mzd_transpose(ee_T, ee);
//	mzd_mul_naive(ss_tmp, A, ee_T);
//	mzd_transpose(ss_tmp_T, ss_tmp);
//
//	for (int i = 0; i < ss->ncols; ++i) {
//		EXPECT_EQ(mzd_read_bit(ss, 0, i), mzd_read_bit(ss_tmp, i, 0));
//	}

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
	srand(time(NULL));
	random_seed(rand()*time(NULL));
	return RUN_ALL_TESTS();
}