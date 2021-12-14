#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "challenges/qc42.h"


constexpr uint32_t l  = 34;
constexpr uint32_t l1 = 1;
constexpr uint32_t p  = 14;

#define NUMBER_THREADS          1u
#define CUSTOM_ALIGNMENT 4096
#define BINARY_CONTAINER_ALIGNMENT
//#define USE_PREFETCH

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


TEST(BJMM, qc42) {
	/*
	 {'base_p': 2,
	 'basesize': 16.6753,
	 'intersize': 19.3505,
	 'l': 34,
	 'l1': 14.0,
	 'loops': 23.4634,
	 'nb1': 14.0,
	 'nb2': 20.0,
	 'outsize': 18.701,
	 'p': 6,
	 'runtime': 44.8139,
	 'space': 19.3505}
	[16.675254143155012, 19.350508286310024, 18.70101657262005]
	21.79007095023474
	number of elements in hm1 104652.99999999988 size bucket= 6.387512207031243
	number of elements in hm2 668472.31500244 size bucket= 0.6375048780464554
	 */
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
									   8, 2,
									   l1, (l-l1),
									   w,
									   NUMBER_THREADS);
	BJMM<config> bjmm(ee, ss, A);

	for (int i = 0; i < 1; ++i) {
		bjmm.BJMMF();
		EXPECT_GE(w, hamming_weight(ee));
	}

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
	random_seed(rand()*time(NULL));
	return RUN_ALL_TESTS();
}