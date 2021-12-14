#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "challenges/qc10.h"

constexpr uint32_t l = 4;
constexpr uint32_t p = 1;
constexpr uint32_t l1 = 2;

#define NUMBER_THREADS          1u
#define CUSTOM_ALIGNMENT 4096
#define BINARY_CONTAINER_ALIGNMENT
//#define USE_PREFETCH

#include "matrix.h"
#include "bjmm.h"
#include "bjmm_nn.h"
#include "../test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(BJMM, qc10) {
	/*
	{
		'base_p': 2,
		'basesize': 9.2167,
		'intersize': 10.4335,
		'l': 20,
		'l1': 8.0,
		'loops': 3.5397,
		'nb1': 8.0,
		'nb2': 12.0,
		'outsize': 8.867,
		'p': 7,
		'runtime': 15.9732,
		'space': 10.4335
	 }

	number of elements in hm1 595.0000000000002 size bucket= 2.324218750000001
	number of elements in hm2 1382.910156250001 size bucket= 0.3376245498657229
	 */
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee2 = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
									   10, 100,
									   l1, (l-l1),
									   w-3,
									   NUMBER_THREADS, true);
	static constexpr ConfigBJMM config2(n, k, w, p, l, l1,
	                                    10, 100,
	                                    l1, (l-l1),
	                                    w-4,
	                                    NUMBER_THREADS);

	BJMM<config2> bjmm2(ee2, ss, A);
	bjmm2.BJMMF();


	BJMM<config> bjmm(ee, ss, A);
	for (int i = 0; i < 1; ++i) {
		bjmm.BJMMF();
		EXPECT_GE(w, hamming_weight(ee));
	}

	print_matrix("FINAL correct e: ", ee2);
	print_matrix("FINAL DOOM    e: ", ee);
	print_matrix("FINAL SHOULD s: ", ss);

	// check output:
	mzd_transpose(ee_T, ee);
	mzd_mul_naive(ss_tmp, A, ee_T);
	mzd_transpose(ss_tmp_T, ss_tmp);
	print_matrix("FINAL IS s:", ss_tmp_T);

	for (int i = 0; i < ss->ncols; ++i) {
		if (mzd_read_bit(ss, 0, i) != mzd_read_bit(ss_tmp, i, 0)) {
			EXPECT_EQ(false, true);
			break;
		}
	}


	mzd_free(A);
	mzd_free(AT);
	mzd_free(ss);
	mzd_free(ss_tmp);
	mzd_free(ss_tmp_T);
	mzd_free(ee);
	mzd_free(ee2);
	mzd_free(ee_T);
}

TEST(BJMM, qc10SpecialTree) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, 2, l, l1,
	                                   10, 100,
	                                   l1, (l-l1),
	                                   w-4,
	                                   NUMBER_THREADS, true, true, false, 0);
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

TEST(BJMM, qc10FullList) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, 1, l, l1,
	                                   10, 100,
	                                   l1, (l-l1),
	                                   w-3,
	                                   NUMBER_THREADS, true, false, true, 0);
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

TEST(BJMM, qc10c) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
	                                   10, 100,
	                                   l1, (l-l1),
	                                   w - 3,
	                                   NUMBER_THREADS, true, false, false,
	                                   10, false);
	BJMM<config> bjmm(ee, ss, A);

	for (int i = 0; i < 1; ++i) {
		bjmm.BJMMF_Outer();
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
}

TEST(BJMM, q10IM) {
//	Indyk Motwano approach
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint32_t l1 = 2;
	constexpr uint32_t l = l1;
	constexpr uint32_t l2 = 8;
	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
	                                   1<<5, 20,
	                                   l1, l2,
	                                   w-3,
	                                   NUMBER_THREADS, true, false,
	                                   false, 0, false, l2, 2);
	BJMMNN<config> bjmm(ee, ss, A);
	bjmm.BJMMNNF();
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
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	//ident();
	random_seed(rand()*time(NULL));
	return RUN_ALL_TESTS();
}