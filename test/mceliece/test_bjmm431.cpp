#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>
#include <functional>
#include "m4ri/m4ri.h"
#include "challenges/mce431.h"
#include "../test.h"

constexpr uint32_t l = 13;
constexpr uint32_t p = 1;
constexpr uint32_t l1 = 2;

#define NUMBER_THREADS          1u
#define CUSTOM_ALIGNMENT PAGE_SIZE
#define USE_AVX2_SPECIAL_ALIGNMENT
#define BINARY_CONTAINER_ALIGNMENT
//#define USE_PREFETCH

#include "matrix.h"
#include "bjmm.h"
#include "bjmm_nn.h"
#include "bjmm_hybridtree.h"
#include "mo.h"
#include "../test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(MO, t431) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint32_t l2  = 11;
	constexpr uint32_t l1  = 2;
	constexpr uint32_t IMv = 2;
	// constexpr uint32_t l   = l1+IMv*l2;


	static constexpr ConfigBJMM config(n, k, w, 1, l1, l1,
	                                   80, 20,
	                                   l1, 1,
	                                   w-4,
	                                   NUMBER_THREADS, false, false, false, 0, false, l2, IMv);
    MO<config> mo(ee, ss, A);

	for (int i = 0; i < 1; ++i) {
		mo.run();
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

TEST(BJMM, t431) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
	                                   1<<8, 1<<(3+NUMBER_THREADS),
	                                   l1, (l-l1),
	                                   w,
	                                   NUMBER_THREADS, false, false, false, 0);
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
}


TEST(BJMM, t431c) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
	                                   1<<8, 1<<4,
	                                   l1, (l-l1),
	                                   w,
	                                   NUMBER_THREADS, false, false, false, 5);
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

TEST(BJMM, t431p2) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, 2, l, 4,
	                                   1<<10, 3048,
	                                   2, (l-4)-3,
	                                   w-4,
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
}


// This test is designed to test the non fast code path in sort.h
TEST(BJMM, t431_nonfullbuckets) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
	                                   1<<10, 3000,
	                                   l1, (l-l1)-9,
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
}


TEST(BJMM, t431_nonfullbuckets_hm1hm2) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, p, l, 4,
	                                   60, 60,
	                                   4-2, (l-4)-4,
	                                   w,
	                                   1);
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
}

// This test is designed to test the non fast code path in sort.h
TEST(BJMM, t431_nonfullbuckets_interpolationsearch) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
									   1<<10, 1025,
									   l1, (l-l1)-9, w, NUMBER_THREADS,
									   false, false, false, 0, false, 0, 0, false,
									   false, false, false, true);
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
}

//// This test is designed to test the slow code path in sort.h
//TEST(BJMM, t431_nonfullbuckets_notuseloadaccess_and_interpolationsearch) {
//	mzd_t *AT = mzd_from_str(n, n-k, h);
//	mzd_t *A = mzd_transpose(NULL, AT);
//	mzd_t *ss = mzd_from_str(1, n-k, s);
//	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
//	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);
//
//	mzd_t *ee = mzd_init(1, n);
//	mzd_t *ee_T = mzd_init(n, 1);
//
//	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
//									   1<<10, 1025,
//									   l1, (l-l1)-9, w, NUMBER_THREADS,
//									   false, false, false, 0, false, 0, 0, false,
//									   false, false, false, true,
//									   false, false, true, false);
//	BJMM<config> bjmm(ee, ss, A);
//
//	for (int i = 0; i < 1; ++i) {
//		bjmm.BJMMF();
//		EXPECT_GE(w, hamming_weight(ee));
//	}
//
//	print_matrix("FINAL e: ", ee);
//	print_matrix("FINAL SHOULD s: ", ss);
//
//	// check output:
//	mzd_transpose(ee_T, ee);
//	mzd_mul_naive(ss_tmp, A, ee_T);
//	mzd_transpose(ss_tmp_T, ss_tmp);
//	print_matrix("FINAL IS s:", ss_tmp_T);
//
//	for (int i = 0; i < ss->ncols; ++i) {
//		EXPECT_EQ(mzd_read_bit(ss, 0, i), mzd_read_bit(ss_tmp, i, 0));
//	}
//
//
//	mzd_free(A);
//	mzd_free(AT);
//	mzd_free(ss);
//	mzd_free(ss_tmp);
//	mzd_free(ss_tmp_T);
//	mzd_free(ee);
//}

// This test is designed to test the non fast code path in sort.h
TEST(BJMM, t431_nonfullbuckets_notuseloadaccess) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
									   1<<10, 1025,
									   l1, (l-l1)-9, w, NUMBER_THREADS,
									   false, false, false, 0, false, 0, 0, false,
									   false, false, false, false,
									   false, false, true, false);
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
}


TEST(BJMM, t431IM) {
//	Indyk Motwano approach
// hashmap 3 NOT work
//      c UNTESTET
//      full length UNTESTET
//      c+full length UNTESTET
// hashmap 2 work
//      c does work
//      full length does work
//      c+full length does work
// hashmap 1 does NOT WORK
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint32_t l1 = 2;
	constexpr uint32_t l = l1;
	constexpr uint32_t l2 = 14;
	constexpr uint32_t IMviews = 2;
	constexpr uint32_t c = 0;

	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
	                                   1<<8, 2048,
	                                   l1, l2,
	                                   w-4,
	                                   NUMBER_THREADS, false, false,
	                                   false, c, false, l2, IMviews);
	BJMMNN<config> bjmm(ee, ss, A);
	if constexpr(c == 0)
		bjmm.BJMMNNF();
	else
		bjmm.BJMMNNF_Outer();
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


TEST(BJMMHybridTree, t431) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint32_t l = 17;
	constexpr uint32_t l1 = 5;
	constexpr uint32_t IMv = 2;

	static constexpr ConfigBJMM config(n, k, w, 1, l, l1,
	                                   500, 1,
	                                   5, 1,
	                                   w-4,
	                                   NUMBER_THREADS, false, false, false, 0, false, l, IMv);
	BJMMHybridTree<config> bjmm(ee, ss, A);

	for (int i = 0; i < 1; ++i) {
		bjmm.run();
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
