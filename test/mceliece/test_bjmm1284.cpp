#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "challenges/mce1284.h"

constexpr uint32_t l = 19;
constexpr uint32_t p = 1;
constexpr uint32_t l1 = 2;

#define NUMBER_THREADS 1u
#define CUSTOM_ALIGNMENT 4096
//#define BINARY_CONTAINER_ALIGNMENT
//#define USE_PREFETCH
//#define USE_AVX2_SPECIAL_ALIGNMENT

//#define USE_LOOPS 10

#include "matrix.h"
#include "bjmm.h"
//#include "prange.h"
#include "mo.h"
#include "../test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


// Small list instance
TEST(BJMM, t1284_small_normal) {
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
	                                   NUMBER_THREADS);
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

TEST(BJMM, t1284_smallc) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
	                                   150, 4,
	                                   l1, l- l1,
	                                   w,
	                                   NUMBER_THREADS, false, false, false, 100);
	BJMM<config> bjmm(ee, ss, A);
	bjmm.BJMMF_Outer();
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

TEST(BJMM, t1284_small_notFullHasmap) {
	mzd_t *AT       = mzd_from_str(n, n-k, h);
	mzd_t *A        = mzd_transpose(NULL, AT);
	mzd_t *ss       = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp   = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, 2, 24, 6,
	                                   1000, 1000,
	                                   6, 16,
	                                   w, NUMBER_THREADS);
	BJMM<config> bjmm(ee, ss, A);
	bjmm.BJMMF();
	EXPECT_GE(w, hamming_weight(ee));


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

TEST(BJMM, t1284_notFullHasmap) {
    // 26.542806709983612 {'time': 56.942454316331165,'memory': 28.20614612570475,'p1': 2,'l': 35,'l1': 6}
	mzd_t *AT       = mzd_from_str(n, n-k, h);
	mzd_t *A        = mzd_transpose(NULL, AT);
	mzd_t *ss       = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp   = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, 2, 35, 6,
	                                   2560, 19008,
	                                   6, 14,
	                                   w,
	                                   1);
	BJMM<config> bjmm(ee, ss, A);
	bjmm.BJMMF();
	EXPECT_GE(w, hamming_weight(ee));


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

// Small list instance
TEST(BJMM, t1284_small_run_onserver) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint32_t l = 17;
	constexpr uint32_t l1 = 2;

	static constexpr ConfigBJMM config(n, k, w, 1, l, l1,
	                                   146, 4,
	                                   l1, l- l1,
	                                   w,
	                                   NUMBER_THREADS);
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


TEST(MO, t1284) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint32_t l2  = 12;
	constexpr uint32_t l1  = 15;
	constexpr uint32_t IMv = 3;
	// constexpr uint32_t l   = l1+IMv*l2;

	static constexpr ConfigBJMM config(n, k, w, 2, l1, l1, 140, 30, l1, 1, w, NUMBER_THREADS, false, false, false, 0, false, l2, IMv);
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

//TEST(Prange, t1284_small) {
//	mzd_t *AT = mzd_from_str(n, n-k, h);
//	mzd_t *A = mzd_transpose(NULL, AT);
//	mzd_t *ss = mzd_from_str(1, n-k, s);
//	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
//	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);
//
//	mzd_t *ee = mzd_init(1, n);
//	mzd_t *ee_T = mzd_init(n, 1);
//
//
//	static constexpr ConfigPrange config(n, n-k, w, n, -1);
//#pragma omp parallel default(none) shared(cout, ee, ss, A) num_threads(NUMBER_THREADS) if(NUMBER_THREADS != 1)
//	{
//		prange<config>(ee, ss, A);
//	}
//
//	mzd_free(A);
//	mzd_free(AT);
//	mzd_free(ss);
//	mzd_free(ss_tmp);
//	mzd_free(ss_tmp_T);
//	mzd_free(ee);
//	mzd_free(ee_T);
//}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	srand(time(NULL));
	random_seed(rand()*time(NULL));
	return RUN_ALL_TESTS();
}