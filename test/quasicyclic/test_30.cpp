#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "challenges/qc30.h"

constexpr uint32_t l  = 16;
constexpr uint32_t l1 = 1;
constexpr uint32_t p  = 2;

#define NUMBER_THREADS 1
#define CUSTOM_ALIGNMENT 4096
#define BINARY_CONTAINER_ALIGNMENT
//#define USE_PREFETCH
//#define USE_AVX2_SPECIAL_ALIGNMENT

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


TEST(BJMM, qc30) {
	/*
	{
		'base_p': 2,
		'basesize': 14.8078,
		'intersize': 16.6155,
		'l': 30,
		'l1': 13.0,
		'loops': 14.621,
		'nb1': 13.0,
		'nb2': 17.0,
		'outsize': 16.231,
		'p': 6,
		'runtime': 33.255,
		'space': 16.6155
	}

	[14.807757403589267, 16.615514807178535, 16.23102961435707]
	12.951965215727498
	number of elements in hm1 28680.000000000007 size bucket= 3.500976562500001
	number of elements in hm2 100408.00781250004 size bucket= 0.7660523056983951

	 13m53s = 53234iters
	    Erwartet: 197.14486614522892s
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
									   110, 10,
									   l1, (l-l1),
									   w,
									   NUMBER_THREADS, true);
	static constexpr ConfigBJMM config2(n, k, w, p, l, l1,
									   110, 10,
									   l1, (l-l1),
									   w,
									   NUMBER_THREADS);

	BJMM<config> bjmm(ee, ss, A);
	//BJMM<config2> bjmm2(ee2, ss, A);


	double t00 = ((double)clock()/CLOCKS_PER_SEC);
	uint64_t loops2 = 0;//bjmm2.BJMMF();
	double time2 = ((double)clock()/CLOCKS_PER_SEC) - t00;

	double t0 = ((double)clock()/CLOCKS_PER_SEC);
	uint64_t loops1 = bjmm.BJMMF();
	double time1 = ((double)clock()/CLOCKS_PER_SEC) - t0;

	EXPECT_GE(w, hamming_weight(ee));

	print_matrix("FINAL correct e: ", ee2);
	print_matrix("FINAL         e: ", ee);
	print_matrix("FINAL SHOULD s: ", ss);
	std::cout << "Loops: " << loops1 << ":" << loops2 << " " << double (loops2)/ double (loops1) << "\n";
	std::cout << "Time:  " << time1  << ":" << time2  << " " << time2 / time1 << "\n";

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

TEST(BJMM, q30IM) {
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
	constexpr uint32_t l2 = 16;
	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
	                                   1<<7, 3,
	                                   l1, l2,
	                                   w,
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

TEST(BJMM, q30IM_withc) {
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
	constexpr uint32_t l2 = 16;
	static constexpr ConfigBJMM config(n, k, w, 2, l, l1,
	                                   1<<9, 5,
	                                   l1, l2,
	                                   w,
	                                   NUMBER_THREADS, true, false,
	                                   true, 162, false, l2, 2);
	BJMMNN<config> bjmm(ee, ss, A);
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
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(rand()*time(NULL));
	return RUN_ALL_TESTS();
}