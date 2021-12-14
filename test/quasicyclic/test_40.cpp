#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "challenges/qc40.h"

constexpr uint32_t l  = 33;
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


TEST(BJMM, qc40) {
	/*
	{
		'base_p': 2,
		'basesize': 16.4043,
		'intersize': 18.8087,
		'l': 33,
		'l1': 14.0,
		'loops': 21.9022,
		'nb1': 14.0,
		'nb2': 19.0,
		'outsize': 18.6174,
		'p': 6,
		'runtime': 42.7109,
		'space': 18.8087
	 }

	[16.404343291585757, 18.808686583171514, 18.61737316634303]
	20.229358005061442
	number of elements in hm1 86736.00000000009 size bucket= 5.293945312500005
	number of elements in hm2 459175.64062500093 size bucket= 0.875808030366899
	 */
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	static constexpr ConfigBJMM config(n, k, w, p, l, l1,
									   7, 2,
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

TEST(BJMM, qc40_small) {
	/*
		'base_p': 1,
		'basesize': 8.6689,
		'intersize': 14.3378,
		'l': 14,
		'l1': 3.0,
		'loops': 26.2226,
		'nb1': 3.0,
		'nb2': 11.0,
		'outsize': 17.6755,
		'p': 4,
		'runtime': 46.5139,
		'space': 17.6755

	 [8.668884984266247, 14.337769968532495, 17.67553993706499]
	24.811061635704174
	number of elements in hm1 407.0000000000001 size bucket= 50.875000000000014
	number of elements in hm2 20706.12500000001 size bucket= 10.110412597656255
	 */
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	using DecodingValue     = Value_T<BinaryContainer<k + 14>>;
	using DecodingLabel     = Label_T<BinaryContainer<n - k>>;
	using DecodingMatrix    = mzd_t *;
	using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList2     = Parallel_List_T<DecodingElement>;
	using DecodingList      = List_T<DecodingElement>;
	using ChangeList        = std::vector<std::pair<uint32_t, uint32_t>>;

	static constexpr ConfigBJMM config(n, k, w, 1, 14, 2,
	                                   120, 20,
	                                   2, (14-2),
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