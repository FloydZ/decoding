#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"


// #include "challenges/challenge.h"
#include "challenges/100.h"
//#include "challenges/challenge200.h"
//#include "challenges/challenge300.h"

//constexpr unsigned int l = 5;
// constexpr unsigned int p = 3;

// 100 parameter set
static const unsigned int l = 5;
static const unsigned int p = 3;

// 200 parameter set
//static const unsigned int l = 5;
//tatic const unsigned int p = 3;

// 300
//static const unsigned int l = 30;
//static const unsigned int p = 10;


// IMPORTANT: Define 'SSLWE_CONFIG_SET' before one include 'helper.h'.
#define SSLWE_CONFIG_SET
#define G_k                     1
#define G_l                     l                   // unused?
#define G_d                     3                   // Depth of the search Tree
#define G_n                     (k+l)
#define LOG_Q                   1u
#define G_q                     (1u << LOG_Q)
#define G_w                     w
#define BaseList_p              2

#define SORT_INCREASING_ORDER
#define VALUE_BINARY

// only for 100 set.
static  std::vector<uint64_t>           __level_translation_array{{0, 20, 40, G_n}};
constexpr std::array<std::array<uint8_t, 3>, 3>   __level_filter_array{{ {{4,0,0}}, {{1,0,0}}, {{0,0,0}} }};

#include "helper.h"
#include "matrix.h"
#include "combinations.h"
#include "decoding.h"
#include "../test.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

//static const unsigned int nc = 510;
//static const unsigned int nr = 255;
static const unsigned int nc = 40;
static const unsigned int nr = 15;
static const unsigned int weight = 15;
static const unsigned int d = 8;

TEST(decoding, SWAP_BITS) {
	const uint64_t columns = 64*2;
	mzd_t *v = mzd_init(1, columns);

	// first do the null check,
	for (int i = 0; i < columns; ++i) {
		for (int j = 0; j < columns; ++j) {
			SWAP_BITS(v->rows[0], i, j);
			EXPECT_EQ(true, mzd_is_zero(v));
		}
	}

	// switch back and forth
	v->rows[0][0] = 1;
	for (int i = 1; i < columns; ++i) {
		SWAP_BITS(v->rows[0], i, i-1);
		EXPECT_EQ(true, mzd_read_bit(v, 0, i));
		EXPECT_EQ(false, mzd_read_bit(v, 0, i-1));
	}
}


TEST(asd, matrix_generate_random_weighted) {
	mzd_t *A = mzd_init(nr, nc);
	matrix_generate_random_weighted(A, weight);

	print_matrix("matrix_generate_random_weighted A:", A);

	mzd_free(A);
}

TEST(random, m4ri_random_full_rank) {
	mzd_t *A = mzd_init(nr, nc);
	m4ri_random_full_rank(A);

	print_matrix("m4ri_random_full_rank A:", A);

	mzd_free(A);
}

TEST(asd, matrix_create_random_permutation) {
	mzd_t *A = mzd_init(nr, nc);
	mzp_t *P_C = mzp_init(A->ncols);

	matrix_create_random_permutation(A,P_C);

	mzp_free(P_C);
	mzd_free(A);
}

TEST(TestM4RI, invert_perumtation) {
	mzd_t *A = mzd_init(1, nc);
	mzd_t *A_tmp = mzd_init(1, nc);
	mzd_t *A_copy = mzd_init(1, nc);
	mzd_randomize(A);
	mzd_copy(A_copy, A);

	mzp_t *P_C = mzp_init(A->ncols);
	matrix_create_random_permutation(A, P_C);
	EXPECT_NE(0, mzd_cmp(A, A_copy));

	// back transformation.
	for(rci_t j=0; j < nc; ++j)
		mzd_write_bit(A_tmp, 0, P_C->values[j], mzd_read_bit(A, 0, j));

	EXPECT_EQ(0, mzd_cmp(A_tmp, A_copy));

	mzp_free(P_C);
	mzd_free(A);
	mzd_free(A_copy);
	mzd_free(A_tmp);
}


TEST(AlgorithmOnlyM4ri, one ) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);

	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	print_matrix("ss: ", ss);


	auto aswd = 0;
	for (int i = 0; i < 1; ++i) {
	//	std::cout << i << "\n";
		aswd += LeeBrickell<n, k, l, w, p, d>(ee, ss, A);
		EXPECT_EQ(w, hamming_weight(ee));
	}

	std::cout << aswd  << "\n";

	// print_matrix("FINAL A: ", A);
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

	uint64_t r = 0;
    fastrandombytes(&r, 8);
	srandom(r);

	return RUN_ALL_TESTS();
}
