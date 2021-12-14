#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>
#include <chrono>

#include "m4ri/m4ri.h"
#include "challenges/mce1161.h"

#define SSLWE_CONFIG_SET
#define G_n                     n
#define G_k                     k
#define G_w                     w
#define G_l                     31u
#define G_d                     2u                  // Depth of the search Tree
#define LOG_Q                   1u                  // unused
#define G_q                     1u                  // unused
#define G_p                     1u
#define G_l1                    (15)
#define G_l2                    (G_l-G_l1)
#define G_w1                    0u
#define G_w2                    2u
#define G_epsilon               0u
#define NUMBER_THREADS          256
#define SORT_INCREASING_ORDER
#define VALUE_BINARY
//#define CHECK_PERM

static  std::vector<uint64_t>           __level_translation_array{{G_n-G_k-G_l, G_n-G_k-G_l+G_l2, G_n-G_k}};
constexpr std::array<std::array<uint8_t, 2>, 2>   __level_filter_array{{ {{0,0}}, {{0,0}} }};

#include "helper.h"
#include "matrix.h"
#include "combinations.h"
#include "decoding.h"
#include "emz.h"
#include "../test.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// l    47, l1   25, l2   22, w1    0, w2    3, tries 1, log(Permutationen) = 25.85. log(max(Listen))   = 24
TEST(MCEliece_d2, t1161_Set1) {
	mzd_t *AT = mzd_from_str(n, n - k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n - k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init(ss->nrows, ss->ncols);
    mzd_t *correct_e = mzd_from_str(1,n,correct_e_str);
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint64_t c  = 0;
	constexpr uint64_t l  = 34;
	constexpr uint64_t l1 = 17;
	constexpr uint64_t n_prime = n-c;
	constexpr uint64_t k_prime = k-c;

	using DecodingValue     = Value_T<BinaryContainer<k_prime + l>>;
	using DecodingLabel     = Label_T<BinaryContainer<n - k>>;
	using DecodingMatrix    = mzd_t *;
	using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList      = List_T<DecodingElement>;

	for (int i = 0; i < 1; ++i) {
        auto start = std::chrono::system_clock::now();
		emz_d2<G_n, 0, G_k, l, G_w, G_p, G_d, l1, 0, (l - l1), 2, 0, 0, 18, 1, 0, 0, DecodingList>(ee, ss,
		                                                                                           A/*,correct_e*/);
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout<< "\ntime in ms:"<< elapsed.count()<<"\n\n";
        EXPECT_EQ(w, hamming_weight(ee));

	}

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

// l    34, l1   14, l2   20, w1    0, w2    2, tries 1, log(Permutationen) = 31.09. log(max(Listen))   = 19.432
TEST(MCEliece_d2, t1161_Set2) {
	mzd_t *AT = mzd_from_str(n, n - k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n - k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init(ss->nrows, ss->ncols);
	mzd_t *correct_e = mzd_from_str(1,n,correct_e_str);
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint64_t c  = 0;
	constexpr uint64_t l  = 34;
	constexpr uint64_t l1 = 14;
	constexpr uint64_t n_prime = n-c;
	constexpr uint64_t k_prime = k-c;

	using DecodingValue     = Value_T<BinaryContainer<k_prime + l>>;
	using DecodingLabel     = Label_T<BinaryContainer<n - k>>;
	using DecodingMatrix    = mzd_t *;
	using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList      = List_T<DecodingElement>;

	for (int i = 0; i < 1; ++i) {
		emz_d2<G_n, 0, G_k, l, G_w, G_p, G_d, l1, 0, (l - l1), 2, 0, 0, 18, 1, 0, 0, DecodingList>(ee, ss,
		                                                                                           A/*,correct_e*/);
		EXPECT_EQ(w, hamming_weight(ee));
	}

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

// l    34, l1   15, l2   19, w1    0, w2    2, tries 2,
TEST(MCEliece_d2, t1161_Set2b) {
	mzd_t *AT = mzd_from_str(n, n - k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n - k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init(ss->nrows, ss->ncols);
	mzd_t *correct_e = mzd_from_str(1,n,correct_e_str);
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint64_t c  = 0;
	constexpr uint64_t l  = 34;
	constexpr uint64_t l1 = 15;
	constexpr uint64_t n_prime = n-c;
	constexpr uint64_t k_prime = k-c;

	using DecodingValue     = Value_T<BinaryContainer<k_prime + l>>;
	using DecodingLabel     = Label_T<BinaryContainer<n - k>>;
	using DecodingMatrix    = mzd_t *;
	using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList      = List_T<DecodingElement>;

	for (int i = 0; i < 1; ++i) {
		emz_d2<G_n, 0, G_k, l, G_w, G_p, G_d, l1, 0, (l - l1), 2, 0, 0, 18, 2, 0, 0, DecodingList>(ee, ss,
		                                                                                           A/*,correct_e*/);
		EXPECT_EQ(w, hamming_weight(ee));
	}

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

// l    34, l1   16, l2   18, w1    0, w2    2, tries 4
TEST(MCEliece_d2, t1161_Set2c) {
	mzd_t *AT = mzd_from_str(n, n - k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n - k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init(ss->nrows, ss->ncols);
	mzd_t *correct_e = mzd_from_str(1,n,correct_e_str);
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint64_t c  = 0;
	constexpr uint64_t l  = 34;
	constexpr uint64_t l1 = 16;
	constexpr uint64_t n_prime = n-c;
	constexpr uint64_t k_prime = k-c;

	using DecodingValue     = Value_T<BinaryContainer<k_prime + l>>;
	using DecodingLabel     = Label_T<BinaryContainer<n - k>>;
	using DecodingMatrix    = mzd_t *;
	using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList      = List_T<DecodingElement>;

	for (int i = 0; i < 1; ++i) {
		emz_d2<G_n, 0, G_k, l, G_w, G_p, G_d, l1, 0, (l - l1), 2, 0, 0, 18, 4, 0, 0, DecodingList>(ee, ss,
		                                                                                           A/*,correct_e*/);
		EXPECT_EQ(w, hamming_weight(ee));
	}

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

// l    34, l1   17, l2   17, w1    0, w2    2, tries 8
TEST(MCEliece_d2, t1161_Set2d) {
	mzd_t *AT = mzd_from_str(n, n - k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n - k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init(ss->nrows, ss->ncols);
	mzd_t *correct_e = mzd_from_str(1,n,correct_e_str);
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint64_t c  = 0;
	constexpr uint64_t l  = 34;
	constexpr uint64_t l1 = 17;
	constexpr uint64_t n_prime = n-c;
	constexpr uint64_t k_prime = k-c;

	using DecodingValue     = Value_T<BinaryContainer<k_prime + l>>;
	using DecodingLabel     = Label_T<BinaryContainer<n - k>>;
	using DecodingMatrix    = mzd_t *;
	using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList      = List_T<DecodingElement>;

	for (int i = 0; i < 1; ++i) {
		emz_d2<G_n, 0, G_k, l, G_w, G_p, G_d, l1, 0, (l - l1), 2, 0, 0, 18, 8, 0, 0, DecodingList>(ee, ss,
		                                                                                           A/*,correct_e*/);
		EXPECT_EQ(w, hamming_weight(ee));
	}

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

// l    34, l1   17, l2   17, w1    0, w2    2, tries 8
TEST(MCEliece_d2, t1161_Set2d_custom) {
	mzd_t *AT = mzd_from_str(n, n - k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n - k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init(ss->nrows, ss->ncols);
	mzd_t *correct_e = mzd_from_str(1,n,correct_e_str);
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint64_t c  = 0;
	constexpr uint64_t l  = 34;
	constexpr uint64_t l1 = 17;
	constexpr uint64_t n_prime = n-c;
	constexpr uint64_t k_prime = k-c;

	using DecodingValue     = Value_T<BinaryContainer<k_prime + l>>;
	using DecodingLabel     = Label_T<BinaryContainer<n - k>>;
	using DecodingMatrix    = mzd_t *;
	using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList      = List_T<DecodingElement>;

	for (int i = 0; i < 1; ++i) {
		emz_d2<G_n, 0, G_k, l, G_w, G_p, G_d, l1, 0, (l - l1), 3, 0, 0, 18, 1, 0, 0, DecodingList>(ee, ss,
		                                                                                           A/*,correct_e*/);
		EXPECT_EQ(w, hamming_weight(ee));
	}

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

// l    40, l1   21, l2   19, w1    1, w2    2, tries 1
TEST(MCEliece_d2, t1161_Set3) {
	mzd_t *AT = mzd_from_str(n, n - k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n - k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init(ss->nrows, ss->ncols);
	mzd_t *correct_e = mzd_from_str(1,n,correct_e_str);
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint64_t c  = 0;
	constexpr uint64_t l  = 34;
	constexpr uint64_t l1 = 21;
	constexpr uint64_t n_prime = n-c;
	constexpr uint64_t k_prime = k-c;

	using DecodingValue     = Value_T<BinaryContainer<k_prime + l>>;
	using DecodingLabel     = Label_T<BinaryContainer<n - k>>;
	using DecodingMatrix    = mzd_t *;
	using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList      = List_T<DecodingElement>;

	for (int i = 0; i < 1; ++i) {
		emz_d2<G_n, 0, G_k, l, G_w, G_p, G_d, l1, 1, (l - l1), 2, 0, 0, 18, 8, 0, 0, DecodingList>(ee, ss,
		                                                                                           A/*,correct_e*/);
		EXPECT_EQ(w, hamming_weight(ee));
	}

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

// l    34, l1   16, l2   18, w1    1, w2    2, tries 1
TEST(MCEliece_d2, t1161_Set4) {
	mzd_t *AT = mzd_from_str(n, n - k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n - k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init(ss->nrows, ss->ncols);
	mzd_t *correct_e = mzd_from_str(1,n,correct_e_str);
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	constexpr uint64_t c  = 0;
	constexpr uint64_t l  = 34;
	constexpr uint64_t l1 = 16;
	constexpr uint64_t n_prime = n-c;
	constexpr uint64_t k_prime = k-c;

	using DecodingValue     = Value_T<BinaryContainer<k_prime + l>>;
	using DecodingLabel     = Label_T<BinaryContainer<n - k>>;
	using DecodingMatrix    = mzd_t *;
	using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList      = List_T<DecodingElement>;

	for (int i = 0; i < 1; ++i) {
		emz_d2<G_n, 0, G_k, l, G_w, G_p, G_d, l1, 0, (l - l1), 2, 0, 0, 18, 1, 0, 0, DecodingList>(ee, ss,
		                                                                                           A/*,correct_e*/);
		EXPECT_EQ(w, hamming_weight(ee));
	}

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
	ident();

    random_seed(time(NULL));

	return RUN_ALL_TESTS();
}
