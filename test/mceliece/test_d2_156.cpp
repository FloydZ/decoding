#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "challenges/mce156.h"

#define SSLWE_CONFIG_SET
#define G_n                     n
#define G_k                     k
#define G_w                     w
#define G_l                     16
#define G_d                     2u                  // Depth of the search Tree
#define LOG_Q                   1u                  // unused
#define G_q                     1u                  // unused
#define G_p                     1u

#define G_l1                    (12)
#define G_l2                    (G_l-G_l1)
#define G_w1                    1u
#define G_w2                    1u

#define G_epsilon                0u
#define NUMBER_THREADS 1
#define SORT_INCREASING_ORDER
#define VALUE_BINARY

static  std::vector<uint64_t>                     __level_translation_array{{G_n-G_k-G_l, G_n-G_k-G_l+G_l1, G_n-G_k}};
constexpr std::array<std::array<uint8_t, 1>, 1>   __level_filter_array{{ {{0}} }};

#include "helper.h"
#include "matrix.h"
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

TEST(MCEliece_d2, t156) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);

	// print_matrix("A", A);

	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	auto aswd = 0;
	for (int i = 0; i < 1; ++i) {
		std::cout << i << "\n";
		aswd += emz_d2<G_n, 0, G_k, G_l, G_w, G_p, G_d, G_l1, G_w1, G_l2, G_w2, G_epsilon>(ee, ss, A);
		EXPECT_EQ(w, hamming_weight(ee));
	}

	std::cout << aswd  << "\n";
	print_matrix("FINAL A: ", A);
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
    //output final e as binary string
    Value_T<BinaryContainer<n>> e_outi;
    e_outi.data().from_m4ri(ee);
    std::cout<<e_outi<<"\n";

	mzd_free(AT);
	mzd_free(A);
	mzd_free(ss);
	mzd_free(ss_tmp);
	mzd_free(ss_tmp_T);
	mzd_free(ee);
}
/*
TEST(MCEliece_d2, t156_with_custom_e_s) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);

	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(n-k, 1);
	mzd_t *ss_tmp_T = mzd_init(1, n-k);

	mzd_t *ee_rand = mzd_init(1, n);
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	mzd_write_bit(ee_rand, 0, 0, 1);
	for (int i = n-k; i < n-k+1; ++i) {
		mzd_write_bit(ee_rand, 0, i, 1);
	}
	for (int i = n-(k/2); i < n-(k/2)+1; ++i) {
		mzd_write_bit(ee_rand, 0, i, 1);
	}

	print_matrix("ss", ss);
	print_matrix("ee_rand", ee_rand);

	auto aswd = 0;
	for (int i = 0; i < 1; ++i) {
		aswd += mceliece_d2<G_n, 0, G_k, G_l, G_w, G_p, G_d, G_l1, G_w1, G_l2, G_w2, G_epsilon>(ee, ss, A);
		EXPECT_EQ(w, hamming_weight(ee));
	}

	print_matrix("FINAL A: ", A);
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

	mzd_free(AT);
	mzd_free(A);
	mzd_free(ss);
	mzd_free(ss_tmp);
	mzd_free(ss_tmp_T);
	mzd_free(ee);
}

TEST(MCEliece_d2, prepare_generate_base_mitm2_mc_eliece_156) {
	using VValue = DecodingList::ValueType;
	// test if the two functions `mceliece_d2_fill_decoding_lists` and `prepare_generate_base_mitm2` are correctly working.
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *H_prime = mzd_init(G_n-G_k, G_k+G_l);
	mzd_t *H_prime_T = mzd_init(G_k+G_l, G_n-G_k);
	copy_submatrix(H_prime, A, 0, n-k-G_l, n-k, n);
	mzd_transpose(H_prime_T, H_prime);
	Matrix_T<mzd_t *> B((mzd_t *)H_prime);

	DecodingList List1{0}, List2{0};
	std::vector<std::pair<uint64_t, uint64_t>> diff_list1, diff_list2, diff_list3, diff_list4;

	prepare_generate_base_mitm2<DecodingList>(List1, diff_list1, diff_list2, G_l1, G_k, G_l2, G_w1, G_w2, false, 0, true);
	prepare_generate_base_mitm2<DecodingList>(List2, diff_list3, diff_list4, G_l1, G_k, G_l2, G_w1, G_w2, true, 0, true);
	// EXPECT_EQ(List1.get_size(), diff_list1.size()*diff_list2.size());
	EXPECT_EQ(List2.get_size(), bc(G_l1 - (G_l1/2), G_w1)*bc(G_k - (G_k/2), G_w2));

	std::cout << diff_list1.size() << "\n";

	mceliece_d2_fill_decoding_lists<G_n, G_k, G_l, G_l1, G_w1, G_l2, G_w2, 0>(List1, diff_list1, diff_list2, H_prime_T);
	EXPECT_EQ(true, check_correctness(List1, B));

	mceliece_d2_fill_decoding_lists<G_n, G_k, G_l, G_l1, G_w1, G_l2, G_w2, 0>(List1, List2, diff_list1, diff_list2, diff_list3, diff_list4,
																		   nullptr, H_prime_T);
	EXPECT_EQ(true, check_correctness(List1, B));
	// EXPECT_EQ(true, check_correctness(List2, B));

	mzd_free(AT);
	mzd_free(A);
	mzd_free(H_prime);
	mzd_free(H_prime_T);
}

TEST(MCEliece_d2, prepare_generate_base_mitm2_zero_w1) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *H_prime = mzd_init(G_n-G_k, G_k+G_l);
	mzd_t *H_prime_T = mzd_init(G_k+G_l, G_n-G_k);

	copy_submatrix(H_prime, A, 0, G_n-G_k-G_l, G_n-G_k, G_n);
	mzd_transpose(H_prime_T, H_prime);
	Matrix_T<mzd_t *> B((mzd_t *)H_prime);

	// Checks if the Implementation of `prepare_generate_base_mitm2` can handle zero weight.
	DecodingList List1{0}, List2{0};
	std::vector<std::pair<uint64_t, uint64_t>> diff_list1, diff_list2, diff_list3, diff_list4;

	// Left Side
	constexpr uint64_t win = G_l2;
	prepare_generate_base_mitm2<DecodingList>(List1, diff_list1, diff_list2, G_l1, G_k, win, 0, 1, false, 0, true);
	EXPECT_EQ(List1.get_size(), diff_list2.size() + 1);

	// Right Side
	prepare_generate_base_mitm2<DecodingList>(List2, diff_list3, diff_list4, G_l1, G_k, win, 0, 1, true, 0, true);
	EXPECT_EQ(List2.get_size(), diff_list4.size() + 1);

	mceliece_d2_fill_decoding_lists<G_n, G_k, G_l, G_l1, G_w1, G_l2, G_w2, G_epsilon>(List1, List2, diff_list1, diff_list2, diff_list3, diff_list4, nullptr, H_prime_T);
	EXPECT_EQ(true, check_correctness(List1, B));
	EXPECT_EQ(true, check_correctness(List2, B));

	mzd_free(AT);
	mzd_free(A);
	mzd_free(H_prime);
	mzd_free(H_prime_T);
}

TEST(MCEliece_d2, prepare_generate_base_mitm2_zero_w2) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *H_prime = mzd_init(G_n-G_k, G_k+G_l);
	mzd_t *H_prime_T = mzd_init(G_k+G_l, G_n-G_k);

	copy_submatrix(H_prime, A, 0, G_n-G_k-G_l, G_n-G_k, G_n);
	mzd_transpose(H_prime_T, H_prime);
	Matrix_T<mzd_t *> B((mzd_t *)H_prime);

	// Checks if the Implementation of `prepare_generate_base_mitm2` can handle zero weight.
	DecodingList List1{0}, List2{0};
	std::vector<std::pair<uint64_t, uint64_t>> diff_list1, diff_list2, diff_list3, diff_list4;

	// Left Side
	constexpr uint64_t win = G_l2;
	prepare_generate_base_mitm2<DecodingList>(List1, diff_list1, diff_list2, G_l1, G_k, win, 1, 0, false, 0, true);
	EXPECT_EQ(List1.get_size(), diff_list1.size()+1);

	// Right Side
	prepare_generate_base_mitm2<DecodingList>(List2, diff_list3, diff_list4, G_l1, G_k, win, 1, 0, true, 0, true);
	EXPECT_EQ(List2.get_size(), diff_list3.size()+1);

	mceliece_d2_fill_decoding_lists<G_n, G_k, G_l, G_l1, G_w1, G_l2, G_w2, G_epsilon>(List1, List2, diff_list1, diff_list2, diff_list3, diff_list4, nullptr, H_prime_T);
	EXPECT_EQ(true, check_correctness(List1, B));
	EXPECT_EQ(true, check_correctness(List2, B));

	mzd_free(AT);
	mzd_free(A);
	mzd_free(H_prime);
	mzd_free(H_prime_T);
}

TEST(MCEliece_d2, prepare_generate_base_mitm2_epsilon) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *H_prime = mzd_init(G_n-G_k, G_k+G_l);
	mzd_t *H_prime_T = mzd_init(G_k+G_l, G_n-G_k);

	copy_submatrix(H_prime, A, 0, G_n-G_k-G_l, G_n-G_k, G_n);
	mzd_transpose(H_prime_T, H_prime);
	Matrix_T<mzd_t *> B((mzd_t *)H_prime);

	// Checks if the Implementation of `prepare_generate_base_mitm2` can handle epsilon values != 0
	const uint64_t epsilon = 50;
	DecodingList List1{0}, List2{0};
	std::vector<std::pair<uint64_t, uint64_t>> diff_list1, diff_list2, diff_list3, diff_list4;

	// Formally for the McEliece Setting. This tests if List1 and 2 are correctly initialised.
	prepare_generate_base_mitm2<DecodingList>(List1, diff_list1, diff_list2, G_l1, G_k, G_l2, 1, 1, false, epsilon, true);
	prepare_generate_base_mitm2<DecodingList>(List2, diff_list3, diff_list4, G_l1, G_k, G_l2, 1, 1, true, epsilon, true);
	mceliece_d2_fill_decoding_lists<G_n, G_k, G_l, G_l1, G_w1, G_l2, G_w2, epsilon>(List1, List2, diff_list1, diff_list2, diff_list3, diff_list4, nullptr, H_prime_T);
	EXPECT_EQ(true, check_correctness(List1, B));
	EXPECT_EQ(true, check_correctness(List2, B));

	mzd_free(AT);
	mzd_free(A);
	mzd_free(H_prime);
	mzd_free(H_prime_T);
}

TEST(MCEliece_d2, prepare_generate_base_mitm2_epsilon_zero_w1) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *H_prime = mzd_init(G_n-G_k, G_k+G_l);
	mzd_t *H_prime_T = mzd_init(G_k+G_l, G_n-G_k);

	copy_submatrix(H_prime, A, 0, G_n-G_k-G_l, G_n-G_k, G_n);
	mzd_transpose(H_prime_T, H_prime);
	Matrix_T<mzd_t *> B((mzd_t *)H_prime);

	// Checks if the Implementation of `prepare_generate_base_mitm2` can handle zero weight w1 and epsilon values != 0.
	DecodingList List1{0}, List2{0};
	std::vector<std::pair<uint64_t, uint64_t>> diff_list1, diff_list2, diff_list3, diff_list4;

	const uint64_t epsilon = 50;

	prepare_generate_base_mitm2(List1, diff_list1, diff_list2, G_l1, G_k, G_l2, 0, 1, false, epsilon, true);
	prepare_generate_base_mitm2(List2, diff_list3, diff_list4, G_l1, G_k, G_l2, 0, 1, true, epsilon, true);

	mceliece_d2_fill_decoding_lists<G_n, G_k, G_l, G_l1, G_w1, G_l2, G_w2, G_epsilon>(List1, List2, diff_list1, diff_list2, diff_list3, diff_list4, nullptr, H_prime_T);
	EXPECT_EQ(true, check_correctness(List1, B));
	EXPECT_EQ(true, check_correctness(List2, B));

	mzd_free(AT);
	mzd_free(A);
	mzd_free(H_prime);
	mzd_free(H_prime_T);
}


TEST(MCEliece_d2, prepare_generate_base_mitm2_extended_156) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *H_prime = mzd_init(G_n-G_k, G_k+G_l);
	mzd_t *H_prime_T = mzd_init(G_k+G_l, G_n-G_k);

	copy_submatrix(H_prime, A, 0, G_n-G_k-G_l, G_n-G_k, G_n);
	mzd_transpose(H_prime_T, H_prime);
	Matrix_T<mzd_t *> B((mzd_t *)H_prime);

	// Checks if the Implementation of `prepare_generate_base_mitm2_extended` for correctness.
	DecodingList List1{0}, List2{0};
	std::vector<std::vector<std::pair<uint64_t, uint64_t>>> diff_list1, diff_list2, diff_list3, diff_list4;

	prepare_generate_base_mitm2_extended(List1, diff_list1, diff_list2, G_l1, G_k, G_l2, 0, 1, false, 0, true);
	prepare_generate_base_mitm2_extended(List2, diff_list3, diff_list4, G_l1, G_k, G_l2, 0, 1, true, 0, true);

	for (int i = 0; i < diff_list2.size(); ++i)  {
		const auto v = diff_list2[i];
		for (int j = 0; j < v.size(); ++j) {
			std::cout << v[j]. first << " " << v[j].second << "\n";
		}
		std::cout << "\n" << std::flush;
	}

	mceliece_d2_fill_decoding_lists<G_n, G_k, G_l, G_l1, G_w1, G_l2, G_w2, G_epsilon>(List1, List2, diff_list1, diff_list2, diff_list3, diff_list4, H_prime_T);
	EXPECT_EQ(true, check_correctness(List1, B));
	EXPECT_EQ(true, check_correctness(List2, B));

	std::cout << List1;
	std::cout << "\n";
	std::cout << List2;

	mzd_free(AT);
	mzd_free(A);
	mzd_free(H_prime);
	mzd_free(H_prime_T);
}
*/
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));

	return RUN_ALL_TESTS();
}
