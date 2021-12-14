#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>
#include <chrono>

#include "m4ri/m4ri.h"
#include "challenges/mce431_v2.h"

#define SSLWE_CONFIG_SET
#define G_n                     n
#define G_k                     k
#define G_w                     w
#define G_l                     13 //28
#define G_d                     2u                  // Depth of the search Tree
#define LOG_Q                   1u                  // unused
#define G_q                     1u                  // unused
#define G_p                     1u
#define G_l1                    2 //(14)
#define G_l2                    (G_l-G_l1)
#define G_w1                    0u
#define G_w2                    1u
#define G_epsilon               0u

#define NUMBER_THREADS 1
#define SORT_INCREASING_ORDER
#define VALUE_BINARY
//#define CHECK_PERM

//#define CUSTOM_ALIGNMENT 4096
//#define BINARY_CONTAINER_ALIGNMENT
//#define USE_PREFETCH

// IMPORTANT
// sudo rdmsr --all 0x1a4
// sudo wrmsr --all 0x1a4 1
// reset: sudo wrmsr --all 0x1a4 0

static  std::vector<uint64_t>           __level_translation_array{{G_n-G_k-G_l, G_n-G_k-G_l+G_l1, G_n-G_k}};
constexpr std::array<std::array<uint8_t, 1>, 1>   __level_filter_array{{ {{0}} }};

#include "helper.h"
#include "matrix.h"
#include "decoding.h"
#include "emz.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// TODO not finished implementing
//TEST(EMZ_NewHashmap, t431) {
//	mzd_t *AT = mzd_from_str(n, n-k, h);
//	mzd_t *A = mzd_transpose(NULL, AT);
//	mzd_t *ss = mzd_from_str(1, n-k, s);
//	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
//	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);
//
//	mzd_t *ee = mzd_init(1, n);
//	mzd_t *ee_T = mzd_init(n, 1);
//
//	constexpr uint32_t r1 = 0;
//	constexpr uint32_t tries = 14;
//	constexpr uint32_t nbretries = 18;
//
//	static constexpr ConfigEMZ2DThread config(G_n, G_k, G_l, G_w, G_p, G_l1, G_l2, G_w1, G_w2, G_epsilon, r1, tries, nbretries);
//	EMZ<config, DecodingList> emz(ee, ss, A);
//	uint64_t loops = emz.emz_new_data_structure_d2_thread();
//
//
//	std::cout << "Loops: " << loops << "\n";
//	print_matrix("FINAL e: ", ee);
//	print_matrix("FINAL s: ", ss);
//
//	// check output:
//	mzd_transpose(ee_T, ee);
//	mzd_mul_naive(ss_tmp, A, ee_T);
//	mzd_transpose(ss_tmp_T, ss_tmp);
//	print_matrix("FINAL ss:", ss_tmp_T);
//
//	for (int i = 0; i < ss->ncols; ++i) {
//		EXPECT_EQ(mzd_read_bit(ss, 0, i), mzd_read_bit(ss_tmp, i, 0));
//	}
//
//	mzd_free(AT);
//	mzd_free(A);
//	mzd_free(ss);
//	mzd_free(ss_tmp);
//	mzd_free(ss_tmp_T);
//	mzd_free(ee);
//	mzd_free(ee_T);
//}

TEST(MCEliece_d2, t431) {
	DecodingElement tt;
	DecodingList t{10};
	t.set_load(10);
//	printf("%p\n", t.data());
//	printf("%p, %p\n", t[0].get_label_container_ptr(), t[0].get_value_container_ptr());
//	printf("%p, %p\n", t[1].get_label_container_ptr(), t[1].get_value_container_ptr());
//	printf("%p, %p\n", t[2].get_label_container_ptr(), t[2].get_value_container_ptr());
//	printf("%lu, %lu\n", DecodingElement::size(), DecodingElement::bytes());

	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);
#ifdef CHECK_PERM
	mzd_t *correct_e = mzd_from_str(1,n, correct_e_str);
#endif
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	auto loops = 0;
	for (int i = 0; i < 1; ++i) {
	    std::cout << i << "\n";
		loops += emz_d2<G_n, 0, G_k, G_l, G_w, G_p, G_d, G_l1, G_w1, G_l2, G_w2, G_epsilon, 0, 14, 8, 0, 0>(ee, ss, A
#ifdef CHECK_PERM
				,correct_e
#endif
		);

		EXPECT_EQ(w, hamming_weight(ee));
	}
	std::cout << "Loops: " << loops << "\n";
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
	mzd_free(ee_T);

#ifdef CHECK_PERM
	mzd_free(correct_e);
#endif
}


/*
TEST(MCEliece_d2_outer, t431) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);
    mzd_t *correct_e = mzd_from_str(1,n,correct_e_str);
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	auto aswd = 0;
	for (int i = 0; i < 1; ++i) {
		aswd += mceliece_d2_outer<G_n, 50, G_k, G_l, G_w, G_p, G_d, G_l1, G_w1, G_l2, G_w2, G_epsilon, 0, 15, 1, 0, 9999>(ee, ss, A);
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

	mzd_free(AT);
	mzd_free(A);
	mzd_free(ss);
	mzd_free(ss_tmp);
	mzd_free(ss_tmp_T);
	mzd_free(ee_T);
	mzd_free(ee);
#ifdef CHECK_PERM
	mzd_free(correct_e);
#endif
}

// this test evaluates our algorithm `mceliece_d2` with a crafted `e` and `s`.
TEST(MCEliece_d2, t431_with_custom_e_s) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
    mzd_t *correct_e = mzd_from_str(1,n,correct_e_str);
	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(n-k, 1);
	mzd_t *ss_tmp_T = mzd_init(1, n-k);

	mzd_t *ee_rand = mzd_init(1, n);
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	// create a special e of the form:
	// 0        +l1 +l2   +k     +k/2     n
	// [11      |0  |0    |1111   |1111   ]
	mzd_write_bit(ee_rand, 0, 0, 1);
	mzd_write_bit(ee_rand, 0, 1, 1);

	for (int i = n-k; i < n-k+4; ++i) { mzd_write_bit(ee_rand, 0, i, 1); }
	for (int i = n-k+(k/2); i < n-k+(k/2)+4; ++i) { mzd_write_bit(ee_rand, 0, i, 1); }

	mzd_mul_naive(ss, ee_rand, AT);

	mceliece_d2<G_n, 0, G_k, G_l, G_w, G_p, G_d, G_l1, G_w1, G_l2, G_w2, G_epsilon>(ee, ss, A
#ifdef CHECK_PERM
																				 ,correct_e
#endif
				);
	EXPECT_EQ(w, hamming_weight(ee));

	// check output:
	mzd_transpose(ee_T, ee);
	mzd_mul_naive(ss_tmp, A, ee_T);
	mzd_transpose(ss_tmp_T, ss_tmp);
	for (int i = 0; i < ss->ncols; ++i) { EXPECT_EQ(mzd_read_bit(ss, 0, i), mzd_read_bit(ss_tmp, i, 0)); }
	for (int i = 0; i < ee->ncols; ++i) { EXPECT_EQ(mzd_read_bit(ee_rand, 0, i), mzd_read_bit(ee, 0, i)); }

	// free everything.
	mzd_free(AT);
	mzd_free(A);
	mzd_free(ss);
	mzd_free(ss_tmp);
	mzd_free(ss_tmp_T);
	mzd_free(ee);
	mzd_free(ee_T);
	mzd_free(ee_rand);
#ifdef CHECK_PERM
	mzd_free(correct_e);
#endif
}

#ifndef CUSTOM_ALIGNMENT
// This test evaluates a special unrolled algorithm. and special parameters to ensure a fast tree join algorithm.
TEST(MCEliece_d2, t431_with_custom_e_s_custom_algo) {
	constexpr uint64_t l = 28;
	constexpr uint64_t l1 = 13;
	constexpr uint64_t l2 = l-l1;
	constexpr uint64_t w1 = 0;
	constexpr uint64_t w2 = 2;
	constexpr uint64_t epsilon = 0;

	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);

	mzd_t *ss = mzd_from_str(1, n-k, s);
	mzd_t *ss_tmp = mzd_init(n-k, 1);
	mzd_t *ss_tmp_T = mzd_init(1, n-k);

	mzd_t *ee_rand = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);

	// create a special e of the form:
	// 0        +l1 +l2   +k     +k/2     n
	// [11      |0  |0    |1111   |1111   ]
	mzd_write_bit(ee_rand, 0, 0, 1);
	mzd_write_bit(ee_rand, 0, 1, 1);

	for (int i = n-k; i < n-k+4; ++i) { mzd_write_bit(ee_rand, 0, i, 1); }
	for (int i = n-k+(k/2); i < n-k+(k/2)+4; ++i) { mzd_write_bit(ee_rand, 0, i, 1); }

	mzd_mul_naive(ss, ee_rand, AT);


//######################################################################################################################
	// NOW rewrite the Algorithm.
	// for the matrix permutations. Fix the permutation.
	mzp_t *P_C = mzp_init(n);
	for(rci_t i = 0; i < n; i++) P_C->values[i] = i;

	mzd_t *work_matrix_H = mzd_init(A->nrows, A->ncols+1);
	mzd_t *columnTransposed = mzd_transpose(NULL, ss);
	mzd_concat(work_matrix_H, A, columnTransposed);

	// helper functions to better access H_1, H_2
	mzd_t *H_prime = mzd_init(n-k, k+l);
	mzd_t *H_prime_T = mzd_init(k+l, n-k);

	mzd_t *working_s = mzd_init(n-k, 1);
	mzd_t *working_s_T = mzd_init(1, n-k);

	mzd_echelonize(work_matrix_H, true);

	copy_submatrix(working_s, work_matrix_H, 0, n, n-k, n+1);
	mzd_transpose(working_s_T, working_s);

	copy_submatrix(H_prime, work_matrix_H, 0, n-k-l, n-k, n);
	mzd_transpose(H_prime_T, H_prime);

	Matrix_T<mzd_t *> B(H_prime);
	Matrix_T<mzd_t *> B2(work_matrix_H);


	DecodingList List1{0}, List2{0}, List3{0}, List4{0}, out{0}, iL{0};
	std::vector<std::pair<uint64_t, uint64_t>> changelist11, changelist12, changelist21, changelist22;
	prepare_generate_base_mitm2(List1, changelist11, changelist12, l1, k, l2, w1, w2, false, epsilon, true, true);
	prepare_generate_base_mitm2(List2, changelist21, changelist22, l1, k, l2, w1, w2, true, epsilon, true, true);

	mceliece_d2_fill_decoding_lists<n, k, l, l1, w1, l2, w2, epsilon>(List1, List2, changelist11, changelist12, changelist21, changelist22, H_prime_T);
	List3 = List1;
	List4 = List2;
	ASSERT(check_correctness(List1, B));
	ASSERT(check_correctness(List2, B));
	ASSERT(check_correctness(List3, B));
	ASSERT(check_correctness(List4, B));

	// extract the label.
	DecodingLabel target; target.data().from_m4ri(working_s_T);

	// now start the tree merge
	constexpr uint64_t k_lower1 = n-k-l;
	constexpr uint64_t k_upper1 = n-k-l+l1;
	constexpr uint64_t k_lower2 = n-k-l+l1;
	constexpr uint64_t k_upper2 = n-k;

	DecodingElement T, R; R.zero();
	DecodingLabel zero; zero.zero();

	// create the intermediate target
	for (int i = l; i < l+2; ++i) {
		R.get_value()[i] = true;
	}

	for (int i = l+(k/2); i < l+(k/2)+2; ++i) {
		R.get_value()[i] = true;
	}

	R.recalculate_label(B);

	// create the element to search for
	for (int i = l; i < l+4; ++i) {
		T.get_value()[i] = true;
	}
	for (int i = l+(k/2); i < l+(k/2)+4; ++i) {
		T.get_value()[i] = true;
	}


	if (!target.data().is_zero()) {
		for (int i = 0; i < List2.get_load(); ++i) {
			DecodingLabel::add(List2[i].get_label(), List2[i].get_label(), R.get_label(), k_lower1, k_upper1);
		}

		for (int i = 0; i < List4.get_load(); ++i) {
			DecodingLabel::sub(List4[i].get_label(), List4[i].get_label(), R.get_label(), k_lower1, k_upper1);
			// add is on the full length
			DecodingLabel::sub(List4[i].get_label(), List4[i].get_label(), target, 0, k_upper2);
		}
	}

	DecodingTree::join2lists(iL, List1, List2, zero, k_lower1, k_upper1, false);

	// Check if the searched element is the intermediate List.
	for (int i = 0; i < iL.get_load(); ++i) {
		if (T.get_value().is_equal(iL[i].get_value(), 0, T.get_value_size())){
			std::cout << "found\n";
		}
	}

	// Now run the merge procedure for the right part of the tree.
	DecodingTree::twolevel_streamjoin(out, iL, List3, List4, k_lower1, k_upper1, k_lower2, k_upper2);

	// Check if the searched element is the final List.
	for (int i = 0; i < out.get_load(); ++i) {
		if (T.get_value().is_equal(out[i].get_value(), 0, T.get_value_size())){
			std::cout << "found\n";
		}
	}

	if (out.get_load() == 0){
		std::cout << "empty\n";
	}

	std::cout << List2.get_load() << "\n";

	std::cout << iL.get_load() << "\n";
	std::cout << out.get_load() << "\n";

	mzd_t *ee = nullptr;//check_resultlist<DecodingList, n, 0, k, l, l1, w>(out, w, ss, working_s_T, P_C, A, H_prime);

	ASSERT(ee != nullptr);

//######################################################################################################################

	EXPECT_EQ(w, hamming_weight(ee));

	// check output:
	mzd_transpose(ee_T, ee);
	mzd_mul_naive(ss_tmp, A, ee_T);
	mzd_transpose(ss_tmp_T, ss_tmp);
	for (int i = 0; i < ss->ncols; ++i) { EXPECT_EQ(mzd_read_bit(ss, 0, i), mzd_read_bit(ss_tmp, i, 0)); }
	for (int i = 0; i < ee->ncols; ++i) { EXPECT_EQ(mzd_read_bit(ee_rand, 0, i), mzd_read_bit(ee, 0, i)); }

	// free everything.
	mzd_free(AT);
	mzd_free(A);
	mzd_free(ss);
	mzd_free(ss_tmp);
	mzd_free(ss_tmp_T);
	mzd_free(ee);
	mzd_free(ee_T);
	mzd_free(ee_rand);
}
#endif
TEST(MCEliece_d2, prepare_generate_base_mitm2_mc_eliece_431) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *H_prime = mzd_init(G_n-G_k, G_k+G_l);
	mzd_t *H_prime_T = mzd_init(G_k+G_l, G_n-G_k);

	mzd_echelonize(A, true);

	copy_submatrix(H_prime, A, 0, n-k-G_l, n-k, n);
	mzd_transpose(H_prime_T, H_prime);
	Matrix_T<mzd_t *> B((mzd_t *)H_prime);

	DecodingList List1{0}, List2{0};
	std::vector<std::pair<uint64_t, uint64_t>> diff_list1, diff_list2, diff_list3, diff_list4;

	prepare_generate_base_mitm2(List1, diff_list1, diff_list2, G_l1, G_k, G_l2, 1, 1, false, 0, true);
	prepare_generate_base_mitm2(List2, diff_list3, diff_list4, G_l1, G_k, G_l2, 1, 1, true, 0, true);
	EXPECT_EQ(List1.get_size(), (diff_list1.size()+1)*diff_list2.size()+1);
	EXPECT_EQ(List1.get_size(), bc(G_l1/2, 1)*bc(G_k/2, 1));
	EXPECT_EQ(List2.get_size(), bc(G_l1 - (G_l1/2), 1)*bc(G_k - (G_k/2), 1));

	mceliece_d2_fill_decoding_lists<G_n, G_k, G_l, G_l1, 1, G_l2, 1, 0>(List1, List2, diff_list1, diff_list2, diff_list3, diff_list4,  H_prime_T);
	EXPECT_EQ(true, check_correctness(List1, B));
	EXPECT_EQ(true, check_correctness(List2, B));

	mzd_free(AT);
	mzd_free(A);
	mzd_free(H_prime);
	mzd_free(H_prime_T);
}

TEST(MCEliece_d2, prepare_generate_base_mitm2_epsilon_431) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *H_prime = mzd_init(G_n-G_k, G_k+G_l);
	mzd_t *H_prime_T = mzd_init(G_k+G_l, G_n-G_k);

	mzd_echelonize(A, true);

	copy_submatrix(H_prime, A, 0, n-k-G_l, n-k, n);
	mzd_transpose(H_prime_T, H_prime);

	Matrix_T<mzd_t *> B((mzd_t *)H_prime);

	// Checks if the Implementation of `prepare_generate_base_mitm2` can handle epsilon values != 0
	const uint64_t epsilon = 1;
	DecodingList List1{0}, List2{0};
	std::vector<std::pair<uint64_t, uint64_t>> diff_list1, diff_list2, diff_list3, diff_list4;

	// Formally for the McEliece Setting. This tests if List1 and 2 are correctly initialised.
	prepare_generate_base_mitm2<DecodingList>(List1, diff_list1, diff_list2, G_l1, G_k, G_l2, G_w1, G_w2, false, epsilon, true);
	prepare_generate_base_mitm2<DecodingList>(List2, diff_list3, diff_list4, G_l1, G_k, G_l2, G_w1, G_w2, true, epsilon, true);
	mceliece_d2_fill_decoding_lists<G_n, G_k, G_l, G_l1, G_w1, G_l2, G_w2, epsilon>(List1, List2, diff_list1, diff_list2, diff_list3, diff_list4, H_prime_T);
	EXPECT_EQ(true, check_correctness(List1, B));
	EXPECT_EQ(true, check_correctness(List2, B));

	mzd_free(AT);
	mzd_free(A);
	mzd_free(H_prime);
	mzd_free(H_prime_T);
}

TEST(MCEliece_d2, prepare_generate_base_mitm2_extended_431) {
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *H_prime = mzd_init(G_n-G_k, G_k+G_l);
	mzd_t *H_prime_T = mzd_init(G_k+G_l, G_n-G_k);

	mzd_echelonize(A, true);

	copy_submatrix(H_prime, A, 0, G_n-G_k-G_l, G_n-G_k, G_n);
	mzd_transpose(H_prime_T, H_prime);
	Matrix_T<mzd_t *> B((mzd_t *)H_prime);

	constexpr uint64_t w1 = 1;
	constexpr uint64_t w2 = 2;

	// Checks if the Implementation of `prepare_generate_base_mitm2_extended` for correctness.
	DecodingList List1{0}, List2{0};
	std::vector<std::vector<std::pair<uint64_t, uint64_t>>> diff_list1, diff_list2, diff_list3, diff_list4;

	prepare_generate_base_mitm2_extended(List1, diff_list1, diff_list2, G_l1, G_k, G_l2, w1, w2, false, 0, true);
	prepare_generate_base_mitm2_extended(List2, diff_list3, diff_list4, G_l1, G_k, G_l2, w1, w2, true, 0, true);

	print_diff_list(diff_list1[0]);

	mceliece_d2_fill_decoding_lists<G_n, G_k, G_l, G_l1, w1, G_l2, w2, 0>(List1, List2, diff_list1, diff_list2, diff_list3, diff_list4, H_prime_T);
	EXPECT_EQ(true, check_correctness(List1, B));
	EXPECT_EQ(true, check_correctness(List2, B));

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
