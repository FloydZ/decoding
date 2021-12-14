#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "challenges/mce982.h"

#define SSLWE_CONFIG_SET
#define G_n                     n
#define G_k                     k
#define G_w                     w
#define G_l                     33u
#define G_d                     2u                  // Depth of the search Tree
#define LOG_Q                   1u                  // unused
#define G_q                     1u                  // unused
#define G_p                     1u
#define G_l1                    (15)
#define G_l2                    (G_l-G_l1)
#define G_w1                    0u
#define G_w2                    2u
#define G_epsilon               0u
#define NUMBER_THREADS 1
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
#include <chrono>
#include <omp.h>

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(MCEliece_d2, t982) {
	mzd_t *AT = mzd_from_str(n, n - k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss = mzd_from_str(1, n - k, s);
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init(ss->nrows, ss->ncols);
    mzd_t *correct_e = mzd_from_str(1,n,correct_e_str);
	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);
	auto aswd = 0;

	for (int i = 0; i < 1; ++i) {
        auto start = std::chrono::system_clock::now();
        //aswd += mceliece_d2<G_n, 0, G_k, 21, G_w, G_p, G_d, 4, 1, 17, 2, 0,13,17,1,0,0>(ee, ss, A,correct_e);
        aswd += emz_d2<G_n, 0, G_k, 33, G_w, G_p, G_d, 16, 0, 17, 2, 0, 0, 17, 1, 0, 0>(ee, ss, A
#ifdef CHECK_PERM
		        ,correct_e
#endif
        );

        //aswd += mceliece_d2<G_n, 0, G_k, 48, G_w, G_p, G_d, 22, 0, 26, 3, 0,0,18,1,0,0>(ee, ss, A,correct_e);
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout<< "\ntime in ms:"<< elapsed.count()<<"\n\n";
		//aswd += mceliece_d2<G_n, 0, G_k, 21, G_w, G_p, G_d, 4, 1, 17, 2, 0,12,17,1,0,0>(ee, ss, A);
		EXPECT_EQ(w, hamming_weight(ee));
	}

	print_matrix("FINAL e: ", ee);

	//output final e as binary string
    //Value_T<BinaryContainer<n>> e_outi;
    //e_outi.data().from_m4ri(ee);
    //std::cout<<e_outi<<"\n";


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
