#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"
#include "mceliece/challenges/mce156.h"

// IMPORTANT: Define 'SSLWE_CONFIG_SET' before one include 'helper.h'.
#ifndef SSLWE_CONFIG_SET
#define SSLWE_CONFIG_SET
#define G_n                     n
#define G_k                     k
#define G_w                     w
#define G_l                     1u
#define G_d                     0u                  // unused
#define LOG_Q                   1u                  // unused
#define G_q                     (1u << LOG_Q)       // unused
#define G_l1                    1u                  // unused
#define G_l2                    (G_l-G_l1)          // unused
#define G_w1                    1u                  // unused
#define G_w2                    1u                  // unused

#define G_t                     10u                 // Additional Columns in the Sparsisty Matrix.
#define G_epsilon                0u                 // unused
#define NUMBER_THREADS 1

#define SORT_INCREASING_ORDER
#define VALUE_BINARY

#if G_n >= 30
static const std::vector<uint64_t> __level_translation_array{ {0, 10, 20, 30, G_n} };
#else
static const std::vector<uint64_t> __level_translation_array{ {0, G_n/2, G_n} };
#endif
constexpr std::array<std::array<uint8_t, 3>, 3>   __level_filter_array{{ {{4,0,0}}, {{1,0,0}}, {{0,0,0}} }};
#endif

#include "helper.h"
#include "decoding.h"
#include "sparsity.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(mzd_perm, permut) {
	constexpr uint32_t n = 1161;
	constexpr uint32_t k = 929;

	mzd_t *A = mzd_init(k, n);
	mzd_t *A_T = mzd_init(n, k);
	mzd_t *B = mzd_init(k, n);




	mzd_free(A);
	mzd_free(A_T);
	mzd_free(B);
}

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	return RUN_ALL_TESTS();
}
#endif
