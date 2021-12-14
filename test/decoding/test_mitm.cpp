#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "m4ri/m4ri.h"

//#include "challenges/challenge10.h"
//#include "challenges/challenge100.h"
#include "challenges/challenge200.h"
//#include "challenges/challenge300.h"

// IMPORTANT: Define 'SSLWE_CONFIG_SET' before one include 'helper.h'.
#define SSLWE_CONFIG_SET

// parameter set für d = 2 n = 200
#define G_n                     n
#define G_k                     k
#define G_w                     w
#define G_l                     16u
#define G_d                     2u                  // Depth of the search Tree
#define LOG_Q                   1u                  // unused
#define G_q                     1u                  // unused
#define G_p                     4u
#define BaseList_p              2u
#define BaseList_wl             2u

/*#define G_k                     k
#define G_l                     10
#define G_d                     1                   // Depth of the search Tree
#define G_n                     n
#define LOG_Q                   1u
#define G_q                     (1u << LOG_Q)
#define G_w                     w
#define G_p                     4
#define BaseList_p              1*/
#define SORT_INCREASING_ORDER
#define VALUE_BINARY
//#define SORT_PARALLEL

// only for 100 set.
//static  std::vector<uint64_t>                       __level_translation_array{{G_n - G_k - G_l, G_n - G_k}};
static  std::vector<uint64_t>                       __level_translation_array{{G_n - G_k - G_l, G_n - G_k - G_l+7, G_n - G_k}};
constexpr std::array<std::array<uint8_t, 3>, 3>     __level_filter_array{{ {{0,0,0}}, {{0,0,0}}, {{0,0,0}} }};

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


TEST(generate_decoding_lists, one ) {
	DecodingList l1{0};
	DecodingList l2{0};
	std::vector<std::pair<uint64_t, uint64_t>> v;

	generate_decoding_lists<G_k + G_l>(l1, l2, v, BaseList_p);

	// little helper function.
	auto apply_diff = [](DecodingValue &out, const DecodingValue &in, const std::pair<uint64_t, uint64_t> &pos, const uint64_t offset = 0) {
		out = in;
		out.data().flip(pos.first+offset);
		out.data().flip(pos.second+offset);
	};

	EXPECT_EQ(l1.get_load(), l2.get_load());
	EXPECT_EQ(l1.get_load(), v.size());
	EXPECT_EQ(l1.get_size(), l2.get_size());
	EXPECT_EQ(l1.get_size(), v.size());

	DecodingValue tmp;
	for (int i = 0; i < l1.get_size()-1; ++i) {
		apply_diff(tmp, l1[i].get_value(), v[i]);
		EXPECT_EQ(true, tmp.is_equal(l1[i+1].get_value(), 0, tmp.size()));

		apply_diff(tmp, l2[i].get_value(), v[i], (G_n - G_k)/2);
		EXPECT_EQ(true, tmp.is_equal(l2[i+1].get_value(), 0, tmp.size()));
	}

}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();

	uint64_t r = 0;
    fastrandombytes(&r, 8);
	srandom(r);

	return RUN_ALL_TESTS();
}
