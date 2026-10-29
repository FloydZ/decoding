#include <gtest/gtest.h>

#include "../decoding/challenges/100.h"
#include "challenges/codeword.h"
#include "stern.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// decodingchallenge.org: Challenges/SD/SD_100_0, reduced to a low weight
// codeword challenge, see `append_syndrome`.
constexpr uint64_t lw_n = n + 1;
constexpr uint64_t lw_k = k + 1;
constexpr uint64_t lw_w = w + 1;
constexpr auto lw_h = append_syndrome<n, k>(h, s);
constexpr auto lw_s = zero_syndrome<n - k>();

TEST(Stern, lw101) {
	static constexpr ConfigISD isdConfig{.n=lw_n,.k=lw_k,.q=2,.w=lw_w,.p=2,.l=8,.c=0,.threads=1};
	static constexpr ConfigStern config{isdConfig, 16};

	Stern<isdConfig, config> stern{};
	stern.from_string(lw_h.data(), lw_s.data());
	stern.run();
	EXPECT_EQ(stern.correct(), true);
	// the codeword must be non-zero
	EXPECT_GT(stern.e.row_popcnt(0), 0u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	srand(time(NULL));
	rng_seed(rand());
	return RUN_ALL_TESTS();
}
