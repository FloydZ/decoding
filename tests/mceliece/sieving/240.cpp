#include <gtest/gtest.h>

#include "../challenges/mce240.h"
#include "fq/sieving.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(Sieving, t240) {
	// 3 sieving steps of 4 coordinates
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=12,.c=0,.threads=1};
	static constexpr ConfigFqSieving config{isdConfig, 16, 3, 2};

	FqSieving<isdConfig, config> sieve{};
	sieve.from_string(h, s);
	sieve.run();
	EXPECT_EQ(sieve.correct(), true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	srand(time(NULL));
	rng_seed(rand());
	return RUN_ALL_TESTS();
}
