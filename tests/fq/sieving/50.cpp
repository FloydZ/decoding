#include <gtest/gtest.h>

#include "../challenges/50_20_4.h"
#include "fq/sieving.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(Sieving, t1p2) {
	// 5 sieving steps of 3 coordinates
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=q,.w=w,.p=2,.l=15,.c=0,.threads=1};
	static constexpr ConfigFqSieving config{isdConfig, 20, 5, 3};

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
