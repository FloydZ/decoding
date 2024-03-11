#include <gtest/gtest.h>
#include <iostream>

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
	constexpr uint32_t bucketsize = 16;

	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=q,.w=w,.p=2,.l=3,.c=0,.threads=1};
	static constexpr ConfigFqSieving config{isdConfig, .HM_bs=20, .sieving_steps=5, .enumeration_q=3};

	FqSieving<isdConfig, config> sieve{};
	sieve.from_string(h, s);
	sieve.run();
	EXPECT_EQ(sieve.correct(), true);
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	srand(time(NULL));
	random_seed(rand());
	return RUN_ALL_TESTS();
}