#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "../challenges/mce240.h"
#include "bjmm.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(Bjmm, t240) {
	constexpr uint32_t bucketsize1 = 16;
	constexpr uint32_t bucketsize2 = 8;

	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=10,.c=0,.threads=1};
	static constexpr ConfigBJMM config{isdConfig, .l1=4, .HM1_bucketsize=bucketsize1, .HM2_bucketsize=bucketsize2};

	BJMM<isdConfig, config> bjmm{};
	bjmm.from_string(h, s);
	bjmm.run();
	EXPECT_EQ(bjmm.correct(), true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	srand(time(NULL));
	random_seed(rand());
	return RUN_ALL_TESTS();
}
