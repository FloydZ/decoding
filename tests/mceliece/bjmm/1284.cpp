#include <gtest/gtest.h>
#include <iostream>

#include "../challenges/mce640.h"
#include "bjmm.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(BJMM, t1284t1p1) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=19,.c=0,.threads=1};
	static constexpr ConfigBJMM config{isdConfig, 2, 1<<6, 4};

	BJMM<isdConfig, config> bjmm{};
	bjmm.from_string(h, s);
	bjmm.run();
	EXPECT_EQ(bjmm.correct(), true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	srand(time(NULL));
	random_seed(rand());
	return RUN_ALL_TESTS();
}
