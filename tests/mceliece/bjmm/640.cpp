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


TEST(BJMM, t640t1p2) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=16,.c=0,.threads=1};
	static constexpr ConfigBJMM config{isdConfig, 2, 1<<6, 20};

	BJMM<isdConfig, config> bjmm{};
	bjmm.from_string(h, s);
	bjmm.run();
	EXPECT_EQ(bjmm.correct(), true);
}

TEST(BJMM, t640t1p2c20) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=16,.c=20,.threads=1};
	static constexpr ConfigBJMM config{isdConfig, 2, 1<<6, 20};

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
