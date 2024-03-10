#include <gtest/gtest.h>
#include <iostream>

#include "../challenges/mce431.h"
#include "prange.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(Prange, t431) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=0,.l=0,.c=0,.threads=1};

	Prange<isdConfig> prange{};
	prange.from_string(h, s);
	prange.run();
	EXPECT_EQ(prange.correct(), true);
}

TEST(Prange, t431c10) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=0,.l=0,.c=10,.threads=1};
	Prange<isdConfig> prange{};
	prange.from_string(h, s);
	prange.run();
	EXPECT_EQ(prange.correct(), true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	srand(time(NULL));
	random_seed(rand());
	return RUN_ALL_TESTS();
}