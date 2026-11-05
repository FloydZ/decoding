#include <gtest/gtest.h>
#include <iostream>

#include "../challenges/mce640.h"
#include "cuda/cuLeeBrickellNoMem.cuh"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// NOTE: p=2 needs ~2^20 permutations, p=3 ~2^16
TEST(cuLeeBrickellNoMem, t640p3) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=3,.l=0,.c=20,.threads=1};

	cuLeeBrickellNoMem<isdConfig> lb{};
	lb.from_string(h, s);
	lb.run();
	EXPECT_EQ(lb.correct(), true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	srand(time(NULL));
	rng_seed(rand());
	return RUN_ALL_TESTS();
}
