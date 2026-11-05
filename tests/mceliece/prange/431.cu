#include <gtest/gtest.h>
#include <iostream>

#include "../challenges/mce431.h"
#include "cuda/cuLeeBrickell.cuh"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// NOTE: nvcc needs unique names for function local configs
TEST(cuLeeBrickell, t431p2) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=0,.c=0,.threads=1};

	cuLeeBrickell<isdConfig> lb{};
	lb.from_string(h, s);
	lb.run();
	EXPECT_EQ(lb.correct(), true);
}

TEST(cuLeeBrickell, t431p2c10) {
	static constexpr ConfigISD isdConfigc10{.n=n,.k=k,.q=2,.w=w,.p=2,.l=0,.c=10,.threads=1};

	cuLeeBrickell<isdConfigc10> lb{};
	lb.from_string(h, s);
	lb.run();
	EXPECT_EQ(lb.correct(), true);
}

TEST(cuLeeBrickell, t431p3) {
	static constexpr ConfigISD isdConfigp3{.n=n,.k=k,.q=2,.w=w,.p=3,.l=0,.c=0,.threads=1};

	cuLeeBrickell<isdConfigp3> lb{};
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
