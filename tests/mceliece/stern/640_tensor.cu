#include <gtest/gtest.h>

#include "../challenges/mce640.h"
#include "cuda/cuStern_Tensor.cuh"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


// NOTE: nvcc needs unique names for function local configs
TEST(cuSternTensor, t640c20) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=16,.c=20,.threads=1};
	static constexpr ConfigStern config{isdConfig, 16};

	cuSternTensor<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

// full buckets, s.t. the tensor tiles are used
TEST(cuSternTensor, t640l13c20) {
	static constexpr ConfigISD isdConfigl13{.n=n,.k=k,.q=2,.w=w,.p=2,.l=13,.c=20,.threads=1};
	static constexpr ConfigStern configl13{isdConfigl13, 16};

	cuSternTensor<isdConfigl13, configl13> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	srand(time(NULL));
	rng_seed(rand());
	return RUN_ALL_TESTS();
}
