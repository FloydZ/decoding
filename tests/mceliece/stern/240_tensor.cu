#include <gtest/gtest.h>

#include "../challenges/mce240.h"
#include "cuda/cuStern_Tensor.cuh"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


// NOTE: nvcc needs unique names for function local configs
TEST(cuSternTensor, t240) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=13,.c=0,.threads=1};
	static constexpr ConfigStern config{isdConfig, 16, };

	cuSternTensor<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(cuSternTensor, t240c20) {
	static constexpr ConfigISD isdConfigc20{.n=n,.k=k,.q=2,.w=w,.p=2,.l=13,.c=20,.threads=1};
	static constexpr ConfigStern configc20{isdConfigc20, 16, };

	cuSternTensor<isdConfigc20, configc20> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

// full buckets, s.t. the tensor tiles are used
TEST(cuSternTensor, t240l10) {
	static constexpr ConfigISD isdConfigl10{.n=n,.k=k,.q=2,.w=w,.p=2,.l=10,.c=0,.threads=1};
	static constexpr ConfigStern configl10{isdConfigl10, 16, };

	cuSternTensor<isdConfigl10, configl10> stern{};
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
