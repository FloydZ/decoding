#include <gtest/gtest.h>

#include "../challenges/mce240.h"
#include "cuda/cuPCS.cuh"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(cuPCS, t240p1) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=7,.c=0,.threads=1};
	static constexpr ConfigPollard config{isdConfig, 1u<<8, true, true, 1, 8, 1};

	cuPCS<isdConfig, config> pcs{};
	pcs.from_string(h, s);
	pcs.run();
	EXPECT_EQ(pcs.correct(), true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	srand(time(NULL));
	rng_seed(rand());
	return RUN_ALL_TESTS();
}
