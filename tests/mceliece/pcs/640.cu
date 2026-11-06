#include <gtest/gtest.h>

#include "../challenges/mce640.h"
#include "cuda/cuPCS.cuh"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


// NOTE: 2^l >= bc((k+l)/2, p), otherwise not every pair is reachable
TEST(cuPCS, t640p2) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=16,.c=20,.threads=1};
	static constexpr ConfigPollard config{isdConfig, 1u<<16, true, true, 1, 14, 1};

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
