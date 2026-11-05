#include <gtest/gtest.h>

#include "../challenges/mce240.h"
#include "cuda/cuPollard.cuh"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(cuPollard, t240p1) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=7,.c=0,.threads=1};
	static constexpr ConfigPollard config{isdConfig, 1u<<8, true, true};

	cuPollard<isdConfig, config> pollard{};
	pollard.from_string(h, s);
	pollard.run();
	EXPECT_EQ(pollard.correct(), true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	srand(time(NULL));
	rng_seed(rand());
	return RUN_ALL_TESTS();
}
