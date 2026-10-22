#include <gtest/gtest.h>

#include "../challenges/mce431.h"
#include "pcs.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(PCS, t431p2) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=14,.c=0,.threads=1};
	static constexpr ConfigPCS config{isdConfig, 4, 1u<<14, 14};

	PCS<isdConfig, config> pcs{};
	pcs.from_string(h, s);
	pcs.run();
	EXPECT_EQ(pcs.correct(), true);
}

TEST(PCS, t431p2c20) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=14,.c=20,.threads=1};
	static constexpr ConfigPCS config{isdConfig, 4, 1u<<14, 14};

	PCS<isdConfig, config> pcs{};
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
