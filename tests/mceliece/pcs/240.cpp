#include <gtest/gtest.h>

#include "../challenges/mce240.h"
#include "pcs.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(PCS, t240p2) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=13,.c=0,.threads=1};
	static constexpr ConfigPCS config{isdConfig, 4, 1u<<10, 10};

	PCS<isdConfig, config> pcs{};
	pcs.from_string(h, s);
	pcs.run();
	EXPECT_EQ(pcs.correct(), true);
}

TEST(PCS, t240p1) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=7,.c=0,.threads=1};
	static constexpr ConfigPCS config{isdConfig, 2, 1u<<8, 8};

	PCS<isdConfig, config> pcs{};
	pcs.from_string(h, s);
	pcs.run();
	EXPECT_EQ(pcs.correct(), true);
}

TEST(PCS, t240p2c20) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=13,.c=20,.threads=1};
	static constexpr ConfigPCS config{isdConfig, 4, 1u<<10, 10};

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
