#include <gtest/gtest.h>

#include "../challenges/mce1284.h"
#include "stern_mo.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr size_t loops = 10000;

TEST(SternMO, t1284t1p2) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=0,.c=0,.threads=1, .loops=loops};
	// n-k = 256: r = 4 limbs of 64 bits
	static constexpr ConfigSternMO config{isdConfig, 4, 50, 14, 64};

	SternMO<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	srand(time(NULL));
	rng_seed(rand());
	return RUN_ALL_TESTS();
}
