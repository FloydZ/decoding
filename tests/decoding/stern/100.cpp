#include <gtest/gtest.h>
#include <iostream>

#include "../challenges/100.h"
#include "stern.h"
#include "stern_im.h"
#include "stern_mo.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(Stern, t100) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=8,.c=0,.threads=1};
	static constexpr ConfigStern config{isdConfig, 16};

	Stern<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(SternIM, t100) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=8,.c=0,.threads=1};
	static constexpr ConfigStern configStern{isdConfig, 16};
	static constexpr ConfigSternIM config{isdConfig, 2};

	SternIM<isdConfig, configStern, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(SternMO, t100) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=0,.c=0,.threads=1};
	// n-k = 50: r = 1 limb of 50 bits
	static constexpr ConfigSternMO config{isdConfig, 1, 50, 16, 50};

	SternMO<isdConfig, config> stern{};
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