#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "../challenges/mce1284.h"
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

constexpr size_t loops = 10000;

TEST(Stern, t1284t1p2) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=19,.c=0,.threads=1, .loops=loops};
	static constexpr ConfigStern config{isdConfig, 8};

	Stern<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(Stern, t1284t1p2c20) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=21,.c=20,.threads=1, .loops=loops};
	static constexpr ConfigStern config{isdConfig, 4};

	Stern<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(SternIM, t1284t1p2c20) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=21,.c=20,.threads=1, .loops=loops};
	static constexpr ConfigStern configStern{isdConfig, 4};
	static constexpr ConfigSternIM config{isdConfig, 2};

	SternIM<isdConfig, configStern, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

//TEST(SternMO, t1284t1p2) {
//	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=17,.c=0,.threads=1, .loops=loops};
//	static constexpr ConfigSternMO config{isdConfig, 2};
//
//	SternMO<isdConfig, config> stern{};
//	stern.from_string(h, s);
//	stern.run();
//	EXPECT_EQ(stern.correct(), true);
//}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	srand(time(NULL));
	random_seed(rand());
	return RUN_ALL_TESTS();
}
