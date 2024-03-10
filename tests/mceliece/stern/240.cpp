#include <gtest/gtest.h>
#include <iostream>

#include "../challenges/mce240.h"
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


TEST(Stern, t240) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=13,.c=0,.threads=1};
	static constexpr ConfigStern config{isdConfig, .HM_bucketsize=16, };

	Stern<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(Stern, t240c20) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=13,.c=20,.threads=1};
	static constexpr ConfigStern config{isdConfig, .HM_bucketsize=16, };

	Stern<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(SternIM, t240) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=10,.c=0,.threads=1};
	static constexpr ConfigStern configStern{isdConfig, .HM_bucketsize=16};
	static constexpr ConfigSternIM config{isdConfig, .nr_views=2};

	SternIM<isdConfig, configStern, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(SternIM, t240c20) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=10,.c=20,.threads=1};
	static constexpr ConfigStern configStern{isdConfig, .HM_bucketsize=16};
	static constexpr ConfigSternIM config{isdConfig, .nr_views=2};

	SternIM<isdConfig, configStern, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(SternMO, t240) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=10,.c=0,.threads=1};
	static constexpr ConfigSternMO config{isdConfig, .nr_views=2};

	SternMO<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(SternMO, t240c20) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=10,.c=20,.threads=1};
	static constexpr ConfigSternMO config{isdConfig, .nr_views=2};

	SternMO<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	srand(time(NULL));
	random_seed(rand());
	return RUN_ALL_TESTS();
}