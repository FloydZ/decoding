#include <gtest/gtest.h>
#include <iostream>

#include "../challenges/50_20_4.h"
#include "fq/stern.h"
#include "fq/stern_v2.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(Stern, t1p1) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=q,.w=12,.p=1,.l=5,.c=0,.threads=1};
	static constexpr ConfigStern config{isdConfig, .HM_bucketsize=10};
	FqStern<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(SternV2, t1p1) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=q,.w=12,.p=1,.l=5,.c=0,.threads=1};
	static constexpr ConfigStern config{isdConfig, .HM_bucketsize=10};
	FqSternV2<isdConfig, config> stern{};
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