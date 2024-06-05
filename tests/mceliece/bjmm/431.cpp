#include <gtest/gtest.h>
#include <iostream>

#include "../challenges/mce431.h"
#include "bjmm.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(BJMM, t431t1p2) {
	constexpr uint32_t bucketsize1 = 16;
	constexpr uint32_t bucketsize2 = 8;

	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=16,.c=0,.threads=1};
	static constexpr ConfigBJMM config{isdConfig, .l1=2, .HM1_bucketsize=bucketsize1, .HM2_bucketsize=bucketsize2};

	BJMM<isdConfig, config> bjmm{};
	bjmm.from_string(h, s);
	bjmm.run();
	EXPECT_EQ(bjmm.correct(), true);
}

TEST(BJMM, t431t1p2c20) {
	constexpr uint32_t bucketsize1 = 16;
	constexpr uint32_t bucketsize2 = 8;

	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=16,.c=20,.threads=1};
	static constexpr ConfigBJMM config{isdConfig, .l1=2, .HM1_bucketsize=bucketsize1, .HM2_bucketsize=bucketsize2};

	BJMM<isdConfig, config> bjmm{};
	bjmm.from_string(h, s);
	bjmm.run();
	EXPECT_EQ(bjmm.correct(), true);
}

TEST(EB, t431t1p2) {
	constexpr uint32_t bucketsize1 = 16;
	constexpr uint32_t bucketsize2 = 8;

	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=16,.c=0,.threads=1};
	static constexpr ConfigBJMM configBJMM{isdConfig, .l1=2, .HM1_bucketsize=bucketsize1, .HM2_bucketsize=bucketsize2};
	static constexpr ConfigEB config{isdConfig, .l1=2, .HM1_bucketsize=bucketsize1, .HM2_bucketsize=bucketsize2, .p2=1};

	EB<isdConfig, configBJMM, config> bjmm{};
	bjmm.from_string(h, s);
	bjmm.run();
	EXPECT_EQ(bjmm.correct(), true);
}
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	srand(time(NULL));
	random_seed(rand());
	return RUN_ALL_TESTS();
}
