#include <gtest/gtest.h>
#include <iostream>

#include "../challenges/50_20_4.h"
#include "fq/prange.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(Prange, t1p0) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=q,.w=w,.p=0,.l=0,.c=0,.threads=1};

	FqPrange<isdConfig> prange{};
	prange.from_string(h, s);
	prange.run();
	EXPECT_EQ(prange.correct(), true);
}

TEST(Prange, q3t1p1) {
	static constexpr ConfigISD isdConfig{.n=50,.k=20,.q=3,.w=2,.p=1,.l=0,.c=0,.threads=1};
	FqPrange<isdConfig> prange{};
	prange.random();
	prange.run();
	EXPECT_EQ(prange.correct(), true);
}

TEST(Prange, q3t1p1c5) {
	static constexpr ConfigISD isdConfig{.n=50,.k=20,.q=3,.w=2,.p=1,.l=0,.c=5,.threads=1};
	FqPrange<isdConfig> prange{};
	prange.random();
	prange.run();
	EXPECT_EQ(prange.correct(), true);
}

TEST(Prange, t1p1) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=q,.w=w-2,.p=1,.l=0,.c=0,.threads=1};

	FqPrange<isdConfig> prange{};
	prange.from_string(h, s);
	prange.run();
	EXPECT_EQ(prange.correct(), true);
}

TEST(Prange, t1p2) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=q,.w=w,.p=2,.l=0,.c=0,.threads=1};

	FqPrange<isdConfig> prange{};
	prange.from_string(h, s);
	prange.run();
	EXPECT_EQ(prange.correct(), true);
}
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	srand(time(NULL));
	random_seed(rand());
	return RUN_ALL_TESTS();
}