#include <gtest/gtest.h>

#include "challenges/qc146.h"
#include "prange.h"
#include "stern.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(Prange, qc146) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=0,.l=0,.c=0,.threads=1};

	Prange<isdConfig> prange{};
	prange.from_string(h, s);
	prange.run();
	EXPECT_EQ(prange.correct(), true);
}

TEST(Stern, qc146) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=10,.c=0,.threads=1};
	static constexpr ConfigStern config{isdConfig, 16};

	Stern<isdConfig, config> stern{};
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
