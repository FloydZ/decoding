#include <gtest/gtest.h>

#include "challenges/lw20.h"
#include "fq/prange.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// NOTE: large weight, hence the information part is enumerated with weight k-1
TEST(Prange, t20) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=q,.w=w,.p=k-1,.l=0,.c=0,.threads=1,
	                                     .compare=compare_weight<std::greater<uint32_t>>};

	FqPrange<isdConfig> prange{};
	prange.from_string(h, s);
	prange.run();
	EXPECT_EQ(prange.correct(), true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	srand(time(NULL));
	rng_seed(rand());
	return RUN_ALL_TESTS();
}
