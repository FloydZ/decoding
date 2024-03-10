#include <gtest/gtest.h>
#include <iostream>

#include "decoding/challenges/100.h"
#include "mitm.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(EnumHashMapConfig, print) {
	static constexpr ConfigISD config{.n=n,.k=k,.q=2,.w=w,.p=2,.l=10,.c=0,.threads=1};
	config.print();
}

TEST(EnumHashMap, fromstring) {
	static constexpr ConfigISD config{.n=n,.k=k,.q=2,.w=w,.p=2,.l=10,.c=0,.threads=1};
	ISDInstance<uint64_t, config> isd{};
	isd.from_string(h, s);
	isd.print();
}

TEST(EnumHashMap, random) {
	static constexpr ConfigISD config{.n=n,.k=k,.q=2,.w=w,.p=2,.l=10,.c=0,.threads=1};
	ISDInstance<uint64_t, config> isd{};
	isd.random();

	isd.wA.print();
	isd.s.print();
	isd.e.print();
}

TEST(EnumHashMap, step) {
	constexpr uint32_t l = 10;
	static constexpr ConfigISD config{.n=n,.k=k,.q=2,.w=w,.p=2,.l=l,.c=20,.threads=1};
	ISDInstance<uint64_t, config> isd{};
	isd.from_string(h, s);

	for (size_t i = 0; i < 1; ++i) {
		isd.step();

		// check systemized
		for (uint32_t j = 0; j < n-k; ++j) {
			for (uint32_t m = 0; m < n-k-l; ++m) {
				EXPECT_EQ(isd.wA.get(j, m), j == m);
			}
		}

		// check syndrome, note: swapped
		for (uint32_t j = 0; j < n - k; ++j) {
			EXPECT_EQ(isd.wA.get(j, n), isd.ws.get(n-k-j-1));
		}

		// note swapped
		for (uint32_t j = 0; j < l; ++j) {
			EXPECT_EQ(isd.ws.get(j), (isd.syndrome >> j) & 1u);
		}

		// check submatrix, note H is swapped
		for (uint32_t j = 0; j < n-k; ++j) {
			for (uint32_t m = 0; m < k+l; ++m) {
				EXPECT_EQ(isd.wA.get(j, m+(n-k-l)), isd.H.get(n-k-j-1, m));
			}
		}

		// check transpose/swap
		for (uint32_t j = 0; j < n-k; ++j) {
			for (uint32_t m = 0; m < k+l; ++m) {
				EXPECT_EQ(isd.H.get(j, m), isd.HT.get(m, j));
			}
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	srand(time(NULL));
	random_seed(rand());
	return RUN_ALL_TESTS();
}
