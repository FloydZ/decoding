#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

// NOTE: all challenges share the same include guard
namespace c100 {
#include "100.h"
}
#undef SMALLSECRETLWE_DECODING_FILE_H
namespace c240 {
#include "mce240.h"
}
#undef SMALLSECRETLWE_DECODING_FILE_H
namespace c431 {
#include "mce431.h"
}

#include "may_ozerov.h"

using ::testing::InitGoogleTest;

// n-k-l = 44 bits, single limb
TEST(MayOzerov, t240p1) {
	using namespace c240;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=4,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 1, 10, 10, 32, 256, 4};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

// same as `t240p1`, but always the bruteforce instead of the NN
TEST(MayOzerov, t240p1bf) {
	using namespace c240;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=4,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 1, 10, 10, 32, 256, 4, true};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

// n-k-l = 78 bits, two limbs, bruteforce on small lists
TEST(MayOzerov, t431p1) {
	using namespace c431;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=8,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 1, 10, 14, 32, 1024, 64};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

// n-k-l = 80 bits, the lists are above `bf`: NN with 2 levels
TEST(MayOzerov, t431p1nn) {
	using namespace c431;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=6,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 2, 10, 14, 32, 64, 16};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

// p = 2, n-k-l = 42 bits
TEST(MayOzerov, t100p2) {
	using namespace c100;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=8,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 1, 10, 10, 32, 1024, 8};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

// p = 3, n-k-l = 38 bits
TEST(MayOzerov, t100p3) {
	using namespace c100;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=3,.l=12,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 1, 10, 10, 32, 1u << 13u, 8};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	srand(time(NULL));
	rng_seed(rand());
	return RUN_ALL_TESTS();
}
