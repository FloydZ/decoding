#include <gtest/gtest.h>

#include "../challenges/mce431.h"
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


TEST(Stern, t431t1p2) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=13,.c=0,.threads=1};
	static constexpr ConfigStern config{isdConfig, 16};

	Stern<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(Stern, t431t1p2c20) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=13,.c=20,.threads=1};
	static constexpr ConfigStern config{isdConfig, 16};

	Stern<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(SternIM, t431t1p2) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=17,.c=0,.threads=1};
	static constexpr ConfigStern configStern{isdConfig, 16};
	static constexpr ConfigSternIM config{isdConfig, 3};

	SternIM<isdConfig, configStern, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

TEST(SternIM, t431t1p2c20) {
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=19,.c=20,.threads=1};
	static constexpr ConfigStern configStern{isdConfig, 16};
	static constexpr ConfigSternIM config{isdConfig, 3};

	SternIM<isdConfig, configStern, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

// to small. bruteforcing finds to many solutions
//TEST(SternMO, t431t1p2) {
//	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=19,.c=0,.threads=1};
//	static constexpr ConfigSternMO config{isdConfig};
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
