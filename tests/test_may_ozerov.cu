#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

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
#undef SMALLSECRETLWE_DECODING_FILE_H
namespace c1284 {
#include "mce1284.h"
}

#include "may_ozerov.h"

// the GPU level-1 lists must be the CPU ones, up to the order
template<typename MO>
static void check_lists(const char *h, const char *s) {
	auto *mo = new MO{};
	mo->from_string(h, s);
	mo->cuda_alloc();
	for (uint32_t it = 0; it < 3; it++) {
		mo->cuda_step(0);
		mo->cuda_tree(0);
		const uint32_t t = it;
		mo->cuda_merge(0, t);
		checkCudaErrors(cudaStreamSynchronize(mo->stream));

		constexpr uint32_t L = MO::NR_LIMBS;
		using A = std::array<uint64_t, L>;
		std::vector<A> g1(mo->hE[0]), g2(mo->hE[1]);
		checkCudaErrors(cudaMemcpy(g1.data(), mo->algo.dL1, g1.size() * sizeof(A), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(g2.data(), mo->algo.dL2, g2.size() * sizeof(A), cudaMemcpyDeviceToHost));

		mo->construct_base_lists();
		const size_t e1 = mo->template merge<false>(mo->algo.L1, t);
		const size_t e2 = mo->template merge<true>(mo->algo.L2, t ^ uint32_t(mo->syndrome));
		std::vector<A> c1((A *)mo->algo.L1, (A *)mo->algo.L1 + e1), c2((A *)mo->algo.L2, (A *)mo->algo.L2 + e2);

		ASSERT_EQ(mo->dropped, 0u);
		ASSERT_EQ(g1.size(), e1);
		ASSERT_EQ(g2.size(), e2);
		std::sort(g1.begin(), g1.end()); std::sort(c1.begin(), c1.end());
		std::sort(g2.begin(), g2.end()); std::sort(c2.begin(), c2.end());
		EXPECT_EQ(g1, c1);
		EXPECT_EQ(g2, c2);
	}
	delete mo;
}

// the scan of `n` random values must be the CPU exclusive scan
static void check_scan(const uint32_t n) {
	const uint32_t tiles = (n + MO_SCAN_TILE - 1u) / MO_SCAN_TILE;
	std::vector<uint32_t> in(n), out(n), cur(n - 1u);
	for (auto &x : in) { x = rng() % 16u; }

	uint32_t *dIn, *dOut, *dCur, *dSums;
	checkCudaErrors(cudaMalloc(&dIn, n * sizeof(uint32_t)));
	checkCudaErrors(cudaMalloc(&dOut, n * sizeof(uint32_t)));
	checkCudaErrors(cudaMalloc(&dCur, n * sizeof(uint32_t)));
	checkCudaErrors(cudaMalloc(&dSums, tiles * sizeof(uint32_t)));
	checkCudaErrors(cudaMemcpy(dIn, in.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
	mo_scan_tiles<<<tiles, MO_SCAN_THREADS>>>(dOut, dSums, dIn, n);
	mo_scan_sums<<<1, MO_SCAN_THREADS>>>(dSums, tiles);
	mo_scan_add<<<1024, 256>>>(dOut, dCur, dSums, n);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(out.data(), dOut, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cur.data(), dCur, (n - 1u) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	cudaFree(dIn); cudaFree(dOut); cudaFree(dCur); cudaFree(dSums);

	size_t errors = 0;
	uint32_t sum = 0;
	for (uint32_t i = 0; i < n; i++) {
		errors += out[i] != sum;
		errors += (i + 1u < n) && (cur[i] != sum);
		sum += in[i];
	}
	EXPECT_EQ(errors, 0u);
}

TEST(MayOzerov, cuda_scan) {
	check_scan(1u);
	check_scan(17u);
	check_scan(MO_SCAN_TILE);
	check_scan((1u << 16u) + 1u);
	// NOTE: more tile sums than one tile, `mo_scan_sums` loops
	check_scan((1u << 22u) + 1u);
}

// n-k-l = 74 bits: 2 limbs
TEST(MayOzerov, cuda_lists_431) {
	using namespace c431;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=12,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 2, 20, 14, 32, 1024, 1, false, true};
	check_lists<MayOzerov<isdConfig, config>>(h, s);
}

// n-k-l = 240 bits: 4 limbs
TEST(MayOzerov, cuda_lists_1284) {
	using namespace c1284;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=16,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 2, 36, 4, 16, 1024, 1, false, true};
	check_lists<MayOzerov<isdConfig, config>>(h, s);
}

// p = 3, n-k-l = 38 bits: 1 limb
TEST(MayOzerov, cuda_lists_100p3) {
	using namespace c100;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=3,.l=12,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 1, 10, 10, 32, 1u << 13u, 8, false, true};
	check_lists<MayOzerov<isdConfig, config>>(h, s);
}

// same as `MayOzerov.t240p1` in tests/mceliece/mo/may_ozerov.cpp, but with `cuda`
TEST(MayOzerov, t240p1_cuda) {
	using namespace c240;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=4,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 1, 10, 10, 32, 256, 4, false, true};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

// same as `MayOzerov.t431p1nn`, but with `cuda`
TEST(MayOzerov, t431p1_cuda) {
	using namespace c431;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=4,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 2, 10, 14, 32, 256, 4, false, true};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

// same as `MayOzerov.t100p2`, but with `cuda`
TEST(MayOzerov, t100p2_cuda) {
	using namespace c100;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=8,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 1, 10, 10, 32, 1024, 8, false, true};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

// same as `MayOzerov.t100p3`, but with `cuda`
TEST(MayOzerov, t100p3_cuda) {
	using namespace c100;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=3,.l=12,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 1, 10, 10, 32, 1u << 13u, 8, false, true};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

// same as `t431p1_cuda`, but with the flat NN tree (graph, no merge sync).
// NOTE: the flat tree never bruteforces small nodes, hence N = 50 for lists of ~1900
TEST(MayOzerov, t431p1_cuda_flat) {
	using namespace c431;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=4,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 2, 50, 12, 32, 256, 4, false, true, true};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

// same as `t100p2_cuda`, but with the flat NN tree (2 levels)
TEST(MayOzerov, t100p2_cuda_flat) {
	using namespace c100;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=8,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 2, 10, 5, 16, 1024, 8, false, true, true};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

// same as `t431p2_cuda`, but with the flat NN tree
TEST(MayOzerov, t431p2_cuda_flat) {
	using namespace c431;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=12,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 2, 50, 11, 32, 4096, 16, false, true, true};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

// p = 2 with lists of ~60k elements: 1 level, ~1.5k elements per root
TEST(MayOzerov, t431p2_cuda) {
	using namespace c431;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=12,.c=0,.threads=1};
	static constexpr ConfigMayOzerov config{isdConfig, 1, 100, 10, 32, 4096, 16, false, true};

	MayOzerov<isdConfig, config> mo{};
	mo.from_string(h, s);
	mo.run();
	EXPECT_EQ(mo.correct(), true);
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	srand(time(NULL));
	rng_seed(rand());
	return RUN_ALL_TESTS();
}
