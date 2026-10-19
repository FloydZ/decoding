#include <gtest/gtest.h>
#include <iostream>

#include <cstdint>

// NOTE: both challenges share the same include guard
namespace c100 {
#include "100.h"
}
#undef SMALLSECRETLWE_DECODING_FILE_H
namespace c431 {
#include "mce431.h"
}

#include "stern_mo.h"

// 2 levels on 64 bit limbs, small bruteforce threshold s.t. the recursion is used
constexpr NN_Config nn_config{128, 2, 100, 64, 1u << 12u, 28, 4, 0, 16};

// the GPU NN must find the inserted solution
TEST(NN, cuda_nn) {
	NN<nn_config> algo{};
	algo.generate_random_instance();
	algo.run<true>();

	EXPECT_GE(algo.solutions_nr, 1u);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	EXPECT_EQ(algo.solutions[0].first, algo.solution_l);
	EXPECT_EQ(algo.solutions[0].second, algo.solution_r);
}

// no solution inserted: nothing must be found
TEST(NN, cuda_nn_no_solution) {
	NN<nn_config> algo{};
	algo.generate_random_instance(false);
	algo.run<true>();

	EXPECT_EQ(algo.solutions_nr, 0u);
}

// `sternmo_lists` with unit vectors as rows: element `i` of `L1` must be the
// rows `biject(i)`, of `L2` the same plus the syndrome
template<const uint32_t p>
static void check_lists() {
	constexpr uint32_t half = 100, limbs = 2;
	constexpr uint32_t size = bc(half, p);
	std::vector<uint64_t> HT((2 * half + 1) * limbs, 0), L1(size_t(size) * limbs), L2(size_t(size) * limbs);
	for (uint32_t i = 0; i < half; i++) {
		HT[i * limbs + i / 64] = 1ull << (i % 64);
		HT[(i + half) * limbs + i / 64] = 1ull << (i % 64);
	}
	HT[2 * half * limbs + 1] = 1ull << 63;

	uint64_t *dHT, *dL1, *dL2;
	checkCudaErrors(cudaMalloc(&dHT, HT.size() * sizeof(uint64_t)));
	checkCudaErrors(cudaMalloc(&dL1, L1.size() * sizeof(uint64_t)));
	checkCudaErrors(cudaMalloc(&dL2, L2.size() * sizeof(uint64_t)));
	checkCudaErrors(cudaMemcpy(dHT, HT.data(), HT.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
	sternmo_lists<uint64_t, limbs, half, p><<<(size + 255) / 256, 256>>>(dL1, dL2, dHT, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(L1.data(), dL1, L1.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(L2.data(), dL2, L2.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	cudaFree(dHT); cudaFree(dL1); cudaFree(dL2);

	size_t errors = 0;
	uint16_t rows[p];
	for (uint32_t i = 0; i < size; i++) {
		biject<half, p>(i, rows);
		uint64_t e[limbs] = {0};
		for (uint32_t j = 0; j < p; j++) {
			e[rows[j] / 64] ^= 1ull << (rows[j] % 64);
		}
		errors += (L1[i * limbs] != e[0]) || (L1[i * limbs + 1] != e[1]);
		errors += (L2[i * limbs] != e[0]) || (L2[i * limbs + 1] != (e[1] ^ (1ull << 63)));
	}
	EXPECT_EQ(errors, 0u);
}

TEST(SternMO, cuda_lists_p1) { check_lists<1>(); }
TEST(SternMO, cuda_lists_p2) { check_lists<2>(); }
TEST(SternMO, cuda_lists_p3) { check_lists<3>(); }
TEST(SternMO, cuda_lists_p4) { check_lists<4>(); }

// same as `SternMO.t100` in tests/decoding/stern/100.cpp, but with `cuda`
TEST(SternMO, t100_cuda) {
	using namespace c100;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=1,.l=0,.c=0,.threads=1};
	static constexpr ConfigSternMO config{isdConfig, 1, 50, 16, 50, false, true};

	SternMO<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

// same as `SternMO.t431t1p2` in tests/mceliece/stern/431.cpp, but with `cuda`
TEST(SternMO, t431_cuda) {
	using namespace c431;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=0,.c=0,.threads=1};
	static constexpr ConfigSternMO config{isdConfig, 2, 50, 15, 43, false, true};

	SternMO<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

// same as `t431_cuda`, but with p = 3
TEST(SternMO, t431p3_cuda) {
	using namespace c431;
	static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=3,.l=0,.c=0,.threads=1};
	static constexpr ConfigSternMO config{isdConfig, 2, 50, 12, 43, false, true};

	SternMO<isdConfig, config> stern{};
	stern.from_string(h, s);
	stern.run();
	EXPECT_EQ(stern.correct(), true);
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	srand(time(NULL));
	rng_seed(rand());
	return RUN_ALL_TESTS();
}
