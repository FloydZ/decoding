#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "challenges/t100transformed.h"

#define NUMBER_THREADS          1u
#define CUSTOM_ALIGNMENT 4096
#define BINARY_CONTAINER_ALIGNMENT
//#define USE_PREFETCH

#include "matrix.h"
#include "ternary.h"
#include "../test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// Some tested parameters
//  l,      l1,      w,      a,      Bucket Size            information
// 15,       7,      6,      0,         10/10
// 11,       6,      5,      0,         10/10           Gauss zu teuer 36%
// 13,       4,      6,      0,         100/30
// 23,      12,      2,      5,         50/50
TEST(MMT, t100) {
	constexpr uint32_t l  = 21;//17;//16;
	constexpr uint32_t l1 = 11;//11;//8;
	constexpr uint32_t p  = 3;
	constexpr uint32_t a  = 48;//40;

	// unbeschraenkt
	static constexpr ConfigMMTTernary MMTconfig1(n, k, w/2, p, l, l1, l1, l-l1, 15, 15,
	                                            5, 2, 1, k+l-a, SinglePartialSingle);
	// disjunt
	static constexpr ConfigMMTTernary MMTconfig2(n, k, w/2, p, 13, 7, 7, 13-7, 20, 4,
	                                            0, 6, 1, k+13-0, SinglePartialSingle);
	// mem1
	static constexpr ConfigMMTTernary MMTconfig3(n, k, w/2, p, 17, 8, 8, 17-8, 5, 10,
												3, 3, 1, k+17-32, SinglePartialSingle);
	// disjunt
	static constexpr ConfigMMTTernary MMTconfig4(n, k, w/2, p, 13, 7, 7, 13-7, 20, 12,
	                                             0, 6, 1, k+13-0, EnumSinglePartialSingle);
	// mem1
	static constexpr ConfigMMTTernary MMTconfig5(n, k, w/2, p, 17, 8, 8, 17-8, 20, 15,
	                                             3, 3, 1, k+17-32, EnumSinglePartialSingle);

	static constexpr ConfigMMTTernary MMTconfig6(n, k, w/2, p, 15, 7, 7, 15-7, 10, 14,
	                                             0, 6, 1, k+15-0, EnumSinglePartialSingle);

	static constexpr ConfigMMTTernary MMTconfig7(n, k, w/2, p, 26, 13, 13, 26-13, 22, 5,
	                                             6, 1, 1, k+26-56, EnumSinglePartialSingle);


	using MatrixType = TernaryMMT<MMTconfig1>::Matrix;

	MatrixType AT(n-k, n, h);
	MatrixType A(n, n-k);
	MatrixType syndrom(n-k, 1, s);
	MatrixType syndrom_tmp(n-k, 1);
	MatrixType error_transfomer(n, 1);
	for (uint32_t i = 0; i < n; ++i) {
		error_transfomer.set(2, 0, i);
	}
	MatrixType error(n, 1);
	MatrixType::transpose(A, AT);
	TernaryMMT<MMTconfig7> MMT(&error, A, syndrom);
	MMT.MMT();
	A.matrix_vector_mul(syndrom_tmp, error);

	std::cout << "Error:\n";
	error.print();

	std::cout << "Syndrome should vs is:\n";
	syndrom.print();
	syndrom_tmp.print();
	for (uint32_t i = 0; i < n - k; ++i) {
		EXPECT_EQ(unsigned(syndrom.get(0, i)), unsigned(syndrom_tmp.get(0, i)));
	}
	MatrixType::InternalRowType::sub(error.__data[0], error.__data[0], error_transfomer.__data[0]);
	error.print();
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(rand()*time(nullptr));
	return RUN_ALL_TESTS();
}