#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "challenges/t60transformed.h"

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

TEST(MMT, t60) {
	constexpr uint32_t l = 8;
	constexpr uint32_t l1 = 4;
	constexpr uint32_t p = 3;
	constexpr uint32_t a = 8;

	// sb1 = 10, sb2=50 if Enumerationtype = SinglePartialSingle
	static constexpr ConfigMMTTernary MMTconfig(n, k, w/2, p, l, l1, l1, l-l1, 100, 500,
												1, 2, 1, k+l-a, SinglePartialSingle);

	using MatrixType = TernaryMMT<MMTconfig>::Matrix;

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
	TernaryMMT<MMTconfig> MMT(&error, A, syndrom);
	MMT.MMT();
	A.matrix_vector_mul(syndrom_tmp, error);

	std::cout << "Error:\n";
	error.print();

	std::cout << "Syndrome should vs is:\n";
	syndrom.print();
	syndrom_tmp.print();
	for (int i = 0; i < n - k; ++i) {
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