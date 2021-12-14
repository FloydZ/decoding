#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

//#include "challenges/t10.h"
#include "challenges/t10transformed.h"

#define SSLWE_CONFIG_SET
#define G_n                     n
#define G_k                     k
#define G_w                     w
#define G_l                     5u
#define G_d                     0u                  // Depth of the search Tree
#define LOG_Q                   1u                  // unused
#define G_q                     1u                  // unused
#define BaseList_p              1u

#define G_l1                    2u
#define G_t                     1u
#define G_epsilon               0u
#define NUMBER_THREADS          1u


#define SORT_INCREASING_ORDER
#define VALUE_BINARY
#define SORT_PARALLEL

// DO NOT DISABLE
#define CUSTOM_ALIGNMENT 4096
#define BINARY_CONTAINER_ALIGNMENT
//#define USE_PREFETCH

static  std::vector<uint64_t>                     __level_translation_array{{G_n-G_k-G_l, G_n-G_k}};
constexpr std::array<std::array<uint8_t, 1>, 1>   __level_filter_array{{ {{0}} }};

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

//TEST(MatrixVectorMult, t10) {
//	// NOTE: This only works if the NON transformed instances are loaded.
//	static constexpr ConfigTernary config(G_n, G_k, G_w, 3);
//	using RowType = kAryPackedContainer_T<uint64_t, 3, G_n>;
//
//	TernaryMatrix<RowType> AT(n-k, n, h);
//	TernaryMatrix<RowType> A(n, n-k);
//	TernaryMatrix<RowType> syndrom(n-k, 1, s);
//	TernaryMatrix<RowType> syndrom_tmp(n-k, 1);
//	TernaryMatrix<RowType> error(n, 1, "2101212111");
//	TernaryMatrix<RowType>::transpose(A, AT);
//	A.matrix_vector_mul(syndrom_tmp, error);
//
//
//	for (int i = 0; i < n - k; ++i) {
//		EXPECT_GE(syndrom.get(0, i), syndrom_tmp.get(0, i));
//	}
//}

TEST(MatrixVectorMult, t10transformed) {
	// NOTE: This only works if the transformed instances are loaded.
	static constexpr ConfigTernary config(G_n, G_k, G_w, 3);
	using RowType = kAryPackedContainer_T<uint64_t, 3, G_n>;

	TernaryMatrix<RowType> AT(n-k, n, h);
	TernaryMatrix<RowType> A(n, n-k);
	TernaryMatrix<RowType> syndrom(n-k, 1, s);
	TernaryMatrix<RowType> syndrom_tmp(n-k, 1);
	TernaryMatrix<RowType> syndrom_correction(n-k, 1);

	TernaryMatrix<RowType> error(n, 1, "2101212111");
	TernaryMatrix<RowType> error_correction(n, 1, "2222222222");

	TernaryMatrix<RowType>::transpose(A, AT);
	A.matrix_vector_mul(syndrom_tmp, error);
	A.matrix_vector_mul(syndrom_correction, error_correction);

//	syndrom.print();
//	syndrom_correction.print();
//	syndrom_tmp.print();

	RowType::sub(syndrom_tmp.__data[0], syndrom_tmp.__data[0], syndrom_correction.__data[0], 0, n-k);

//	syndrom_tmp.print();

	for (int i = 0; i < n - k; ++i) {
		EXPECT_GE(syndrom.get(0, i), syndrom_tmp.get(0, i));
	}
}

TEST(Prange, t10) {
	static constexpr ConfigTernary config(G_n, G_k, G_w/2, 1);
	using MatrixType = Ternary<config>::MatrixType;

	MatrixType AT(n-k, n, h);
	MatrixType A(n, n-k);
	MatrixType syndrom(n-k, 1, s);
	MatrixType syndrom_tmp(n-k, 1);

	MatrixType error(n, 1);
	MatrixType::transpose(A, AT);

	Ternary<config> t(&error, A, syndrom);
	t.run();

	A.matrix_vector_mul(syndrom_tmp, error);
	std::cout << "Syndrome should vs is:\n";
	syndrom.print();
	syndrom_tmp.print();

	error.print();

	for (int i = 0; i < n - k; ++i) {
		EXPECT_EQ(syndrom.get(0, i), syndrom_tmp.get(0, i));
	}
}

TEST(MMT, t10) {
	static constexpr ConfigMMTTernary MMTconfig(G_n, G_k, G_w, 4, 2, 1, 1, 1, 10, 10, 1, 0);

	using MatrixType = TernaryMMT<MMTconfig>::Matrix;

	MatrixType AT(n-k, n, h);
	MatrixType A(n, n-k);
	MatrixType syndrom(n-k, 1, s);
	MatrixType syndrom_tmp(n-k, 1);

	MatrixType error(n, 1);
	MatrixType::transpose(A, AT);
	TernaryMMT<MMTconfig> MMT(&error, A, syndrom);
	MMT.MMT();

	A.matrix_vector_mul(syndrom_tmp, error);

	std::cout << "Syndrome should vs is:\n";
	syndrom.print();
	syndrom_tmp.print();

	for (int i = 0; i < n - k; ++i) {
		EXPECT_EQ(syndrom.get(0, i), syndrom_tmp.get(0, i));
	}
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(rand()*time(nullptr));
	return RUN_ALL_TESTS();
}
