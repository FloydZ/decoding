#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>

#include "decoding/challenges/100.h"
#include "mitm.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(EnumHashMap, alloc) {
	constexpr uint32_t p = 2, l = 10;
	using T = MinLogTypeTemplate<l, 32>;
	using V = CollisionType<T, uint16_t, p>;

	constexpr static SimpleHashMapConfig s{.bucketsize=10u, .nrbuckets=1u<<l, .threads=1};
	using HM = SimpleHashMap<T, V, s, Hash<T, 0, l>>;
	HM *hm = new HM{};

	static constexpr ConfigEnumHashMap config{k+l, p, l, 1};
	T *data = (T *)cryptanalysislib::aligned_alloc(1024, roundToAligned<1024>(sizeof(T) * (k+l)));
	EnumHashMap<config, HM> mitm{data, hm};

	free(data);
	delete hm;
}

TEST(EnumHashMap, enumeration_changelist) {
	// as k = 50 you cannot choose l to big
	constexpr uint32_t p = 2, l = 5, epsilon=10;
	using T = MinLogTypeTemplate<l, 32>;
	using V = CollisionType<T, uint16_t, p>;

	constexpr static SimpleHashMapConfig s{.bucketsize=4u, .nrbuckets=1u<<l, .threads=1};
	using HM = SimpleHashMap<T, V, s, Hash<T, 0, l>>;
	HM *hm = new HM{};

	static constexpr ConfigEnumHashMap config{k+l, p, l, 1, epsilon, false, true};
	T *data = (T *)cryptanalysislib::aligned_alloc(1024, roundToAligned<1024>(sizeof(T) * (k+l)));
	for (uint32_t i = 0; i < k+l; ++i) {
		data[i] = fastrandombytes_uint64() & ((1u << l) - 1u);
	}

	/// create the changelist
	chase<(k+l)/2 + epsilon, p, 2> c;
	std::vector<std::pair<uint16_t, uint16_t>> cL;
	cL.resize(c.list_size());
	size_t ctr = 0;
	c.enumerate([&, this](const uint16_t p1, const uint16_t p2){
	  cL[ctr] = std::pair<uint16_t, uint16_t>{p1, p2};
	  ctr += 1;
	});

	EnumHashMap<config, HM> mitm{data, hm, cL.data()};
	mitm.print();
	for (uint32_t simd = 0; simd < 2; ++simd) {
		mitm.step(0, 0, simd);

		for (size_t i = 0; i < (k+l)/2; i++){
			HM::load_type load = 0;
			size_t k = hm->find(data[i], load);
			EXPECT_GT(load, 0);
			if constexpr (p == 1) {
				EXPECT_EQ(k, data[i] * s.bucketsize);
			}
		}
	}

	free(data);
	delete hm;
}

TEST(EnumHashMap, enumeration) {
	// as k = 50 you cannot choose l to big
	constexpr uint32_t p = 2, l = 5, epsilon=10;
	using T = MinLogTypeTemplate<l, 32>;
	using V = CollisionType<T, uint16_t, p>;

	constexpr static SimpleHashMapConfig s{.bucketsize=10u, .nrbuckets=1u<<l, .threads=1};
	using HM = SimpleHashMap<T, V, s, Hash<T, 0, l>>;
	HM *hm = new HM{};

	static constexpr ConfigEnumHashMap config{k+l, p, l, 1, epsilon};
	T *data = (T *)cryptanalysislib::aligned_alloc(1024, roundToAligned<1024>(sizeof(T) * (k+l)));
	for (uint32_t i = 0; i < k+l; ++i) {
		data[i] = fastrandombytes_uint64() & ((1u << l) - 1u);
	}

	EnumHashMap<config, HM> mitm{data, hm};
	mitm.print();
	for (uint32_t simd = 0; simd < 2; ++simd) {
		mitm.step(0, 0, simd);

		for (size_t i = 0; i < (k+l)/2 + epsilon; i++){
			HM::load_type load = 0;
			size_t k = hm->find(data[i], load);
			EXPECT_GT(load, 0);
			if constexpr (p == 1) {
				EXPECT_EQ(k, data[i] * s.bucketsize);
			}
		}
	}

	free(data);
	delete hm;
}

TEST(EnumHashMap, enumeration_index) {
	constexpr uint32_t p = 2, l = 5;
	using T = MinLogTypeTemplate<l, 32>;
	using V = CollisionType<T, uint32_t, 1>;

	constexpr static SimpleHashMapConfig s{.bucketsize=10u, .nrbuckets=1u<<l, .threads=1};
	using HM = SimpleHashMap<T, V, s, Hash<T, 0, l>>;
	HM *hm = new HM{};

	static constexpr ConfigEnumHashMap config{k+l, p, l, 1, true};
	T *data = (T *)cryptanalysislib::aligned_alloc(1024, roundToAligned<1024>(sizeof(T) * (k+l)));
	for (uint32_t i = 0; i < k+l; ++i) {
		data[i] = fastrandombytes_uint64() & ((1u << l) - 1u);
	}

	EnumHashMap<config, HM> mitm{data, hm};
	mitm.print();
	for (uint32_t simd = 0; simd < 2; ++simd) {
		mitm.step(0, 0, simd);

		for (size_t i = 0; i < (k+l)/2; i++){
			HM::load_type load = 0;
			size_t k = hm->find(data[i], load);
			EXPECT_GT(load, 0);
			EXPECT_EQ(k, data[i] * s.bucketsize);
		}
	}

	free(data);
	delete hm;
}

TEST(EnumHashMap, multithreaded_enumeration) {
	constexpr uint32_t p = 2, l = 17, threads=2;
	using T = MinLogTypeTemplate<l, 32>;
	using V = CollisionType<T, uint16_t, p>;

	constexpr static SimpleHashMapConfig s{.bucketsize=10u, .nrbuckets=1u<<l, .threads=threads};
	using HM = SimpleHashMap<T, V, s, Hash<T, 0, l>>;
	HM *hm = new HM{};
	hm->print();

	static constexpr ConfigEnumHashMap config{k+l, p, l, threads};
	T *data = (T *)cryptanalysislib::aligned_alloc(1024, roundToAligned<1024>(sizeof(T) * (k+l)));
	for (uint32_t i = 0; i < k+l; ++i) {
		data[i] = fastrandombytes_uint64() & ((1u << l) - 1u);
	}

	auto *mitm = new EnumHashMap<config, HM>{data, hm};
	for (uint32_t simd = 0; simd < 2; ++simd) {
		#pragma omp parallel default(none) shared(mitm, hm) num_threads(threads)
		{
			const uint32_t tid = omp_get_thread_num();
			mitm->step(0, tid, false);
		}

		for (size_t i = 0; i < (k+l)/2; i++){
			HM::load_type load = 0;
			size_t k = hm->find(data[i], load);
			// NOTE: only valid if p == 1
			if constexpr (p == 1) {
				EXPECT_GT(load, 0);
				EXPECT_EQ(k, data[i] * s.bucketsize);
			}
		}

	}

	free(data);
	delete hm;
	delete mitm;
}

TEST(CollisionHashMap, enumeration) {
	constexpr uint32_t p = 2, l = 5;
	using T = MinLogTypeTemplate<l, 32>;
	using V = CollisionType<T, uint16_t, p>;

	constexpr static SimpleHashMapConfig s{.bucketsize=10u, .nrbuckets=1u<<l, .threads=1};
	using HM = SimpleHashMap<T, V, s, Hash<T, 0, l>>;
	using HM_DataType = HM::data_type;
	using HM_DataType_IndexType = HM_DataType::index_type;
	HM *hm = new HM{};

	static constexpr ConfigEnumHashMap config{k+l, p, l, 1};
	T *data = (T *)cryptanalysislib::aligned_alloc(1024, roundToAligned<1024>(sizeof(T) * (k+l)));
	for (uint32_t i = 0; i < k+l; ++i) {
		data[i] = fastrandombytes_uint64() & ((1u << l) - 1u);
	}

	EnumHashMap<config, HM> base{data, hm};
	base.step(0);

	CollisionHashMap<config, HM> mitm{data, hm};
	mitm.print();


	auto f = [&](const T lp1, const T lp2,
				 HM_DataType_IndexType *index1,
				 HM_DataType_IndexType *index2,
	             const uint32_t nr_cols) {
		(void) nr_cols;
	    bool found = false;
		const T a = lp1 ^ lp2;
		for (size_t i = 0; i < k + l; ++i) {
			  if (a == data[i]) {
				  found = true;
				  break;
			  }
		}

		if (!found) {
			  std::cout << a << " " << index1[0] << "," << index2[0] << std::endl;
		}


	  EXPECT_EQ(found, true);
	};

	mitm.step(0, f, 0, false);

	for (size_t i = 0; i < (k+l)/2; i++){
		HM::load_type load = 0;
		const size_t m = hm->find(data[i], load);
		EXPECT_GT(load, 0);
		EXPECT_EQ(m, data[i] * s.bucketsize);
	}

	free(data);
	delete hm;
}

TEST(CollisionHashMap, enumerationSIMD) {
	constexpr uint32_t p = 1, l = 4;
	using T = MinLogTypeTemplate<l, 32>;
	using V = CollisionType<T, uint32_t, 1>;

	constexpr static SimpleHashMapConfig s{.bucketsize=2u, .nrbuckets=1u<<l, .threads=1};
	using HM = SimpleHashMap<T, V, s, Hash<T, 0, l>>;
	using HM_DataType = HM::data_type;
	using HM_DataType_IndexType = HM_DataType::index_type;
	HM *hm = new HM{};

	static constexpr ConfigEnumHashMap config{k+l, p, l, 1, true};
	T *data = (T *)cryptanalysislib::aligned_alloc(1024, roundToAligned<1024>(sizeof(T) * (k+l)));
	for (uint32_t i = 0; i < k+l; ++i) {
		data[i] = fastrandombytes_uint64() & ((1u << l) - 1u);
	}

	EnumHashMap<config, HM> base{data, hm};
	base.step(0);

	CollisionHashMap<config, HM> mitm{data, hm};
	mitm.print();

	auto f = [&](const T lp1, const T lp2,
	             HM_DataType_IndexType *index1,
	             HM_DataType_IndexType *index2,
	             const uint32_t nr_cols) __attribute__((always_inline)) {
		bool found = true;
		(void) lp1;
		(void) lp2;
		for (uint32_t i = 0; i < nr_cols; ++i) {
			found &= data[index1[i]] == data[index2[i]];
		}
		EXPECT_EQ(found, true);
	};

	mitm.step(0, f, 0, true);

	for (size_t i = 0; i < (k+l)/2; i++){
		HM::load_type load = 0;
		const size_t m = hm->find(data[i], load);
		EXPECT_GT(load, 0);
		EXPECT_EQ(m, data[i] * s.bucketsize);
	}

	free(data);
	delete hm;
}

//TEST(CollisionHashMapD2, enumeration) {
//	constexpr uint32_t p = 1, l1=4, l2=6, l=l1+l2;
//	using T = MinLogTypeTemplate<l, 32>;
//	using V1 = CollisionType<T, 1>;
//	using V2 = CollisionType<T, 2>;
//
//	constexpr static SimpleHashMapConfig s1{.bucketsize=8u, .nrbuckets=1u<<l1, .threads=1};
//	constexpr static SimpleHashMapConfig s2{.bucketsize=16u, .nrbuckets=1u<<l2, .threads=1};
//	using HM1 = SimpleHashMap<T, V1, s1, Hash<T, 0, l1>>;
//	using HM2 = SimpleHashMap<T, V2, s2, Hash<T, 0, l2>>;
//	HM1 *hm1 = new HM1{};
//	HM2 *hm2 = new HM2{};
//
//	using HM_DataType = HM2::data_type;
//	using HM_DataType_IndexType = HM_DataType::index_type;
//
//	static constexpr ConfigEnumHashMap configD1{k+l, p, l1, 1};
//	static constexpr ConfigEnumHashMapD2 configD2{configD1, l2};
//	T *data = (T *)cryptanalysislib::aligned_alloc(1024, roundToAligned<1024>(sizeof(T) * (k+l)));
//	for (uint32_t i = 0; i < k+l; ++i) {
//		data[i] = fastrandombytes_uint64() & ((1u << l) - 1u);
//	}
//
//	EnumHashMap<configD1, HM1> mitmD1{data, hm1};
//	CollisionHashMapD2<configD2, HM1, HM2> mitmD2{data, hm1, hm2};
//
//	auto f = [](const T k, HM_DataType_IndexType *rows){
//		std::cout << k << std::endl;
//	};
//	mitmD1.step(0, 0, false);
//	mitmD2.step(0, f, false);
//
//	for (size_t i = 0; i < (k+l)/2; i++){
//		//size_t load = 0;
//		//size_t k = hm->find(data[i], load);
//		//EXPECT_GT(load, 0);
//		//EXPECT_EQ(k, data[i] * s.bucketsize);
//	}
//
//
//	free(data);
//	delete hm1;
//	delete hm2;
//}
//

TEST(CollisionHashMapD3, enumeration) {
	constexpr uint32_t d=3,p = 2, l1=3, l2=4, l3=5, l=l1+l2+l3, threads=1;
	using T = MinLogTypeTemplate<l, 32>;
	using V1 = CollisionType<T, uint16_t, 1*p>;
	using V2 = CollisionType<T, uint16_t, 2*p>;
	using V3 = CollisionType<T, uint16_t, 4*p>;

	constexpr static SimpleHashMapConfig s1{.bucketsize=16u, .nrbuckets=1u<<l1, .threads=threads};
	constexpr static SimpleHashMapConfig s2{.bucketsize=12u, .nrbuckets=1u<<l2, .threads=threads};
	constexpr static SimpleHashMapConfig s3{.bucketsize=8u,  .nrbuckets=1u<<l3, .threads=threads};
	using HM1 = SimpleHashMap<T, V1, s1, Hash<T, 0, l1>>;
	using HM2 = SimpleHashMap<T, V2, s2, Hash<T, 0, l2>>;
	using HM3 = SimpleHashMap<T, V3, s3, Hash<T, 0, l3>>;
	HM1 *hm1 = new HM1{};
	HM2 *hm2 = new HM2{};
	HM3 *hm3 = new HM3{};

	using HM_DataType = HM3::data_type;
	using HM_DataType_IndexType = HM_DataType::index_type;

	static constexpr ConfigEnumHashMap configD1{k+l, p, l1, threads};
	// static constexpr ConfigEnumHashMapD2 configD2{configD1, l2};
	static constexpr ConfigEnumHashMapD<3> configD3{configD1, {l1, l2, l3}};
	T *data = (T *)cryptanalysislib::aligned_alloc(1024, roundToAligned<1024>(sizeof(T) * (k+l)));
	for (uint32_t i = 0; i < k+l; ++i) { data[i] = fastrandombytes_uint64() & ((1u << l) - 1u);}


	const uint32_t syndrome = 0xff;
	EnumHashMap<configD1, HM1> 				enumD1{data, hm1};
	//CollisionHashMap<configD1, HM1> 		mitmD1{data, hm1};
	//CollisionHashMapD2<configD2, HM1, HM2> 	mitmD2{data, hm1, hm2};
	CollisionHashMapD<d, configD3, HM1, HM2, HM3> mitmD3{
	        data, std::ref(*hm1), std::ref(*hm2), std::ref(*hm3),
	};
	mitmD3.print();


	auto f1 = [&](const T a, uint16_t *index){
	  	// ASSERT(enumD1.check_hashmap2(iT1, index, 2*p, l1));
	  	hm2->insert(a>>l1, V2::create(a, index));
	};

	auto f2 = [&](const T a, uint16_t *index){
	  	// ASSERT(enumD1.check_hashmap2(iT2, index, 4*p, l1 + l2));
	  	hm3->insert(a>>l1, V3::create(a, index));
	};

	auto f3 = [&](const T a, uint16_t *index){
		(void)index;
	  	// ASSERT(enumD1.check_hashmap2(iT2, index, 8*p, l));
	  	std::cout << a << std::endl;
	};

	const uint32_t tid = 0;
	enumD1.step(0, tid, false);
	//mitmD1.step(iT1, f1, tid, false);
	//mitmD2.step(iT2, f2, tid, false);
	mitmD3.step(syndrome, tid, false, f1, f2, f3);

	free(data);
	delete hm1;
	delete hm2;
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	srand(time(NULL));
	random_seed(rand());
	return RUN_ALL_TESTS();
}
