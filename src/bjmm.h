#ifndef DECODING_BJMM_H
#define DECODING_BJMM_H

#include "matrix/binary_matrix.h"
#include "container/hashmap.h"
#include "popcount/popcount.h"
#include "math/bc.h"
#include "alloc/alloc.h"
#include "hash/simple.h"
#include "mitm.h"

#include "isd.h"

using namespace cryptanalysislib;


struct ConfigBJMM: public ConfigISD {
public:
	const uint32_t l1 = 0;

	// number of elements per bucket in the hashmap
	const uint32_t HM1_bucketsize = 0;
	const uint32_t HM2_bucketsize = 0;

	// number of elements which can be contained in the last list.
	// this is only a tmp storage, after this many elements were added
	// the implementation automatically flushes it.
	const uint32_t final_list_size = 1024;

	const uint32_t intermediate_loops = 1;


	// returns the expected number of iterations
	// NOTE: m4ri is not supported.
	[[nodiscard]] consteval uint64_t compute_loops() const noexcept {
#ifdef EXPECTED_PERMUTATIONS
		return EXPECTED_PERMUTATIONS;
#else
		return bc(n, w)/( bc(n-k-l, w-2*p) * bc((k+l)/2, p) * bc((k+l)/2, p));
#endif
	}

	// print some information about the configuration
	void print() const noexcept {
		ConfigISD::print();
		std::cout << "{ \"HM1_bucketsize\": " << HM1_bucketsize
				  << ", \"HM2_bucketsize\": " << HM2_bucketsize
		          << ", \"l1\": " << l1
				  << ", \"final_list_size\": " << final_list_size
				  << " }" << std::endl;
	}
};


template<const ConfigISD &isd, const ConfigBJMM &config>
class BJMM : public ISDInstance<uint64_t, isd> {
public:
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t k = config.k;
	constexpr static uint32_t w = config.w;
	constexpr static uint32_t p = config.p; // NOTE: from now on p is the baselist p not the full p.
	constexpr static uint32_t l = config.l;
	constexpr static uint32_t l1 = config.l1;
	constexpr static uint32_t l2 = l - config.l1;
	constexpr static uint32_t q = config.q;
	constexpr static uint32_t gaus_c = config.c;

	using ISD = ISDInstance<uint64_t, isd>;
	using PCMatrixOrg 	= typename ISD::PCMatrixOrg;
	using PCMatrixOrg_T = typename ISD::PCMatrixOrg_T;
	using PCMatrix 		= typename ISD::PCMatrix;
	using PCMatrix_T 	= typename ISD::PCMatrix_T;
	using PCSubMatrix 	= typename ISD::PCSubMatrix;
	using PCSubMatrix_T = typename ISD::PCSubMatrix_T;
	using Syndrome 		= typename ISD::Syndrome;
	using Error 		= typename ISD::Error;
	using Label 		= typename ISD::Label;
	using ISD::A,ISD::H,ISD::wA,ISD::wAT,ISD::HT,ISD::s,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::ghz;
	using ISD::cycles,ISD::periodic_print,ISD::packed,ISD::simd;

	using l_type   = typename ISD::l_type;
	using keyType = LogTypeTemplate<l>;
	// using valueType = TypeTemplate<bc(n, p)>[1];

	// base datatype of the hashmap
	using listType = uint32_t; 	// DO NOT CHANGE

	constexpr static uint32_t kl_half = (k+l)/2;
	constexpr static uint32_t wt_limit =  w - (4*p);

	/// list entries of the current permutation
	/// if a solution was found.
	listType solutions[4*p] = {0};

	/// internal arrays pointing to
	l_type *pHT, *pHTr, *lHT, *lHTr;

	/// Maximal list size of the final list
	constexpr static size_t final_list_max_size = config.final_list_size;

	/// Current load of the final list
	size_t final_list_current_size = 0;

	/// TODO generalize
	constexpr static uint32_t NR_HT_LIMBS 	= (n-k+63)/64;
	constexpr static uint32_t NR_HT_T_LIMBS = (n-k+31)/32;
	constexpr static size_t baselist_enumeration_length = (k+l)/2u;

	// hashmap stuff
	constexpr static uint64_t HM1_nrbuckets = 1u << l1;
	constexpr static uint64_t HM1_bucketsize = bc(n, p) >> l1; // TODO I think mem leak
	constexpr static uint64_t HM2_nrbuckets = 1u << l2;
	constexpr static uint64_t HM2_bucketsize = (HM1_bucketsize*HM1_bucketsize) >> l2;

	using V1 = CollisionType<l_type, uint16_t, 1*p>;
	using V2 = CollisionType<l_type, uint16_t, 2*p>;
	constexpr static SimpleHashMapConfig simpleHashMapConfig{
			HM1_bucketsize, HM1_nrbuckets, config.threads
	};
	constexpr static SimpleHashMapConfig simpleHashMapConfigD2 {
			HM2_bucketsize, HM2_nrbuckets, config.threads
	};
	using HM1 = SimpleHashMap<keyType, V1, simpleHashMapConfig, Hash<l_type, 0, l1>>;
	using HM2 = SimpleHashMap<keyType, V2, simpleHashMapConfigD2, Hash<l_type, 0, l2>>;
	HM1 *hm1; HM2 *hm2;

	constexpr static ConfigEnumHashMap configEnum{config};
	constexpr static ConfigEnumHashMapD2 configEnumD2{config, l1, l2};
	EnumHashMap<configEnum, HM1> *bEnum;
	CollisionHashMap<configEnum, HM1> *cEnum;
	CollisionHashMapD2<configEnumD2, HM1, HM2> *cEnumD2;

	/// NOTE: that we increase the size with about one bucket size to make sure that we can fully insert a bucket
	// TODO think about the formular
	constexpr static size_t final_list_real_max_size = (final_list_max_size + HM2_bucketsize*HM2_bucketsize) + ((8u - ((final_list_max_size + HM2_bucketsize*HM2_bucketsize)%8u))%8u);
	alignas(256) std::array<uint16_t[4], final_list_real_max_size> final_list;

	BJMM() noexcept {
		constexpr size_t size_lHT = roundToAligned<1024>(sizeof(l_type) * (k+l));
		lHT =(l_type *)cryptanalysislib::aligned_alloc(1024, size_lHT);
		ASSERT(lHT);
		lHTr = lHT + baselist_enumeration_length; // pointer in the middle

		constexpr size_t size_pHT = roundToAligned<1024>(sizeof(l_type) * (k+l) * NR_HT_T_LIMBS);
		pHT =(l_type *)cryptanalysislib::aligned_alloc(1024, size_pHT);
		ASSERT(pHT);
		pHTr = pHT + baselist_enumeration_length *NR_HT_T_LIMBS;

		hm1 = new HM1{};
		ASSERT(hm1);
		hm2 = new HM2{};
		ASSERT(hm2);

		bEnum = new EnumHashMap<configEnum, HM1>{lHT, hm1};
		cEnum = new CollisionHashMap<configEnum, HM1>{lHT, hm1};
		cEnumD2 = new CollisionHashMapD2<configEnumD2, HM1, HM2>{lHT, hm1, hm2};
	}

	~BJMM() noexcept {
		free(lHT);
		free(pHT);

		delete hm1;
		delete hm2;
		delete bEnum;
		delete cEnum;
		delete cEnumD2;
	}

	/// takes the indices from `final_list` and checks if they are a full
	/// collision
	/// NOTE: without AVX
	/// TODO: simd
	bool compute_finale_list() noexcept {
		// go through the list
		for (size_t cindex = 0; cindex < final_list_current_size; cindex++) {
			uint16_t *rows = final_list[cindex];

			auto climb = ws.ptr(0);
			for (uint32_t j = 0; j < 4*p; j++) {
				ASSERT(rows[j] < (k+l));
				climb ^= HT[rows[j]][0];
			}

			// match on zeros on a different window
			ASSERT((climb & ((1u << l) - 1u)) == 0);
			uint32_t wt = cryptanalysislib::popcount::popcount(climb);

			// early exit
			if (likely(wt > wt_limit)) { continue; }

			/// compute the remaining limbs
			for (uint32_t i = 1; (i < NR_HT_LIMBS) && (wt <= wt_limit); i++) {
				climb = ws.ptr(i);

				#pragma unroll
				for (uint32_t j = 0; j < 4*p; j++) {
					ASSERT(rows[j] < k + l);
					climb ^= HT[rows[j]][i];
				}

				wt += cryptanalysislib::popcount::popcount(climb);
			}

			if (unlikely((wt <= wt_limit) && not_found)) {
				not_found = false;
				cycles = cpucycles() - cycles;
				for (uint32_t j = 0; j < 4*p; ++j) {
					solutions[j] = rows[j];
				}

				return true;
			}
		}

		final_list_current_size = 0;
		return false;
	}


		/// reconstruct the final solution
	void __attribute__ ((noinline))
	reconstruct() noexcept {
		Error tmpe, tmpe2;
		Label tmp;
		// copy in the syndrome
		tmp = ws;
		for (uint32_t i = 0; i < 4*p; ++i) {
			ASSERT(solutions[i] < k+l);
			Label::add(tmp.ptr(), tmp.ptr(), HT.row(solutions[i]));
		}

		for (uint32_t i = 0; i < n-k; ++i) {
			const auto bit = tmp.get(i);
			tmpe.set(bit, 0, n-k-1-i);
		}

		for (uint32_t i = 0; i < 4*p; ++i) {
			ASSERT(n-k-l + solutions[i] < n);
			tmpe.set(1, 0, n-k-l + solutions[i]);
		}

		// apply back permutation.
		for (uint32_t i = 0; i < n; ++i) {
			e.set(tmpe.get(0, i), 0, P.values[i]);
		}
	}

	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();

		while (not_found && loops < isd.loops) {
			const uint32_t tid = 0;
			ISD::step();
			ISD::extract_lHT(lHT);
			ISD::extract_pHT(pHT);


			/// TODO describe the parameters
			/// \param a1
			/// \param a2
			/// \param index1
			/// \param index2
			/// \param nr_cols
			auto f1 = [&, this](const l_type a1, const l_type a2,
			                    const uint16_t *index1, const uint16_t *index2,
			                    const uint32_t nr_cols) __attribute__((always_inline)) {
				// NOTE: iT1 curently unknown in the global scope
			  	// ASSERT(bEnum->check_hashmap2(iT1, index1, 2*p, l1));
				(void) index2;
				(void) nr_cols;
				const l_type a = a1 ^ a2;
				hm2->insert(a>>l1, V2::create(a, index1));
			};

			auto f2 = [&, this](const l_type a1, const l_type a2,
								const uint16_t *index1, const uint16_t *index2,
								const uint32_t nr_cols) __attribute__((always_inline)) {
			  	ASSERT(bEnum->check_hashmap2(syndrome, index1, 4*p, l));
				(void) index2;
				(void) nr_cols;
				(void) a1;
				(void) a2;

				for (uint32_t i = 0; i < 4; ++i) {
					final_list[final_list_current_size][i] = index1[i];
				}
				final_list_current_size += 1;

				if (final_list_current_size >= final_list_max_size) {
					  compute_finale_list();
				}

				return false;
			};

			constexpr bool simd = false;
			bEnum->step(0, tid, simd);

			l_type iT1 = 0;
			for (uint32_t loops = 0; loops < config.intermediate_loops; ++loops, iT1 += 1) {
				// generate a random intermediate target, except for the fact
				// that we are not completely randomly choosing it. Because
				// there is a good chance that we are choosing two distinct
				// targets s.t. t1 xor s = t2. This will not help the algorithm
				// to recover the error
				while ((iT1 ^ syndrome) < iT1) {
					iT1 += 1;
				}

				const l_type iT2 = iT1 ^ syndrome;

				cEnum->step(iT1, f1, tid, simd);
				cEnumD2->step(iT2, f2, tid, simd);
				compute_finale_list();
				hm2->clear(tid);
			}
		}

		// compute the final solution
		reconstruct();
		return loops;
	}

	// print one time information
	void info() const noexcept {
		const size_t load1 = hm1->load();
		const size_t load2 = hm2->load();
		const double load_per_bucket1 = double(load1) / double(HM1_nrbuckets);
		const double load_per_bucket2 = double(load2) / double(HM2_nrbuckets);
		std::cout << "{ \"HM1_nrbuckets\": " << HM1_nrbuckets
				  << ", \"HM2_nrbuckets\": " << HM2_nrbuckets
				  << ", \"final_list_size\": " << final_list_max_size
		          << ", \"NR_HT_LIMBS\": " << NR_HT_LIMBS
				  << ", \"NR_HT_T_LIMBS\": " << NR_HT_T_LIMBS
				  << ", \"load_per_bucket1\": " << load_per_bucket1
				  << ", \"load_per_bucket2\": " << load_per_bucket2
				  << std::endl;
	}
};

#endif //SMALLSECRETLWE_BJMM_H
