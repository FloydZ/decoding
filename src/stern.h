#ifndef DECODING_STERN_H
#define DECODING_STERN_H

#include <cstdint>

#include "matrix/binary_matrix.h"
#include "container/hashmap.h"
#include "popcount/popcount.h"
#include "math/bc.h"
#include "alloc/alloc.h"
#include "hash/simple.h"
#include "mitm.h"
#include "isd.h"

using namespace cryptanalysislib;


struct ConfigStern : public ConfigISD {
public:
	// number of elements per bucket in the hashmap
	const uint32_t HM_bucketsize = bc((k+l)/2, p) >> l;

	// number of elements which can be contained in the last list.
	// this is only a tmp storage, after this many elements were added
	// the implementation automatically flushes it.
	const uint32_t final_list_size = 1024;


	// returns the expected number of iterations
	[[nodiscard]] constexpr uint64_t compute_loops() const noexcept {
#ifdef EXPECTED_PERMUTATIONS
		return EXPECTED_PERMUTATIONS;
#else
		return bc(n, w)/( bc(n-k-l, w-2*p) * bc((k+l)/2, p) * bc((k+l)/2, p));
#endif
	}

	// print some information about the configuration
	void print() const noexcept {
		ConfigISD::print();
		std::cout << "{ \"bucketsize\": " << HM_bucketsize
				  << ", \"final_list_size\": " << final_list_size
				  << " }" << std::endl;
	}
};


template<const ConfigISD &isd, const ConfigStern &config>
class Stern : public ISDInstance<uint64_t, isd> {
public:
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t k = config.k;
	constexpr static uint32_t w = config.w;
	constexpr static uint32_t p = config.p;
	constexpr static uint32_t l = config.l;
	constexpr static uint32_t q = config.q;

	static_assert(p > 0);
	static_assert(l > 0);
	static_assert(config.HM_bucketsize > 0);

	using ISD = ISDInstance<uint64_t, isd>;
	using Error 		= typename ISD::Error;
	using Label 		= typename ISD::Label;
	using limb_type 	= typename ISD::limb_type;
	using ISD::A,ISD::H,ISD::wA,ISD::wAT,ISD::HT,ISD::s,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::ghz,ISD::expected_loops;
	using ISD::cycles,ISD::periodic_print,ISD::packed,ISD::simd;


	// base datatype of the hashmap
	using l_type   = typename ISD::l_type;

	constexpr static uint32_t kl_half = (k+l)/2;

	/// list entries of the current permutation
	/// if a solution was found.
	size_t solutions[2*p] = {0};

	/// internal arrays pointing to
	l_type *pHT, *pHTr, *lHT;

	/// Maximal list size of the final list
	constexpr static size_t final_list_max_size = config.final_list_size;

	/// Current load of the final list
	size_t final_list_current_size = 0;

	constexpr static uint32_t NR_HT_LIMBS 	= (n-k+sizeof(limb_type)*8 - 1u)/((sizeof(limb_type)*8));
	constexpr static uint32_t NR_HT_T_LIMBS = (n-k+(sizeof(l_type)*8) - 1u)/(sizeof(l_type)*8);
	constexpr static size_t baselist_enumeration_length = (k+l)/2u;

	// hashmap stuff
	constexpr static bool SternCollType = false;
	using keyType = LogTypeTemplate<l>;
	using V1 = typename std::conditional<SternCollType,
	    SternCollisionType<l_type, uint16_t, p>,
		CollisionType<l_type, uint16_t, p>
	>::type;

	constexpr static uint64_t nrbuckets = 1u << l;
	constexpr static uint64_t bucketsize = config.HM_bucketsize;//bc(n, p) >> l;

	constexpr static SimpleHashMapConfig simpleHashMapConfig{
			bucketsize, nrbuckets, config.threads
	};

	using HM = SimpleHashMap<keyType, V1, simpleHashMapConfig, Hash<l_type, 0, l>>;
	HM *hm;

	constexpr static ConfigEnumHashMap configEnum{config};
	EnumHashMap<configEnum, HM> *bEnum;
	CollisionHashMap<configEnum, HM> *cEnum;

	/// NOTE: that we increase the size with about one bucket size to make sure that we can fully insert a bucket
	constexpr static size_t final_list_real_max_size = (final_list_max_size + bucketsize*bucketsize) + ((8u - ((final_list_max_size + bucketsize*bucketsize)%8u))%8u);
	alignas(256) std::array<uint16_t[p], final_list_real_max_size> final_list_left;
	alignas(256) std::array<uint16_t[p], final_list_real_max_size> final_list_right;

	Stern() noexcept {
		expected_loops = config.compute_loops();

		// TODO move to ISD
		constexpr size_t size_lHT = roundToAligned<1024>(sizeof(l_type) * (k+l));
		lHT =(l_type *)cryptanalysislib::aligned_alloc(1024, size_lHT);
		ASSERT(lHT);

		constexpr size_t size_pHT = roundToAligned<1024>(sizeof(l_type) * (k+l) * NR_HT_T_LIMBS);
		pHT =(l_type *)cryptanalysislib::aligned_alloc(1024, size_pHT);
		ASSERT(pHT);
		pHTr = pHT + baselist_enumeration_length *NR_HT_T_LIMBS;

		hm = new HM{};
		ASSERT(hm);

		bEnum = new EnumHashMap<configEnum, HM>{lHT, hm};
		cEnum = new CollisionHashMap<configEnum, HM>{lHT, hm};
	}

	~Stern() noexcept {
		free(lHT);
		free(pHT);

		delete hm;
		delete bEnum;
		delete cEnum;
	}


	/// takes the indices from `final_list` and checks if they are a full 
	/// collision
	/// NOTE: without AVX
	bool compute_finale_list() noexcept {
		if constexpr (simd) {
			return compute_finale_list_simd();
		}

		// go through the list
		for (size_t cindex = 0; cindex < final_list_current_size; cindex++) {
			const uint16_t *left = final_list_left[cindex];
			const uint16_t *right = final_list_right[cindex];

			auto climb = ws.ptr(0);
			for (uint16_t j = 0; j < p; j++) {
				ASSERT(left[j] < kl_half);
				ASSERT(right[j] < (k+l));

				climb ^= (HT[left[j]][0] ^ HT[right[j]][0]);
			}

			// this is only correct if not STERNIM. In the IM case we
			// match on zeros on a different window
			// ASSERT((climb & ((1u << l) - 1u)) == 0);
			uint32_t wt = cryptanalysislib::popcount::popcount(climb);

			// early exit
			if (likely(wt > w-(2*p))) {
				continue;
			}

			/// compute the remaining limbs
			for (uint32_t i = 1; i < NR_HT_LIMBS; i++) {
				climb = ws.ptr(i);
				for (uint16_t j = 0; j < p; j++) {
					ASSERT(left[j] < kl_half);
					ASSERT(right[j] < (k+l));

					climb ^= HT[left[j]][i] ^ HT[right[j]][i];
				}

				wt += cryptanalysislib::popcount::popcount(climb);
			}

			if ((wt <= w - (2*p)) && not_found) {
				not_found = false;
				cycles = cpucycles() - cycles;
				for (uint16_t j = 0; j < p; ++j) {
					solutions[j*p + 0] = left[j];
					solutions[j*p + 1] = right[j];
				}

				return true;
			}
		}

		final_list_current_size = 0;
		return false;
	}

	/// checks for collisions in the final list.
	/// NOTE: using AVX2, therefore 8 elements are checked simultaneously
	bool compute_finale_list_simd() noexcept {
		uint32x8_t rows1[p], rows2[p];
		const uint32x8_t filter_mask = uint32x8_t::set1(w - 2*p + 1);
		const uint32x8_t mul_mask    = uint32x8_t::set1(NR_HT_T_LIMBS);

		const uint32x8_t *final_list_left256  = (uint32x8_t *)final_list_left.data();
		const uint32x8_t *final_list_right256 = (uint32x8_t *)final_list_right.data();

		for (size_t cindex = 0; cindex < final_list_current_size/8; cindex++) {
			ASSERT(cindex < final_list_left.size());

			const uint32x8_t left  = uint32x8_t::load(final_list_left256  + cindex);
			const uint32x8_t right = uint32x8_t::load(final_list_right256 + cindex);

			biject_simd<baselist_enumeration_length, p>(left, rows1);
			biject_simd<baselist_enumeration_length, p>(right, rows2);

			// ignore special case for lowest limb
			uint32x8_t wt{};
			int wt_ = 0;

			#pragma unroll
			for (uint16_t i = 0; i < p; ++i) {
				// NOTE: this is important, but stupid.
				// we need to multiply the offset of the rows, by the number of limbs
				// in each row.
				rows1[i] = rows1[i] * mul_mask;
				rows2[i] = rows2[i] * mul_mask;
			}

			// #pragma unroll
			for (uint32_t i = 0; i < NR_HT_T_LIMBS; i++) {
				uint32x8_t climb = uint32x8_t::set1(ws.ptr(i));

				const l_type *base_pHT = pHT + i;
				const l_type *base_pHTr = pHTr + i;

				// #pragma unroll
				for (uint16_t j = 0; j < p; j++) {
					const uint32x8_t t11 = uint32x8_t::template gather <sizeof(l_type)>(base_pHT, rows1[j]);
					const uint32x8_t t21 = uint32x8_t::template gather <sizeof(l_type)>(base_pHTr, rows2[j]);
					climb = climb ^ t11 ^ t21;
				}

				// sanity check
				if(i == 0) {
					ASSERT(((climb.v32[0]) & ((1u << l) - 1u)) == 0);
				}

				// popcount
				const uint32x8_t wtt = uint32x8_t::popcnt(climb);
				wt = wt + wtt;
				wt_ = filter_mask > wt;

				// if the mask `wt_` is zero, we can do an early exit, because the weight
				// is already too big, for all elements.
				if (wt_ == 0) {
					break;
				}
			}

			// if we pass this check, we found a global solution
			if (unlikely((wt_ != 0) && not_found)) {
				not_found = false;

				const uint32_t pos = __builtin_ctz(wt_);
				for (uint16_t j = 0; j < p; ++j) {
					solutions[j*p + 0] = rows1[j].v32[pos]/NR_HT_T_LIMBS;
					solutions[j*p + 1] = rows1[j].v32[pos]/NR_HT_T_LIMBS + baselist_enumeration_length;
				}

				return true;
			}
		}

		// reset the list counter
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
		for (uint32_t i = 0; i < 2*p; ++i) {
			ASSERT(solutions[i] < k+l);
			Label::add(tmp.ptr(), tmp.ptr(), HT.row(solutions[i]));
		}

		for (uint32_t i = 0; i < n-k; ++i) {
			const auto bit = tmp.get(i);
			tmpe.set(bit, 0, n-k-1-i);
		}

		for (uint32_t i = 0; i < 2*p; ++i) {
			ASSERT(n-k-l + solutions[i] < n);
			tmpe.set(1, 0, n-k-l + solutions[i]);
		}

		// apply back permutation.
		for (uint32_t i = 0; i < n; ++i) {
			e.set(tmpe.get(0, i), 0, P.values[i]);
		}
	}

	// runs: optimized stern
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		loops = 0;
		not_found = true;
		cycles = cpucycles();

		while (not_found && loops < isd.loops) {
			const uint32_t tid = 0;
			ISD::step();
			ISD::template extract_lHT<l_type>(lHT);
			ISD::extract_pHT(pHT);

			auto f = [&, this](const l_type a1, const l_type a2,
			                   const uint16_t *index1, const uint16_t *index2,
			                   const uint32_t nr_cols) __attribute__((always_inline)) -> bool {
				(void)nr_cols;
				if constexpr (!SternCollType) {
					ASSERT(bEnum->check_hashmap2(syndrome, index1, 2 * p, l));
					const l_type a = a1 ^ a2;
					ASSERT(a == 0);
				}

				for (uint32_t i = 0; i < p; i++) {
					final_list_left[final_list_current_size][i]  = index1[i];
					final_list_right[final_list_current_size][i] = index2[i];
				}

				final_list_current_size += 1;

				if (final_list_current_size >= final_list_max_size) {
					compute_finale_list();
				}
				return false;
			};

			bEnum->step(0, tid, simd);
			cEnum->step(syndrome, f, tid, simd);
			compute_finale_list();
		}

		// compute the final solution
		reconstruct();
		return loops;
	}

	/// important. Dont rename it
	void info() const noexcept {
		const size_t load = hm->load();
		const double load_per_bucket = double(load) / nrbuckets;
		std::cout << "{"
		   		  << " \"nrbuckets\": " << nrbuckets
		          << ", \"load/bucket\": " << load_per_bucket
		          << ", \"final_list_size\": " << final_list_max_size
		          << ", \"NR_HT_LIMBS\": " << NR_HT_LIMBS
		          << ", \"NR_HT_T_LIMBS\": " << NR_HT_T_LIMBS
		          << " }" << std::endl;
	}
};
#endif//DECODING_STERN_H
