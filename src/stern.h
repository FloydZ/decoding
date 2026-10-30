#ifndef DECODING_STERN_H
#define DECODING_STERN_H

#include <cstdint>

#include "matrix/matrix.h"
#include "container/hashmap.h"
#include "algorithm/bits/popcount.h"
#include "math/bc.h"
#include "alloc/alloc.h"
#include "hash/simple.h"
#include "mitm.h"
#include "isd.h"
#include "random.h"

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
	using ISD::lHT,ISD::pHT;


	// base datatype of the hashmap
	using l_type   = typename ISD::l_type;

	constexpr static uint32_t kl_half = (k+l)/2;

	/// list entries of the current permutation
	/// if a solution was found.
	size_t solutions[2*p] = {0};

	/// internal arrays pointing to
	l_type *pHTr;

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

	using HM = SimpleHashMap<keyType, V1, simpleHashMapConfig, Hash<l_type, 0, l, 2>>;
	HM *hm;

	constexpr static ConfigEnumHashMap configEnum{config};
	EnumHashMap<configEnum, HM> *bEnum;
	CollisionHashMap<configEnum, HM> *cEnum;

	/// simd type of `compute_finale_list_simd`, e.g. `uint32x16_t`
	using simd_type = uint32x8_t;
	constexpr static uint32_t simd_lanes = simd_type::LIMBS;
	static_assert(simd_lanes <= 16);
	/// lane indices, loaded into a `simd_type`
	alignas(64) constexpr static uint32_t simd_lane_index[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

	/// NOTE: that we increase the size with about one bucket size to make sure that we can fully insert a bucket
	// NOTE: a multiple of `simd_lanes`, the simd check pads the list
	constexpr static size_t final_list_real_max_size = (final_list_max_size + bucketsize*bucketsize) + ((simd_lanes - ((final_list_max_size + bucketsize*bucketsize)%simd_lanes))%simd_lanes);
	// NOTE: `+1`, the simd check gathers 4 bytes per 2 byte row index
	alignas(256) std::array<uint16_t[p], final_list_real_max_size + 1> final_list_left;
	alignas(256) std::array<uint16_t[p], final_list_real_max_size + 1> final_list_right;

	Stern() noexcept {
		expected_loops = config.compute_loops();
		pHTr = pHT + baselist_enumeration_length *NR_HT_T_LIMBS;

		hm = new HM{};
		assert(hm);

		bEnum = new EnumHashMap<configEnum, HM>{lHT, hm};
		cEnum = new CollisionHashMap<configEnum, HM>{lHT, hm};
	}

	~Stern() noexcept {
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
				assert(left[j] < kl_half + config.epsilon);
				assert(right[j] < (k+l));

				climb ^= (HT[left[j]][0] ^ HT[right[j]][0]);
			}

			// The first `l` coordinates are not covered by the identity part of
			// the parity check matrix, hence they must be zero. Stern guarantees
			// this via the collision, Stern-IM (which matches on a different
			// window) does not. Without this check Stern-IM reports wrong solutions.
			static_assert(l < sizeof(climb)*8u);
			constexpr decltype(climb) l_mask = (decltype(climb)(1u) << l) - 1u;
			if ((climb & l_mask) != 0) {
				continue;
			}
			uint32_t wt = cryptanalysislib::popcount::popcount(climb);

			// early exit
			if (likely(wt > w-(2*p))) {
				continue;
			}

			/// compute the remaining limbs
			for (uint32_t i = 1; i < NR_HT_LIMBS; i++) {
				climb = ws.ptr(i);
				for (uint16_t j = 0; j < p; j++) {
					assert(left[j] < kl_half + config.epsilon);
					assert(right[j] < (k+l));

					climb ^= HT[left[j]][i] ^ HT[right[j]][i];
				}

				wt += cryptanalysislib::popcount::popcount(climb);
			}

			if ((wt <= w - (2*p)) && not_found) {
				not_found = false;
				cycles = cpucycles() - cycles;
				for (uint16_t j = 0; j < p; ++j) {
					solutions[j]     = left[j];
					solutions[p + j] = right[j];
				}

				return true;
			}
		}

		final_list_current_size = 0;
		return false;
	}

	/// checks for collisions in the final list.
	/// NOTE: `simd_lanes` elements are checked simultaneously
	bool compute_finale_list_simd() noexcept {
		static_assert(sizeof(l_type) == sizeof(uint32_t));
		if (final_list_current_size == 0) {
			return false;
		}

		// pad the list to a multiple of `simd_lanes` by repeating the last entry. Otherwise
		// the last `final_list_current_size % simd_lanes` entries are never checked.
		const size_t padded_size = (final_list_current_size + simd_lanes - 1u) / simd_lanes * simd_lanes;
		assert(padded_size <= final_list_real_max_size);
		for (size_t i = final_list_current_size; i < padded_size; i++) {
			// NOTE: the last argument is the number of elements, not bytes
			cryptanalysislib::memcpy(final_list_left[i],  final_list_left[final_list_current_size - 1u],  p);
			cryptanalysislib::memcpy(final_list_right[i], final_list_right[final_list_current_size - 1u], p);
		}

		simd_type rows1[p], rows2[p];
		const simd_type filter_mask = simd_type::set1(w - 2*p + 1);
		const simd_type mul_mask    = simd_type::set1(NR_HT_T_LIMBS);
		const simd_type low16_mask  = simd_type::set1(0xFFFFu);
		const simd_type zero        = simd_type::set1(0u);
		// offset of the row index `j` of lane `c` is `c*p + j` (in uint16_t)
		const simd_type entry_offset = simd_type::load(simd_lane_index) * simd_type::set1(p);
		// NOTE: `ws.ptr(i)` is the i-th 64 bit limb, `pHT` has 32 bit limbs
		const uint32_t *ws32 = (const uint32_t *)ws.ptr();
		// the first `l` coordinates must be zero, see `compute_finale_list`
		const simd_type l_mask = simd_type::set1((l >= 32u) ? uint32_t(-1) : uint32_t((1u << l) - 1u));

		for (size_t cindex = 0; cindex < padded_size; cindex += simd_lanes) {
			const uint16_t *left  = final_list_left[cindex];
			const uint16_t *right = final_list_right[cindex];

			// NOTE: we need to multiply the offset of the rows, by the number of limbs
			// in each row.
			#pragma unroll
			for (uint16_t j = 0; j < p; j++) {
				rows1[j] = (simd_type::template gather<2>(left  + j, entry_offset) & low16_mask) * mul_mask;
				rows2[j] = (simd_type::template gather<2>(right + j, entry_offset) & low16_mask) * mul_mask;
			}

			simd_type wt = zero;
			uint64_t wt_ = 0, valid = 0;

			for (uint32_t i = 0; i < NR_HT_T_LIMBS; i++) {
				simd_type climb = simd_type::set1(ws32[i]);
				const l_type *base_pHT = pHT + i;

				#pragma unroll
				for (uint16_t j = 0; j < p; j++) {
					const simd_type t11 = simd_type::template gather<sizeof(l_type)>(base_pHT, rows1[j]);
					const simd_type t21 = simd_type::template gather<sizeof(l_type)>(base_pHT, rows2[j]);
					climb = climb ^ t11 ^ t21;
				}

				// lanes with a zero `l` window. Stern-IM matches on a different window
				if (i == 0) {
					valid = (climb & l_mask) == zero;
				}

				// popcount
				wt = wt + simd_type::popcnt(climb);
				wt_ = (filter_mask > wt) & valid;

				// if the mask `wt_` is zero, we can do an early exit, because the weight
				// is already too big, for all elements.
				if (wt_ == 0) {
					break;
				}
			}

			// if we pass this check, we found a global solution
			if (unlikely((wt_ != 0) && not_found)) {
				not_found = false;
				cycles = cpucycles() - cycles;

				const uint32_t pos = __builtin_ctzll(wt_);
				for (uint16_t j = 0; j < p; ++j) {
					solutions[j]     = final_list_left[cindex + pos][j];
					solutions[p + j] = final_list_right[cindex + pos][j];
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
			assert(solutions[i] < k+l);
			Label::add(tmp.ptr(), tmp.ptr(), HT.row(solutions[i]));
		}

		for (uint32_t i = 0; i < n-k; ++i) {
			const auto bit = tmp.get(i);
			tmpe.set(bit, 0, n-k-1-i);
		}

		for (uint32_t i = 0; i < 2*p; ++i) {
			assert(n-k-l + solutions[i] < n);
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
					assert(bEnum->check_hashmap2(syndrome, index1, 2 * p, l));
					assert((a1 ^ a2) == 0);
				}
				(void)a1;
				(void)a2;

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
