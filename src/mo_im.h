#ifndef DECODING_MO_IM_H
#define DECODING_MO_IM_H

#include "matrix/binary_matrix.h"
#include "container/hashmap.h"
#include "popcount/popcount.h"
#include "math/bc.h"
#include "alloc/alloc.h"
#include "hash/simple.h"
#include "mitm.h"
#include "isd.h"

using namespace cryptanalysislib;


struct ConfigMOIM : public ConfigISD {
public:
	const uint32_t l1 = 0;
	const uint32_t nr_views = 0;
	const uint32_t final_list_size = 1024;

	void print() const noexcept {
		ConfigISD::print();
		std::cout << "{ \"nr_views\": " << nr_views
				<< ", \"final_list_size\": " << final_list_size
				<< ", \"l1\": " << l1
		        << " }" << std::endl;
	}
};


template<const ConfigISD &isd,  const ConfigMOIM &config>
class MOIM : public ISDInstance<uint64_t, isd>{
public:
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t k = config.k;
	constexpr static uint32_t w = config.w;
	constexpr static uint32_t p = config.p; // NOTE: from now on p is the baselist p not the full p.
	constexpr static uint32_t l = config.l;
	constexpr static uint32_t q = config.q;

	constexpr static uint32_t nr_views = config.nr_views;
	constexpr static uint32_t l1 = config.l1;
	constexpr static uint32_t l2 = l - l1;
	static_assert(l1 <= 32);
	static_assert(l2 <= 32);

	using ISD = ISDInstance<uint64_t, isd>;
	using PCMatrixOrg 	= ISD::PCMatrixOrg;
	using PCMatrixOrg_T = ISD::PCMatrixOrg_T;
	using PCMatrix 		= ISD::PCMatrix;
	using PCMatrix_T 	= ISD::PCMatrix_T;
	using PCSubMatrix 	= ISD::PCSubMatrix;
	using PCSubMatrix_T = ISD::PCSubMatrix_T;
	using Syndrome 		= ISD::Syndrome;
	using Error 		= ISD::Error;
	using Label 		= ISD::Label;
	using limb_type 	= ISD::limb_type;
	using l_type   		= MinLogTypeTemplate<std::max(l1, l2), 32>;
	using ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::ghz,ISD::cycles,ISD::gaus_cycles,ISD::periodic_print,ISD::ws,ISD::HT,ISD::e;

	constexpr static uint32_t kl_half = (k+l)/2;
	size_t solutions[4*p] = {0};

	/// internal arrays pointing to
	l_type *l1HT, *l1HTr;
	constexpr static uint32_t NR_HT_LIMBS 	= (n-k+sizeof(limb_type)*8 - 1u)/((sizeof(limb_type)*8));
	constexpr static uint32_t NR_HT_T_LIMBS = (n-k+(sizeof(l_type)*8) - 1u)/(sizeof(l_type)*8);

	/// Maximal list size of the final list
	constexpr static size_t final_list_size = config.final_list_size;

	// hashmap stuff
	using keyType = LogTypeTemplate<l>;
	using V1 = CollisionType<l_type, 1*p>; // TODO generalize to `SternCollisionType`
	using V2 = CollisionType<l_type, 2*p>;

	constexpr static uint64_t HM1_nrbuckets = 1u << l1;
	constexpr static uint64_t HM1_bucketsize = bc(k+l, p) >> l1;
	constexpr static uint64_t HM2_nrbuckets = 1u << l2;
	constexpr static uint64_t HM2_bucketsize = 16; // TODO

	constexpr static SimpleHashMapConfig HM1c{HM1_bucketsize, HM1_nrbuckets, config.threads};
	constexpr static SimpleHashMapConfig HM2c{HM2_bucketsize, HM2_nrbuckets, config.threads};
	using HM1 = SimpleHashMap<keyType, V1, HM1c, Hash<l_type, 0, l1>>;
	using HM2 = SimpleHashMap<keyType, V2, HM2c, Hash<l_type, 0, l2>>;
	HM1 *hm1; HM2 *hm2;

	using HM1_load_type= HM1::load_type;

	constexpr static ConfigEnumHashMap configEnum{config};
	EnumHashMap<configEnum, HM1> *bEnum;
	CollisionHashMap<configEnum, HM1> *cEnum;

	/// base list stuff
	constexpr static size_t baselist_enumeration_length = kl_half;
	constexpr static size_t baselist_enumeration_weigth = p;
	constexpr static size_t baselist_enumeration_size =
			bc(baselist_enumeration_length, baselist_enumeration_weigth);
	/// NOTE: that we increase the size with about one bucket size to make sure that we can fully insert a bucket
	constexpr static size_t final_list_real_max_size = final_list_size; // TODO correct calc
	size_t fcs = 0;/// Current load of the final list
	alignas(256) std::array<uint16_t [2*p], final_list_real_max_size> final_list_left;
	alignas(256) std::array<uint16_t [2*p], final_list_real_max_size> final_list_right;

	constexpr static size_t intermediate_max_size = final_list_size*2; // TODO correct calc
	size_t intermediate_current_size = 0;/// Current load of the final list
	size_t intermediate_left_size = 0, intermediate_right_size = 0;
	alignas(256) std::array<uint16_t[2*p], intermediate_max_size> intermediate_list_left;
	alignas(256) std::array<uint16_t[2*p], intermediate_max_size> intermediate_list_right;

	MOIM() noexcept {
		constexpr size_t size_lHT = roundToAligned<1024>(sizeof(l_type) * (k+l));
		l1HT =(l_type *)cryptanalysislib::aligned_alloc(1024, size_lHT);
		ASSERT(l1HT);

		hm1 = new HM1{};
		ASSERT(hm1);
		hm2 = new HM2{};
		ASSERT(hm2);

		bEnum = new EnumHashMap<configEnum, HM1>{l1HT, hm1};
		cEnum = new CollisionHashMap<configEnum, HM1>{l1HT, hm1};
	}

	bool compute_finale_list() noexcept {
		// go through the list
		for (size_t cindex = 0; cindex < fcs; cindex++) {
			const uint16_t *left = final_list_left[cindex];
			const uint16_t *right = final_list_right[cindex];

			auto climb = ws.ptr(0);
			for (uint32_t j = 0; j < 2*p; j++) {
				ASSERT(left[j] < (k+l));
				ASSERT(right[j] < (k+l));

				climb ^= (HT[left[j]][0] ^ HT[right[j]][0]);
			}

			ASSERT((climb & ((1ul<<l1)-1ul)) == 0);
			uint32_t wt = cryptanalysislib::popcount::popcount(climb);

			// early exit
			if (likely(wt > (w-(4*p)))) {
				continue;
			}

			/// compute the remaining limbs
			for (uint32_t i = 1; i < NR_HT_LIMBS; i++) {
				climb = ws.ptr(i);
				for (uint32_t j = 0; j < 2*p; j++) {
					ASSERT(left[j] < (k+l));
					ASSERT(right[j] < (k+l));

					climb ^= HT[left[j]][i] ^ HT[right[j]][i];
				}

				wt += cryptanalysislib::popcount::popcount(climb);
			}

			if (unlikely((wt <= (w - (4*p))) && not_found)) {
				not_found = false;
				cycles = cpucycles() - cycles;
				for (uint16_t j = 0; j < 2*p; ++j) {
					solutions[j] = left[j];
					solutions[j + 2*p] = right[j];
				}

				return true;
			}
		}

		fcs = 0;
		return false;
	}


	/// NOTE: assumes that the hashmap is already filled_mm256_i32gather_epi64
	/// computes the left an right intermediate list L, R for the may ozerov
	/// algorithm
	/// NOTE: uses AVX2
	void intermediate_lists_avx2() noexcept {
		static_assert(l1 < 32);
		uint32x8_t rows[p];
		uint32x8_t right_index = uint32x8_t::setr(0, 1, 2, 3, 4, 5, 6, 7);
		constexpr uint32x8_t eight = uint32x8_t::set1(8);
		constexpr uint32x8_t multiplier = uint32x8_t::set1(HM1_bucketsize);
		uint32x8_t tmpl{}, tmpr{};

		// reset the counters
		// TODO replace this with the new `intermediatelist` stuff
		intermediate_L2_ctr = 0; intermediate_L2_ctr = 0;
		L_ctr = 0; R_ctr = 0;

		// first compute collision between (L4,L2) and L1
		for (size_t right_index_ = 0;
		     right_index_ < baselist_enumeration_size/8;
		     right_index_++) {
			biject_avx<baselist_enumeration_length, p>(right_index, rows);

			tmpl = uint32x8_t::set1(0);
			tmpr = uint32x8_t::set1(syndrome);

			// only compute the l-part limb for the left side L2
			// For L4 we simply need to xor the syndrome in
			for (uint16_t j = 0; j < p; j++) {
				const auto data = uint32x8_t::template gather<4>(l1HTr, rows[j]);
				tmpl ^= data;
			}
			tmpr ^= tmpl;

			// Now do the lookup
			const uint32x8_t hashmap_offset_L2 = tmpl * multiplier;
			const uint32x8_t hashmap_offset_L4 = tmpr * multiplier;

			const uint32x8_t left_load_L2 = uint32x8_t::template gather<sizeof(HM1_load_type)>(hm1->load_ptr(), tmpl);
			const uint32x8_t left_load_L4 = uint32x8_t::template gather<sizeof(HM1_load_type)>(hm1->load_ptr(), tmpr);

			for (uint64_t j = 0; j < HM1_bucketsize; j++) {
				const uint32x8_t j1 = uint32x8_t::set1(j);
				const int mask2_L2 = uint32x8_t::gt(left_load_L2, j1);
				const int mask2_L4 = uint32x8_t::gt(left_load_L4, j1);

				// skip if no collision found
				if (mask2_L2 == 0 && mask2_L4) {
					continue;
				}

				// Do the lookup into the hashmap and get their values
				const uint32x8_t left_index_L2 = uint32x8_t::template
				        gather<sizeof(V1)>(hm1->ptr(), hashmap_offset_L2 + j1);
				const uint32x8_t left_index_L4 = uint32x8_t::template
				        gather<sizeof(V1)>(hm1->ptr(), hashmap_offset_L4 + j1);

				// compute the shuffle down permutation.
				// with this permutation one can permute the limbs down
				const uint32x8_t shfl_down_mask_L2 = uint32x8_t::pack(mask2_L2);
				const uint32x8_t shfl_down_mask_L4 = uint32x8_t::pack(mask2_L4);

				// apply the permutation
				const uint32x8_t lt_L2 = uint32x8_t::permute(left_index_L2, shfl_down_mask_L2);
				const uint32x8_t rt_L2 = uint32x8_t::permute(right_index,   shfl_down_mask_L2);
				const uint32x8_t lt_L4 = uint32x8_t::permute(left_index_L4, shfl_down_mask_L4);
				const uint32x8_t rt_L4 = uint32x8_t::permute(right_index,   shfl_down_mask_L4);

				//store the indices of the elements
				_mm256_storeu_si256((__m256i *)(intermediate_L2_left.data()  + intermediate_L2_ctr), lt_L2);
				_mm256_storeu_si256((__m256i *)(intermediate_L2_right.data() + intermediate_L2_ctr), rt_L2);
				_mm256_storeu_si256((__m256i *)(intermediate_L4_left.data()  + intermediate_L4_ctr), lt_L4);
				_mm256_storeu_si256((__m256i *)(intermediate_L4_right.data() + intermediate_L4_ctr), rt_L4);

				intermediate_L2_ctr  += __builtin_popcount(mask2_L2);
				intermediate_L4_ctr += __builtin_popcount(mask2_L4);
			} // end of iterate over HM1

			// write the intermediate collision into the intermediate lists L and R
			// TODO was ist das construct_LR();
		} //  for each element in L2/ L4
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

	// runs: optimized stern
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();

		while (not_found && loops < isd.loops) {
			const uint32_t tid = 0;
			const bool simd = false;
			ISD::step();
			ISD::extract_lHT(l1HT);
			// ISD::extract_pHT(pHT); // TODO

			const l_type iT1 = fastrandombytes_uint64() & ((1ul << l1) - 1ul);
			const l_type iT2 = ((iT1 ^ syndrome) & ((1ul << l1) - 1ul));
			intermediate_current_size = 0;
			intermediate_right_size = 0;
			intermediate_left_size = 0;

			// collision function
			auto f1 = [&](const l_type a, uint16_t *index) __attribute__((always_inline)) {
				ASSERT(bEnum->check_hashmap2(iT1, index, 2*p, l1));
				if (intermediate_current_size >= intermediate_max_size) {
					return;
				}

   				#pragma unroll
				for (uint16_t i = 0; i < 2*p; ++i) {
					intermediate_list_left[intermediate_current_size][i] = index[i];
				}
				intermediate_current_size += 1;
			};

			auto f2 = [&](const l_type a, uint16_t *index) __attribute__((always_inline)) {
				ASSERT(bEnum->check_hashmap2(iT2, index, 2*p, l1));
				if (intermediate_current_size >= intermediate_max_size) {
					return;
				}

				#pragma unroll
				for (uint16_t i = 0; i < 2u*p; ++i) {
					intermediate_list_right[intermediate_current_size][i] = index[i];
				}
				intermediate_current_size += 1;
			};

			bEnum->step(0, tid, simd);
			cEnum->step(iT1, f1, tid, simd);
			intermediate_left_size = intermediate_current_size;
			intermediate_current_size = 0;
			cEnum->step(iT2, f2, tid, simd);
			intermediate_right_size = intermediate_current_size;

			for (uint32_t i = 0; i < nr_views; ++i) {
				constexpr uint32_t offset = l1 + 0; //  TODO richtiger wert

				// TODO: hier optimierung schreiben die einfach 32/64bit aligned l2 bits holt
				ISD::template extract_lHT<limb_type, offset, l2>(l1HT);
				ISD::template extract_syndrome_limb<offset, l2>();

				for (size_t j = 0; j < intermediate_left_size; ++j) {
					l_type tmp = syndrome;
					#pragma unroll
					for (uint32_t s = 0; s < 2*p; s++) {
						ASSERT(intermediate_list_left[j][s] < (k+l));
						tmp ^= l1HT[intermediate_list_left[j][s]];
					}

					ASSERT(tmp < (1u<<l2));
					hm2->insert(tmp, V2::create(tmp, intermediate_list_left[j]));
				}

				for (size_t j = 0; j < intermediate_right_size; ++j) {
					l_type tmp = 0;
					#pragma unroll p
					for (uint32_t s = 0; s < p; s++) {
						ASSERT(intermediate_list_left[j][s] < (k+l));
						tmp ^= l1HT[intermediate_list_right[j][s]];
					}

					ASSERT(tmp < (1u<<l2));
					typename HM2::load_type left_load;
					const size_t left_base = hm2->find(tmp, left_load);
					for (size_t s = left_base; s < left_base + left_load; s++) {
						for (uint32_t m = 0; m < 2*p; ++m) {
							final_list_left [fcs][m] = hm2->ptr(s).index[m];
							final_list_right[fcs][m] = intermediate_list_right[j][m];
						}

						fcs += 1;
						if (fcs >= final_list_real_max_size) {
							compute_finale_list();
						}
					} // left size
				} // right side

				compute_finale_list();
				hm2->clear();
			} // nr_views
		}

		// compute the final solution
		reconstruct();
		return loops;
	}

	// print one-time information
	void info() const noexcept {

	}
};
#endif//DECODING_MO_IM_H
