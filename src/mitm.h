#ifndef DECODING_MITM_H
#define DECODING_MITM_H
#include <tuple>

#include "combination/chase.h"
#include "matrix/matrix.h"
#include "container/hashmap.h"
#include "popcount/popcount.h"
#include "math/bc.h"
#include "alloc/alloc.h"
#include "helper.h"
#include "isd.h"

using namespace cryptanalysislib;

template<typename T, typename indexType, const uint32_t p>
struct CollisionType {
public:
	using index_type = indexType;
	using data_type = T;

	T __data;
	index_type index[p];

	constexpr T data() const noexcept {
		return __data;
	}

	template<const uint32_t strive=1>
	constexpr static __FORCEINLINE__ CollisionType create(const T data,
												 const index_type *index) noexcept {
		static_assert(strive > 0);
		CollisionType ret;
		ret.__data = data;
		if constexpr (strive == 1) {
			memcpy(ret.index, index, p*sizeof(index_type));
		} else {
			for (uint32_t i = 0; i < p; ++i) {
				ret.index[i] = index[i*strive];
			}
		}
		return ret;
	}
};

template<typename T, typename indexType, const uint32_t p>
struct SternCollisionType {
public:
	using index_type = indexType;
	using data_type = T;

	index_type index[p];

	constexpr T data() const noexcept {
		return 0u;
	}

	/// NOTE: this container drops the `data` value
	/// \tparam strive
	/// \param data
	/// \param index
	/// \return
	template<const uint32_t strive=1>
	constexpr static __FORCEINLINE__ SternCollisionType create(const T data,
	                                             const index_type *index) noexcept {
		(void) data;
		static_assert(strive > 0);
		SternCollisionType ret;
		if constexpr (strive == 1) {
			memcpy(ret.index, index, p*sizeof(index_type));
		} else {
			for (uint32_t i = 0; i < p; ++i) {
				ret.index[i] = index[i*strive];
			}
		}
		return ret;
	}
};

struct ConfigEnumHashMap {
private:
	constexpr ConfigEnumHashMap() : n(0), p(0), l(0) {}
public:
	const uint32_t n; // this should be k+l
	const uint32_t p;
	const uint32_t l;
	const uint32_t threads=1;

	const bool full_length=false;
	const uint32_t epsilon=full_length ? n/2 : 0;

	/// if set to true, instead of the row indices, the index in
	/// the chase sequence is saved in the hashmap
	const bool save_index = false;

	/// if true you need to pass the changelist to the algorithms
	const bool use_changelist = false;

	constexpr ConfigEnumHashMap(const uint32_t n,
								const uint32_t p,
								const uint32_t l,
	                            const uint32_t threads=1,
	                            const uint32_t epsilon=0,
	                            const bool save_index=false,
	                            const bool use_changelist=false) noexcept :
	    n(n), p(p), l(l), threads(threads), epsilon(epsilon),
	    save_index(save_index), use_changelist(use_changelist) {}

	constexpr ConfigEnumHashMap(const ConfigISD &config) noexcept :
		n(config.k+config.l) , p(config.p), l(config.l), threads(config.threads), epsilon(config.epsilon) {}
};

struct ConfigEnumHashMapD2 : public ConfigEnumHashMap {
public:
	const uint32_t l1;
	const uint32_t l2 = l - l1;
};

template<const uint32_t d>
struct ConfigEnumHashMapD : public ConfigEnumHashMap {
public:
	const uint32_t ls[d];
};


template<const ConfigEnumHashMap &config,
         class HashMap>
requires HashMapAble<HashMap>
class EnumHashMap {
public:
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t n_half = config.n/2;
	constexpr static uint32_t p = config.p;
	constexpr static uint32_t l = config.l;
	constexpr static uint32_t threads = config.threads;

	// list stuff
	constexpr static size_t enumeration_length = n_half + config.epsilon;
	constexpr static size_t enumeration_weight = p;
	constexpr static size_t enumeration_size = bc(enumeration_length, enumeration_weight);
	constexpr static size_t enumeration_size_per_thread = enumeration_size/threads;

	using HM_DataType 			= HashMap::data_type;
	using HM_DataType_IndexType = HM_DataType::index_type;
	using l_type 				= HM_DataType::data_type;

	using cle = std::pair<uint16_t, uint16_t>;

	HashMap *hm;
	const l_type *lHT;
	cle *cL;

	EnumHashMap(const l_type *lHT,
	            HashMap *hm,
	            cle *cL=nullptr) noexcept :
	   hm(hm), lHT(lHT), cL(cL) {}

	constexpr inline void step(const l_type syndrome, const uint32_t tid=0, const bool simd=true) noexcept {
		if (simd) {
			fill_hashmap_simd(syndrome, tid);
		} else {
			fill_hashmap(syndrome, tid);
		}
	}

	/// checks weather the two given indices are a collision or not.
	/// \return true/false
	constexpr bool check_hashmap2(const l_type syndrome, const uint16_t *index,
	                              const size_t nr_of_index, const uint32_t bits,
	                              const uint32_t offset=0) noexcept {
		const l_type mask = ((1u << bits) - 1u) << offset;

		l_type v = 0;
		for (uint32_t i = 0; i < nr_of_index; ++i) {
			v ^= lHT[index[i]];
		}

		v &= mask;
		return v == syndrome;
	}

	/// first clears the internal hashmap
	/// hashes the left list and safes it in the hashmap
	/// non-avx version
	constexpr inline void fill_hashmap(const l_type syndrome,
	                                   const uint32_t tid=0) noexcept {
		// first clear it
		hm->clear(tid);
		alignas(32) uint16_t rows[p];

		const size_t start = enumeration_size_per_thread * tid;
		const size_t end   = tid == threads-1 ? enumeration_size : enumeration_size_per_thread * (tid + 1);

		if constexpr (config.use_changelist) {
			ASSERT(!config.save_index);
			ASSERT(cL);

			chase<enumeration_length, enumeration_weight> c;

			l_type tmp = syndrome;
			for (uint32_t i = 0; i < p; ++i) { tmp ^= lHT[i]; }

			for (size_t i = start; i < end; i++) {
				if constexpr (config.save_index) {
					hm->insert(tmp, HM_DataType::create(tmp, (HM_DataType_IndexType *)&i));
				} else {
					c.biject(i, rows);
					hm->insert(tmp, HM_DataType::create(tmp, (HM_DataType_IndexType *)rows));
				}

				tmp ^= lHT[cL[i].first] ^ lHT[cL[i].second];
			}
		} else {
			for (size_t i = start; i < end; i++) {
				biject<enumeration_length, p>(i, rows);
				l_type tmp = syndrome;

				#pragma unroll p
				for (uint16_t j = 0; j < p; j++) {
					tmp ^= lHT[rows[j]];
				}

				/// NOTE: this assert only makes sense if we are in the stern case.
				/// In the BJMM/MO case we normally compute bigger l1/l2 to ensure we have this
				/// data already in the hashmap for subsequece steps in the tree
				//ASSERT(tmp < (1u << l));

				if constexpr (config.save_index) {
					hm->insert(tmp, HM_DataType::create(tmp, (HM_DataType_IndexType *)&i));
				} else {
					hm->insert(tmp, HM_DataType::create(tmp, (HM_DataType_IndexType *)rows));
				}
			}
		}
	}

	/// first clears the internal hashmap
	/// hashes the left side into the internal hashmap
	/// NOTE: this is the avx2 version. 8 elements < 32bit are hashed 
	/// simultaneous
	constexpr void fill_hashmap_simd(const l_type syndrome, const uint32_t tid=0) noexcept {
		static_assert(sizeof(l_type) >= 4);
		// first clear it
		hm->clear(tid);

		const size_t start = enumeration_size_per_thread * tid;
		const size_t end   = tid == threads-1 ? enumeration_size : enumeration_size_per_thread * (tid + 1);
		size_t i = start;

		if constexpr (sizeof(l_type) == 4) {
			alignas(32) uint32x8_t rows[p]{};
			alignas(32) uint32x8_t a = uint32x8_t::setr(start+0, start+1, start+2, start+3, start+4, start+5, start+6, start+7);
			alignas(32) const uint32x8_t eight = uint32x8_t::set1(8);
			alignas(32) const uint32x8_t syndrome_ = uint32x8_t::set1(syndrome);

			for (; i + 8 < end; i += 8) {
				biject_simd<enumeration_length, p>(a, rows);
				alignas(32) uint32x8_t tmp = syndrome_;

				#pragma unroll p
				for (uint16_t j = 0; j < p; j++) {
					const auto data = uint32x8_t::template gather<4>(lHT, rows[j]);
					tmp ^= data;
				}

				#pragma unroll 8
				for (uint16_t j = 0; j < 8; j++) {
					if constexpr (config.save_index){
						hm->insert(tmp.v32[j], HM_DataType::template
							create<8>(tmp.v32[j], ((HM_DataType_IndexType *)a.v32) + j));
					} else {
						hm->insert(tmp.v32[j], HM_DataType::template
							create<8>(tmp.v32[j], ((HM_DataType_IndexType *)rows) + j));
					}
				}

				a = a + eight;
			}
		} else if constexpr (sizeof(l_type) == 8){
			alignas(32) uint32x8_t rows[p];
			alignas(32) uint32x8_t a = uint32x8_t::setr(start+0, start+1, start+2, start+3, start+4, start+5, start+6, start+7);
			alignas(32) const uint32x8_t eight = uint32x8_t::set1(8);
			alignas(32) const uint64x4_t syndrome_ = uint64x4_t::set1(syndrome);

			for (; i + 8 < end; i += 8) {
				biject_simd<enumeration_length, p>(a, rows);
				alignas(32) uint64x4_t tmp1 = syndrome_;
				alignas(32) uint64x4_t tmp2 = syndrome_;

				#pragma unroll p
				for (uint16_t j = 0; j < p; j++) {
					const auto data1 = uint32x8_t::template gather<4>(lHT, rows[j].v128[0]);
					const auto data2 = uint32x8_t::template gather<4>(lHT, rows[j].v128[1]);
					tmp1 ^= data1;
					tmp2 ^= data2;
				}

				for (uint16_t j = 0; j < 4; j++) {
					if constexpr (config.save_index) {
						hm->insert(tmp1.v32[j], HM_DataType::template
							create<8>(tmp1.v32[j], ((HM_DataType_IndexType *)a.v64) + j));
					} else {
						hm->insert(tmp1.v32[j], HM_DataType::template
							create<8>(tmp1.v32[j], ((HM_DataType_IndexType *)rows) + j));
					}
				}

				for (uint16_t j = 4; j < 8; j++) {
					if constexpr (config.save_index) {
						hm->insert(tmp2.v32[j], HM_DataType::template
							create<8>(tmp2.v32[j], ((HM_DataType_IndexType *)a.v64) + j));
					} else {
						hm->insert(tmp2.v32[j], HM_DataType::template
							create<8>(tmp2.v32[j], ((HM_DataType_IndexType *)rows) + j));
					}
				}

				a = a + eight;
			}
		}

		// tail mngmt
		alignas(32) uint16_t __rows[p];
		for (; i < end; i++) {
			biject<enumeration_length, p>(i, __rows);
			l_type tmp = syndrome;

			#pragma unroll p
			for (uint16_t j = 0; j < p; j++) {
				tmp ^= lHT[__rows[j]];
			}

			/// NOTE: this assert only makes sense if we are in the stern case.
			/// In the BJMM/MO case we normally compute bigger l1/l2 to ensure we have this
			/// data already in the hashmap for subsequece steps in the tree
			//ASSERT(tmp < (1u << l));

			if constexpr (config.save_index) {
				hm->insert(tmp, HM_DataType::create(tmp, (HM_DataType_IndexType *)&i));
			} else {
				hm->insert(tmp, HM_DataType::create(tmp, (HM_DataType_IndexType *)__rows));
			}
		}
	}

	/// MAYBE: via reflections simplify
	constexpr void print() const noexcept {
		std::cout << "{ \"n\": " << n
				  << ", \"p\": " << p
		          << ", \"l\": " << l
				  << ", \"threads\": " << threads
				  << ", \"epsilon\": " << config.epsilon
				  << ", \"full_length\": " << config.full_length
				  << ", \"save_index\": " << config.save_index
				  << ", \"use_changelist\": " << config.use_changelist
				  << ", \"sizeof(l_type)\": " << sizeof(l_type)
				  << ", \"sizeof(HM_DataType)\": " << sizeof(HM_DataType)
				  << ", \"sizeof(HM_DataType_IndexType)\": " << sizeof(HM_DataType_IndexType )
				  << ", \"enumeration_size\": " << enumeration_size
				  << ", \"enumeration_weight\": " << enumeration_weight
				  << " } " << std::endl;
		hm->print();
	}
};

template<const ConfigEnumHashMap &config,
         class HashMap>
requires HashMapAble<HashMap>
class CollisionHashMap {
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t n_half = n / 2u;
	constexpr static uint32_t p = config.p;
	constexpr static uint32_t l = config.l;
	constexpr static uint32_t threads = config.threads;
	static_assert(config.epsilon <= n_half);

	// list stuff
	constexpr static size_t enumeration_length = n_half + config.epsilon;
	constexpr static size_t enumeration_weight = p;
	constexpr static size_t enumeration_size = bc(enumeration_length, enumeration_weight);
	constexpr static size_t enumeration_size_per_thread = enumeration_size/threads;

	using HM_LoadType  			= typename HashMap::load_type;
	using HM_DataType  			= typename HashMap::data_type;
	using HM_DataType_IndexType = typename HM_DataType::index_type;
	using l_type 				= typename HM_DataType::data_type;
	constexpr static size_t HM1_BUCKETSIZE = HashMap::bucketsize;

	using cle = std::pair<uint16_t, uint16_t>;

	cle *cL = nullptr;
	HashMap *hm = nullptr;
	const l_type *lHT = nullptr,
	             *lHTr = nullptr;

public:
	constexpr CollisionHashMap(const l_type *_lHT, HashMap *hm, cle *cL = nullptr) noexcept :
		cL(cL), hm(hm), lHT(_lHT), lHTr(_lHT + n_half - config.epsilon) {
	}

	///
	/// \tparam F
	/// \param iT
	/// \param f
	/// \param tid
	/// \param simd
	/// \return
	template<typename F>
	constexpr inline bool step(const l_type iT, F &&f,
	                           const uint32_t tid=0,
	                           const bool simd=true) noexcept {
		if (simd) {
			return coll_hashmap_simple_simd(iT, tid, f);
		} else {
			return coll_hashmap_simple(iT, tid, f);
		}
	}

	/// computes the right list and start searching for collision
	/// NOTE: without NN
	/// NOTE: without AVX
	/// \tparam F
	/// \param iT intermediate target
	/// \param f lambda function to execute for each collision (make sure its thread safe)
	/// \return if found
	template<typename F>
	bool coll_hashmap_simple(const l_type iT, const uint32_t tid, F &&f) {
		const size_t start = enumeration_size_per_thread * tid;
		const size_t end   = tid == threads-1 ? enumeration_size : enumeration_size_per_thread * (tid + 1);

		if constexpr (config.use_changelist) {
			ASSERT(cL);

			l_type tmp = iT;
			for (uint32_t i = 0; i < p; ++i) { tmp ^= lHTr[i]; } // TODO correct start
			for (size_t right_index = start; right_index < end;
				 right_index++) {

				// for every collision.
				typename HashMap::load_type left_load;
				const size_t left_base = hm->find(tmp, left_load);
				for (uint64_t j = left_base; j < left_base + left_load; j++) {
					const auto ind = hm->ptr(j).index;
					f(tmp, hm->ptr(j).data, (HM_DataType_IndexType *)&ind,
					  (HM_DataType_IndexType *)right_index, 1);
				}

				tmp ^= lHTr[cL[right_index].first] ^ lHTr[cL[right_index].second];
			}

		} else {
			alignas(32) uint16_t rows[2*p];
			uint16_t *rows2 = rows + p;

			// iterate over the right list
			for (size_t right_index = start; right_index < end;
					right_index++) {
				l_type tmp = iT; // first add the intermediate target
				biject<enumeration_length, p>(right_index, rows2);

				#pragma unroll p
				for (uint16_t j = 0; j < p; j++) {
					rows2[j] += enumeration_length;
					tmp ^= lHT[rows2[j]];
				}

				// for every collision.
				typename HashMap::load_type left_load;
				const size_t left_base = hm->find(tmp, left_load);
				for (uint64_t j = left_base; j < left_base + left_load; j++) {
					// const l_type tmp2 = tmp ^ hm->ptr(j).data;
					for (uint32_t i = 0; i < p; ++i) {
						rows[i] = hm->ptr(j).index[i];
					}

					// f(tmp2, rows);
					f(tmp, hm->ptr(j).data(), (HM_DataType_IndexType *)rows, (HM_DataType_IndexType *)rows+p, 1);
				}
			}
		}
		// do not need to clear the rest in the buffers
		// as it's done in the main loop
		return false;
	}

	/// NOTE the following limitations
	/// - listType must be uint32_T
	/// - 2**l * HM1_BucketSize < 2**32
	/// NOTE: sorting of the final list is not possible because im not working
	/// 	on each bucket separate, but at 8 at the same time, meaning the
	/// 	right list is always sorted in a way.
	template<typename F>
	bool coll_hashmap_simple_simd(const l_type iT, const uint32_t tid, F &&f) noexcept {
		static_assert(sizeof(l_type) == 4);

		const size_t start = enumeration_size_per_thread * tid;
		const size_t end   = tid == threads-1 ? enumeration_size : enumeration_size_per_thread * (tid + 1);

		alignas(32) uint32x8_t rows[p]{};
		alignas(32) uint32x8_t right_index = uint32x8_t::setr(start+0, start+1, start+2, start+3, start+4, start+5, start+6, start+7);
		constexpr uint32x8_t eight  = uint32x8_t::set1(8);
		constexpr uint32x8_t offset = uint32x8_t::set1(HM1_BUCKETSIZE);
		constexpr uint32x8_t hm_load_mask = uint32x8_t::set1(0xff);

		// iterate the right list
		for (size_t right_index_ = start; right_index_ + 8 < end;
				right_index_+=8) {
			biject_simd<enumeration_length, p>(right_index, rows);
			uint32x8_t tmp = uint32x8_t::set1(iT);

			// compute the limb
			#pragma unroll
			for (uint16_t j = 0; j < p; j++) {
				tmp ^= uint32x8_t::template gather<4>(lHTr, rows[j]);
			}

			// NOTE tmp is already hashed
			const uint32x8_t hashmap_offset = tmp * offset;
			const uint32x8_t left_load = uint32x8_t::template gather
			        <sizeof(HM_LoadType)>
			        (hm->__internal_load_array, tmp) & hm_load_mask;

			#pragma unroll
			for (uint64_t j = 0; j < HM1_BUCKETSIZE; j++) {
				const uint32x8_t j1 = uint32x8_t::set1(j);
				const int mask2 = left_load > j1;

				// early exit, if no collision was found
				if (mask2 == 0) {
					break;
				}

				// const listType left_index = hm->__internal_hashmap_array[hashmap_offset + j];
				const uint32x8_t lookup = hashmap_offset + j1;
				const uint32x8_t left_index = uint32x8_t::template gather
				        <sizeof(HM_DataType)>
				        (hm->ptr(), lookup);

				const uint32x8_t shufmask = uint32x8_t::pack(mask2);
				const uint32x8_t lt = uint32x8_t::permute(left_index,  shufmask);
				const uint32x8_t rt = uint32x8_t::permute(right_index, shufmask);
				const uint32_t nr_cols = popcount::popcount(mask2);

				for (uint32_t v = 0; v < nr_cols; v++){
					f(0, 0, (HM_DataType_IndexType *)lt.v32, (HM_DataType_IndexType *)rt.v32, nr_cols);
				}

				//uint32x8_t::store(final_list_left.data() + final_list_current_size, lt);
				//uint32x8_t::store(final_list_right.data() + final_list_current_size, rt);
				//final_list_current_size += nr_cols;
				//ASSERT(final_list_current_size < final_list_real_max_size);
			}

			// prepare everything for the next iteration
			right_index = right_index + eight;
		}

		return false;
	}

	void print() const noexcept{
		hm->print();
		std::cout << "{ \"n\": " << n
		          << ", \"n/2\": " << n_half
				  << ", \"p\": " << p
		          << ", \"l\": " << l
				  << ", \"threads\": " << threads
				  << ", \"epsilon\": " << config.epsilon
				  << ", \"full_length\": " << config.full_length
				  << ", \"save_index\": " << config.save_index
				  << ", \"use_changelist\": " << config.use_changelist
				  << ", \"sizeof(l_type)\": " << sizeof(l_type)
				  << ", \"sizeof(HM_DataType)\": " << sizeof(HM_DataType)
				  << ", \"sizeof(HM_DataType_IndexType)\": " << sizeof(HM_DataType_IndexType )
				  << ", \"enumeration_size\": " << enumeration_size
				  << ", \"enumeration_weight\": " << enumeration_weight
		          << " } " << std::endl;
	}
};



template<const ConfigEnumHashMapD2 &config,
		 class HashMap1,
         class HashMap2>
requires HashMapAble<HashMap1> && HashMapAble<HashMap2>
class CollisionHashMapD2 {
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t n_half = n / 2u;
	constexpr static uint32_t p = config.p;
	constexpr static uint32_t l1 = config.l1;
	constexpr static uint32_t l2 = config.l2;
	constexpr static uint32_t l = l1 + l2;
	constexpr static uint32_t threads = config.threads;

	// list stuff
	constexpr static size_t enumeration_length = n_half + config.epsilon;
	constexpr static size_t enumeration_weigth = p;
	constexpr static size_t enumeration_size = bc(enumeration_length,enumeration_weigth);
	constexpr static size_t enumeration_size_per_thread = enumeration_size/threads;

	using HM_LoadType  			= typename HashMap2::load_type;
	using HM_DataType  			= typename HashMap2::data_type;
	using HM_DataType_IndexType = typename HM_DataType::index_type;
	using l_type 				= typename HM_DataType::data_type;

	const l_type *lHT, *lHTr;
	HashMap1 *hm1;
	HashMap2 *hm2;


public:
	CollisionHashMapD2(const l_type *_lHT,
	                   HashMap1 *hm1,
	                   HashMap2 *hm2) noexcept :
		lHT(_lHT), lHTr(_lHT + n_half - config.epsilon), 
		hm1(hm1), hm2(hm2) {}

	template<typename F>
	bool step(const l_type iT, F &&f, const uint32_t tid = 0, const bool simd=true) {
		/// TODO simd
		(void)simd;
		return coll_hashmap_simple(iT, tid, f);
	}

	/// computes the right list and start searching for collision
	/// NOTE: without NN
	/// NOTE: without AVX
	/// \tparam F
	/// \param iT intermediate target
	/// \param f lambda function to execute for each collision
	/// \return if found
	template<typename F>
	bool coll_hashmap_simple(const l_type iT, const uint32_t tid, F &&f) {
		const size_t start = enumeration_size_per_thread * tid;
		const size_t end   = tid == threads-1 ? enumeration_size : enumeration_size_per_thread * (tid + 1);

		alignas(32) uint16_t rows[4*p];
		uint16_t *rows2 = rows + 2*p;
		uint16_t *rows3 = rows + 3*p;

		// iterate over the right list
		for (size_t right_index = start; right_index < end;
			 right_index++) {
			l_type tmp = iT; // first add the intermediate target
			biject<enumeration_length, p>(right_index, rows3);

			#pragma unroll p
			for (uint16_t j = 0; j < p; j++) {
				ASSERT((rows3[j] + enumeration_length) < n);
				rows3[j] += enumeration_length;
				tmp ^= lHT[rows3[j]];
			}

			// for every collision.
			typename HashMap1::load_type left_base_load;
			const size_t left_base_base = hm1->find(tmp, left_base_load);
			for (size_t j = left_base_base; j < left_base_base + left_base_load; j++) {
				const l_type tmp2 = tmp ^ hm1->ptr(j).data();
				for (uint32_t i = 0; i < p; ++i) {
					rows2[i] = hm1->ptr(j).index[i];
				}

				typename HashMap2::load_type left_load;
				const size_t left_base = hm2->find((tmp2>>l1), left_load);
				for (size_t v = left_base; v < left_base + left_load; v++) {
					const l_type tmp3 = tmp2 ^ hm2->ptr(v).data();
					for (uint32_t i = 0; i < 2*p; ++i) {
						rows[i] = hm2->ptr(v).index[i];
					}

					f(tmp2, tmp3, rows, rows+2, 1);
				}
			}
		}

		return false;
	}

	constexpr void print() const noexcept{
		hm1->print();
		hm2->print();
		std::cout << " {\"n\": " << n
				  << ", \"n/2\": " << n_half
				  << ", \"p\": " << p
				  << ", \"l1\": " << l1
				  << ", \"l2\": " << l2
				  << ", \"threads\": " << threads
				  << ", \"enumeration_size\": " << enumeration_size
		          << " }" << std::endl;
	}
};




template<const uint32_t d,
         const ConfigEnumHashMapD<d> &config,
         class ...HMs>
class CollisionHashMapD {
	static const std::size_t lenHMs = sizeof...(HMs);
	static_assert(lenHMs == d);

	constexpr static uint32_t n = config.n;
	constexpr static uint32_t n_half = n / 2u;
	constexpr static uint32_t p = config.p;
	constexpr static uint32_t threads = config.threads;

	// list stuff
	constexpr static size_t enumeration_length = n_half;
	constexpr static size_t enumeration_weigth = p;
	constexpr static size_t enumeration_size = bc(enumeration_length,enumeration_weigth);
	constexpr static size_t enumeration_size_per_thread = enumeration_size/threads;


	// TODO
	using l_type = uint32_t;

	const l_type *lHT, *lHTr;
	std::tuple<HMs&...> hms;

	alignas(32) std::array<l_type, 1u<<(d-1u)> iTs;
	alignas(32) uint16_t rows[p*(1u << d)];
	l_type sum = 0;
public:
	CollisionHashMapD(const l_type *lHT,
	                  HMs&...hms) noexcept :
	    lHT(lHT), lHTr(lHT + enumeration_length), hms(hms...) {

		sum = 0;
		for (uint32_t i = 0; i < ((1u<<(d-1u)) -1u); ++i) {
			iTs[i] = fastrandombytes_uint64() & ((1ul << config.ls[i]) - 1ul);
			sum ^= iTs[i];
		}
		iTs[((1u<<(d-1u)) -1u)] = sum;

	}

	template<typename ...Fs>
	bool step(const l_type iT, const uint32_t tid = 0, const bool simd=true, Fs& ...fs) {
		return coll_hashmap_simple(iT, tid, fs...);
	}

	/// computes the right list and start searching for collision
	/// NOTE: without NN
	/// NOTE: without AVX
	/// \tparam F
	/// \param iT intermediate target
	/// \param f lambda function to execute for each collision
	/// \return if found
	template<typename ...Fs>
	bool coll_hashmap_simple(const l_type iT, const uint32_t tid, Fs& ...fs) {
		// const size_t start = enumeration_size_per_thread * tid;
		// const size_t end   = tid == threads-1 ? enumeration_size : enumeration_size_per_thread * (tid + 1);

		coll_hashmap_simple_level<d, Fs...>(iT, rows, tid, fs...);
		return false;
	}

	template<const uint32_t dd, class ...Fs>
	bool coll_hashmap_simple_level(const l_type data,
	                               uint16_t *index,
	                               const uint32_t tid, Fs& ...fs) {
		if constexpr (dd == 1u) {
			//f(data, index);
			return false;
		} else {
			/// TODO iT und so

			// the min is needed to please the compiler. This is code is only executed if
			// d - dd < d
			constexpr uint32_t td = std::min(d - dd, d-1);
			using HM = std::tuple_element_t<td, std::tuple<HMs...>>;
			typename HM::load_type left_load;
			HM &hm = std::get<td>(hms);

			const size_t left_base = hm.find(data, left_load);
			for (uint64_t v = left_base; v < left_base + left_load; v++) {
				// const l_type tmp2 = data ^ hm.ptr(v).data();
				for (uint32_t i = 0; i < 2; ++i) { // TODO not correct, need to be an iterator (for appending)
					index[i] = hm.ptr(v).index[i];
				}

				//coll_hashmap_simple_level<dd - 1u>(tmp2, index, tid, f);
			}

			return false;
		}
	}

	void print() const noexcept{
		constexpr_for<0, d, 1>([this](auto i){
		  	std::get<i>(hms).print();
		});

		std::cout << "{ \"n\": " << n
		          << ", \"n/2\": " << n_half
		          << ", \"p\": " << p
				  << ", \"d\": " << d
		          << ", \"threads\": " << threads
		          << ", \"enumeration_size\": " << enumeration_size;

		for (uint32_t i = 0; i < d; i++) {
			std::cout << ", \"l\"" << i + 1<< ": " << config.ls[i];
		}

		std::cout << " }" << std::endl;
	}
};
#endif // DECODING_MITM_H
