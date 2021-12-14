#ifndef SMALLSECRETLWE_DUMER_H
#define SMALLSECRETLWE_DUMER_H

#include "helper.h"
#include "glue_m4ri.h"
#include "custom_matrix.h"
#include "combinations.h"
#include "bjmm.h"
#include "list.h"
#include "list_fill.h"
#include "sort.h"

#include <type_traits>
#include <algorithm>
#include <omp.h>

// Allows efficient and easy access to the l part of the label
template<typename ArgumentLimbType, typename ValueContainerLimbType, uint32_t loffset, uint32_t lshift, uint32_t lVCLTBits,
        uint32_t nk, uint32_t nkl>
struct MemoryAccess {
	// only one limb
	static ArgumentLimbType add_bjmm_l11(ValueContainerLimbType *v3, ValueContainerLimbType const *v1) {
		constexpr static ValueContainerLimbType mask = ~((ValueContainerLimbType(1) << (nkl % lVCLTBits)) - 1);

		v3[loffset] ^= (v1[loffset] & mask);
		return v3[loffset] >> lshift; // IMPORTANT AUTO CAST HERE
	}

	static ArgumentLimbType add_bjmm_l22(ValueContainerLimbType *v3, ValueContainerLimbType const *v1) {
		constexpr static ValueContainerLimbType lmask = ~((ValueContainerLimbType(1) << (nkl % lVCLTBits)) - 1);

		v3[loffset]   ^= (v1[loffset] & lmask);
		v3[loffset+1] ^=  v1[loffset+1];
		// NOTE this must be generalised to 2*ValueContainerLimbType, if you want to be able to use l > 64
		__uint128_t data = v3[loffset];
		data            += (__uint128_t(v3[loffset+1]) << 64);
		return data >> lshift;  // IMPORTANT AUTO CAST HERE
	}

	static ArgumentLimbType add_bjmm(ValueContainerLimbType *v3, ValueContainerLimbType const *v1) {
		if constexpr ((nkl / lVCLTBits) == ((nk - 1) / lVCLTBits)) {
			// if the two limits are in the same limb
			return add_bjmm_l11(v3, v1);
		} else {
			// the two limits are in two different limbs
			return add_bjmm_l22(v3, v1);
		}
	}
};

struct ConfigDumer {
public:
	const uint32_t n, k, w;                 // Instance parameters
	const uint32_t p, l, number_buckets;    // Optimize parameters
	const uint32_t threads = 1;
	const uint32_t epsilon = 0;
	const uint32_t c = 0;


	const uint32_t number_bucket = 1;
	const uint32_t size_bucket = 1;

	const uint32_t weight_threshhold;
	const uint64_t loops = -1;
	const uint32_t m4ri_k = 4;

	const bool Baselist_Full_Length = false;
	const bool DOOM = false;
	const bool LOWWEIGHT = false;

	constexpr ConfigDumer(uint32_t n, uint32_t k, uint32_t w, uint32_t p, uint32_t l, uint32_t buckets, uint32_t thresh) :
			n(n), k(k), w(w), p(p), l(l), number_buckets(buckets), weight_threshhold(thresh)
	{}

};

template<const ConfigDumer &config>
class Dumer {
public:
// Algorithm Picture:
//                            n-k-l                                        n          0
// ┌─────────────────────────────┬─────────────────────────────────────────┐ ┌──────┐
// │                             │                                         │ │      │
// │                             │                                         │ │      │
// │                             │                                         │ │      │
// │             I_n-k-l         │                  H                      │ │  0   │
// │                             │                                         │ │      │
// │                             │                                         │ │      │
// ├─────────────────────────────┼─────────────────────────────────────────┤ ├──────┤ n-k-l
// │              0              │                                         │ │      │
// └─────────────────────────────┴─────────────────────────────────────────┘ └──────┘  n-k
//              w-p                                  p
// ┌─────────────────────────────┬─────────────────────────────────────────┐
// │              e1             │                    e2                   │
// │                             │                                         │
// └─────────────────────────────┴────────────────────┬────────────────────┘
//                                                    │
//                                              ┌─────┴────┐              Match on l
//                                          ┌───┴───┐   ┌──┴────┐
//                                          │       │   │       │
//                                          │       │   │       │
//                                          │       │   │       │
//                                          │       │   │       │
//                                          │       │   │       │
//                                          │       │   │       │
//                                          └───────┘   └───────┘

	// constants from the problem instance
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t k = config.k;
	constexpr static uint32_t w = config.w;
	constexpr static uint32_t p = config.p;                    // NOTE: from now on p is the baselist p not the full p.
	constexpr static uint32_t l = config.l;
	constexpr static uint32_t c = 0;
	constexpr static uint32_t nkl = n - config.k - l;

	using ArgumentLimbType  = LogTypeTemplate<l>;
	using IndexType         = TypeTemplate<(uint64_t(1) << config.number_bucket) * config.size_bucket>;
	using LoadType          = IndexType;

	using DecodingValue     = Value_T<BinaryContainer<k + l - c>>;
	using DecodingLabel     = Label_T<BinaryContainer<n - k>>;
	using DecodingMatrix    = mzd_t *;
	using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList      = Parallel_List_T<DecodingElement>;
	using ChangeList        = std::vector<std::pair<uint16_t, uint16_t>>;

	typedef typename DecodingList::ValueContainerType ValueContainerType;
	typedef typename DecodingList::LabelContainerType LabelContainerType;
	typedef typename DecodingList::ValueContainerLimbType ValueContainerLimbType;
	typedef typename DecodingList::LabelContainerLimbType LabelContainerLimbType;


	constexpr static uint32_t threads = config.threads;

	constexpr static uint32_t llimbs    = LabelContainerType::limbs();
	constexpr static uint32_t llimbs_a  = LabelContainerType::bytes() / 8;

	constexpr static uint32_t lVCLTBits = sizeof(ValueContainerLimbType)*8;
	constexpr static uint32_t loffset   = nkl / lVCLTBits;
	constexpr static uint32_t lshift    = nkl - (loffset * lVCLTBits);


	constexpr static uint64_t
			lsize1 = config.Baselist_Full_Length ?
			         bc(k+l-c, p) :
			         bc(config.epsilon + ((k + l - c) / 2), p),
			lsize2 = (config.Baselist_Full_Length & !config.DOOM) ?
			         bc(k+l-c-config.LOWWEIGHT, p) :
			         bc(config.epsilon + ((k + l - c -config.LOWWEIGHT) - (k + l - c) / 2), p);

	// list per thread size.
	constexpr static uint32_t tL1len = lsize1 / threads;
	constexpr static uint32_t tL2len = lsize2 / threads;

	mzd_t *work_matrix_H, *work_matrix_H_T, *sT, *H, *HT;
	mzp_t *permutation;
	customMatrixData *matrix_data;

	mzd_t *e;
	const mzd_t *s;
	const mzd_t *A;

	constexpr static uint64_t BaseList4Inc = config.DOOM ? MATRIX_AVX_PADDING(n - k) / 64 : llimbs_a;
	constexpr static uint32_t DOOM_nr = config.DOOM ? k : 0;
	mzd_t * DOOM_S;
	mzd_t * DOOM_S_View;

	ChangeList cL1, cL2;

	// Baselists. Holding values and labels, with label=H*value. The values are constant where as the labels must
	// recomputed for every permutation of the working matrix.
	DecodingList L1{lsize1, threads, tL1len}, L2{lsize2, threads, tL2len};

	constexpr static ConfigParallelBucketSort chm{0, 0 + config.number_bucket, 0 + config.number_bucket, config.size_bucket,
	                                               uint64_t(1) << config.number_bucket, threads, 1, n - config.k - l, l, 0, 0,
	                                               true, false, false, true};

	using MemAccess = MemoryAccess<ArgumentLimbType, ValueContainerLimbType, loffset, lshift, lVCLTBits, n-k, n-k-l>;

	template<const uint32_t l, const uint32_t h>
	static ArgumentLimbType Hash(uint64_t a) {
		return 0;
	}
	using HMType = ParallelBucketSort<chm, DecodingList, ArgumentLimbType, IndexType, &Hash<0, config.l>>;
	using HMBucketIndexType = typename HMType::BucketIndexType;
	HMType *hm;

	LabelContainerType target;
	bool not_found = true;

	constexpr static uint32_t npos_size = 2;

	Dumer(mzd_t *e, const mzd_t *const s, const mzd_t *const A, const uint32_t ext_tid = 0, const bool init_=true)
	: e(e), s(s), A(A) {
		static_assert(n > k, "wrong dimension");
		static_assert(config.number_bucket <= l, "wrong hm1 #bucket");

		static_assert(threads <= config.size_bucket, "wrong #threads");
		static_assert(config.size_bucket% threads == 0, "wrong #threads");
		static_assert((config.DOOM + config.LOWWEIGHT) < 2, "DOOM AND LOWWEIGHT are not valid.");
		static_assert(((c != 0) + config.LOWWEIGHT) < 2, "CUTOFF AND LOWWEIGHT are not valid.");
		not_found = true;

		srand(ext_tid + ext_tid * time(nullptr));
		random_seed(ext_tid + rand() * time(nullptr));
#if defined(NUMBER_OUTER_THREADS) && NUMBER_OUTER_THREADS != 1
		// Ok this is ridiculous.
		// Apparently m4ris mzd_init is not thread safe. Cool.
#pragma omp critical
		{
#endif
		// Transpose the input syndrome
		sT = mzd_init(s->ncols, s->nrows);
		mzd_transpose(sT, s);

		// DOOM stuff
		if constexpr(config.DOOM) {
			// Note: k = n//2
			//                n-k-l  n-k                                 n                      n+k
			// ┌────────────────┬─────┬────────────────────────────────────┬───┬───┬─────────┬───┐
			// │                │     │h_0    h_1  h_2     ...        h_k-1│   │   │         │   │
			// │                │     │h_k-1  h_0  h_1     ...         h_0 │   │   │         │   │
			// │                │  0  │h_k-2  h_k-1        ...             │ s │   │         │   │
			// │                │  or │h_k-3  ...                          │ = │   │         │   │
			// │    I_n-n/2-l   │  ┼  │                                    │ s0│ s1│   ...   │s_k-1
			// │                │     │                                    │   │   │    |    │   │
			// │                │     │         --> right shifts           │   │   │    ▼    │   │
			// │ ---------------│-----│                                    │   │   │Down     │   │
			// │                │  I  │                                    │   │   │  Shifts │   │
			// └────────────────┴─────┴────────────────────────────────────┴───┴───┴─────────┴───┘ n-k=n//2
			//    ◄──────────────────       ◄────────────────────────────
			//              Both Blocks cyclic left shifts to receive the original error vector.

			static_assert(n/2 == k);
			DOOM_S = mzd_init(k, k);

			// Generate the DOOM shifts
			for (int i = 0; i < DOOM_nr; ++i) {
				matrix_down_shift_into_matrix(DOOM_S, sT, i, i);
			}

			if constexpr(c == 0) {
				work_matrix_H = matrix_init(n - k, n + DOOM_nr);
				work_matrix_H_T = mzd_init(work_matrix_H->ncols, work_matrix_H->nrows);
				matrix_concat(work_matrix_H, A, DOOM_S);
			} else {
//				outer_matrix_H  = matrix_init(n - k, n + 1);
//				outer_matrix_HT = mzd_init(outer_matrix_H->ncols, outer_matrix_H->nrows);
//				matrix_concat(outer_matrix_H, A, sT);
//
//				work_matrix_H  = matrix_init(n - k, n - c + DOOM_nr + 1);
//				work_matrix_H_T = mzd_init(work_matrix_H->ncols, work_matrix_H->nrows);
			}

			H  = mzd_init(n - k, k + l - c + DOOM_nr);
			HT = matrix_init(H->ncols, H->nrows);
			DOOM_S_View = mzd_init_window(HT, k+l-c, 0, k+l+DOOM_nr-c, HT->ncols);
		} else {
			if constexpr(c == 0) {
				work_matrix_H   = matrix_init(n - k, n+1);
				work_matrix_H_T = mzd_init(work_matrix_H->ncols, work_matrix_H->nrows);
				mzd_t *tmp = matrix_concat(nullptr, A, sT);
				mzd_copy(work_matrix_H, tmp);
				mzd_free(tmp);
			} else {
//				// init all matrix structures
//				outer_matrix_H  = matrix_init(n - k, n+1);
//				outer_matrix_HT = mzd_init(outer_matrix_H->ncols, outer_matrix_H->nrows);
//				matrix_concat(outer_matrix_H, A, sT);
//
//				work_matrix_H  = matrix_init(n - k, n - c + 1);
//				work_matrix_H_T = mzd_init(work_matrix_H->ncols, work_matrix_H->nrows);
			}

			H  = mzd_init(n - k, k + l - c);
			HT = matrix_init(H->ncols, H->nrows);
		}

		// init the helper struct for the gaussian eleimination and permutation data structs.
		matrix_data = init_matrix_data(work_matrix_H->ncols);
		permutation = mzp_init(n-c);

		// Check if the working matrices are all allocated
		if ((work_matrix_H == nullptr)|| (work_matrix_H_T == nullptr) || (sT == nullptr) ||
		    (H == nullptr) || (HT == nullptr) || (permutation == nullptr) || (matrix_data == nullptr)) {
			std::cout << "ExtTID: " << ext_tid << ", alloc error2\n";
			exit(-1);
		}

#if defined(NUMBER_OUTER_THREADS) && NUMBER_OUTER_THREADS != 1
		}
#endif

		// Init the target.
		target.zero();
		hm = new HMType();

		auto list_precompute_time = (double)clock();
		if constexpr(config.Baselist_Full_Length) {
			static_assert(config.Baselist_Full_Length && (config.epsilon == 0));
			// TODO prepare_baselist_fulllength_mitm_with_chase2();
		} else {
			BJMM_prepare_generate_base_mitm2_with_chase2(L1, L2, cL1, cL2);
		}

		ASSERT(L1.size() == cL1.size());
	}

	template<const uint8_t limbs>
	inline void xor_helper(uint64_t *a, const uint64_t *b) {
		// avx2 optimisation is hopefully directly done by the compiler.
		LOOP_UNROLL();
		for (int i = 0; i < limbs; ++i) {
			a[i] ^= b[i];
		}
	};

	template<class Label>
	void xor_helper(Label &a, const word *b, const uint64_t limbs) {
		ASSERT(a.data().limbs() == limbs);
		unsigned int i = 0;

		for (; i < limbs; ++i) {
			a.data().data()[i] ^= b[i];
		}
	};

	template<typename DecodingList=DecodingList>
	void BJMM_prepare_generate_base_mitm2_with_chase2(DecodingList &bL1, DecodingList &bL2,
	                                                  ChangeList &cL1, ChangeList &cL2) {
		typedef typename DecodingList::ValueType Value;
		typedef typename DecodingList::ValueContainerType ValueContainerType;
		typedef typename DecodingList::ValueContainerType::ContainerLimbType VCLT;

		Value e11{}, e12{}, e21{}, e22{};
		e11.zero(); e12.zero(); e21.zero(); e22.zero();



		// Note: if TrivialAppendRows != 0: config.k  != k
		constexpr uint32_t n_full = config.k + config.l - config.c;
		constexpr uint32_t n = n_full / 2;
		constexpr uint32_t p = config.p;
		constexpr uint32_t epsilon = config.epsilon;
		constexpr uint32_t limbs = ValueContainerType::limbs();

		// resize the data.
		constexpr uint64_t lsize1 = bc(n + epsilon, p);
		constexpr uint64_t lsize2 = bc(n_full - n + epsilon - config.LOWWEIGHT, p);
		cL1.resize(lsize1);
		cL2.resize(lsize2);

		ASSERT(cL1.size() == bL1.size() && cL2.size() == bL2.size());

		Combinations_Chase_Binary<VCLT> ccb_l1{n + epsilon, p, 0};
		Combinations_Chase_Binary<VCLT> ccb_l2{n_full - config.LOWWEIGHT, p, n - epsilon};
		ccb_l1.left_init(e11.data().data().data());
		ccb_l1.left_step(e11.data().data().data(), true);
		ccb_l2.left_init(e21.data().data().data());
		ccb_l2.left_step(e21.data().data().data(), true);

		bL1.data_value(0) = e11;
		bL2.data_value(0) = e21;

		uint16_t pos1, pos2;

		auto diff_index = [&](const Value &a, const Value &b) {
			Combinations_Chase_Binary<VCLT>::diff(a.data().data().data(),
			                                      b.data().data().data(),
			                                      limbs, &pos1, &pos2);
		};

		uint64_t c1 = 1, c2 = 1;
		while (c1 < lsize1 || c2 < lsize2) {
			if (c1 < lsize1) {
				e12 = e11;
				ccb_l1.left_step(e11.data().data().data());
				diff_index(e11, e12);
				bL1.data_value(c1) = e11;
				cL1[c1 - 1] = std::pair<uint32_t, uint32_t>(pos1, pos2);
				c1 += 1;
			}

			if (c2 < lsize2) {
				e22 = e21;
				ccb_l2.left_step(e21.data().data().data());
				diff_index(e21, e22);
				bL2.data_value(c2) = e21;
				cL2[c2 - 1] = std::pair<uint32_t, uint32_t>(pos1, pos2);
				c2 += 1;
			}
		}
	}

	void BJMM_fill_decoding_lists(DecodingList &L1, DecodingList &L2,
	                              const ChangeList &v1, const ChangeList &v2,
	                              const mzd_t *HT, const uint32_t tid) {
		const uint32_t start1   = tid * tL1len;   // starting index within the list L1 of each thread
		const uint32_t start2   = tid * tL2len;   // starting index within the list L2 of each thread
		const uint32_t end1     = tid == (threads - 1) ? lsize1 : start1 + tL1len;  // ending index of each thread within the list L1,
		const uint32_t end2     = tid == (threads - 1) ? lsize2 : start2 + tL2len;  // exepct for the last thread. This needs
		const uint32_t end = MAX(end1, end2);                                       // process until the end of the lists.

		uint64_t *Lptr1 = (uint64_t *) L1.data_label() + (start1 * llimbs_a);
		uint64_t *Lptr2 = (uint64_t *) L2.data_label() + (start2 * llimbs_a);


		ASSERT(end > 0 );
		ASSERT(llimbs <= HT->width);
		ASSERT(tid < config.threads);
		uint32_t P1[p] = {0};
		uint32_t P2[p] = {0};

		// extract the bits currently set in the value
		L1.data_value()[start1].data().get_bits_set(P1, p);
		L2.data_value()[start2].data().get_bits_set(P2, p);

		// prepare the first element
		L1.data_label()[start1].zero();
		L2.data_label()[start2].zero();
		for (int i = 0; i < p; ++i) {
			xor_helper<DecodingLabel>(L1.data_label()[start1], HT->rows[P1[i]], llimbs);
			xor_helper<DecodingLabel>(L2.data_label()[start2], HT->rows[P2[i]], llimbs);
		}

		for (uint32_t i = MIN(start1 + 1, start2 + 1);
		     i < MAX(end1, end2); i++) {
			if ((i >= start1+1) && (i < end1)) {
				ASSERT(v1[i].first < HT->nrows && v1[i].second < HT->nrows);

				Lptr1 += llimbs_a;
				L1.data_label(i) = L1.data_label(i - 1);
				xor_helper<llimbs>(Lptr1, HT->rows[v1[i - 1].first]);
				xor_helper<llimbs>(Lptr1, HT->rows[v1[i - 1].second]);
			}
			if ((i >= start2+1) && (i < end2)) {
				ASSERT(v2[i].first < HT->nrows && v2[i].second < HT->nrows);

				Lptr2 += llimbs_a;
				L2.data_label(i) = L2.data_label(i - 1);
				xor_helper<llimbs>(Lptr2, HT->rows[v2[i - 1].first]);
				xor_helper<llimbs>(Lptr2, HT->rows[v2[i - 1].second]);
			}
		}
	}

	__attribute__((noinline))
	void check_final_list(LabelContainerType &label,
	                      IndexType npos[4],
	                      const uint32_t weight,
	                      const uint32_t DOOM_index2) {
#if NUMBER_THREADS != 1
			#pragma omp critical
			{
#endif
			// make really sure that only one thread every runs this code.
			if (not_found) {
#if NUMBER_THREADS != 1
				#pragma omp atomic write
				not_found = false;
				#pragma omp flush(not_found)
#else
				not_found = false;
#endif

				ValueContainerType value;
				ValueContainerType::add_withoutasm(value, L1.data_value(npos[0]).data(), L2.data_value(npos[1]).data());

				uint64_t ctr1 = 0;
				uint64_t ctr2 = 0;

				// recompute the error vector by first setting the label and value at the correct
				// position and then apply the back permutation.
				for (int j = 0; j < n - c; ++j) {
					uint32_t bit;
					constexpr uint32_t limit = n - k - l;
					if (j < limit) {
						bit = label[j];
						ctr1 += bit;
					} else {
						bit = value[j - limit];
						ctr2 += bit;
					}
					mzd_write_bit(e, 0, permutation->values[j], bit);
				}


#ifndef BENCHMARK
				std::cout << " pre perm \n";
				std::cout << "weight n-k-l:" << ctr1 << "\n";
				std::cout << "weight tree: " << ctr2 << "\n";
				std::cout << "weight input: " << weight << "\n";
				std::cout << "hashmap " << DOOM_index2 << " found\n";

				std::cout << target << " target\n";
				std::cout << label << " label\n";
				std::cout << L1.data_label(npos[0]).data() << " npos[0]:" << unsigned(npos[0]) << "\n";
				std::cout << L2.data_label(npos[1]).data() << " npos[1]:" << unsigned(npos[1]) << "\n";

				std::cout << "\n" << value << " value\n";
				std::cout << L1.data_value(npos[0]).data() << " npos[0]:" << unsigned(npos[0]) << "\n";
				std::cout << L2.data_value(npos[1]).data() << " npos[1]:" << unsigned(npos[1]) << "\n";
#endif


				// if we are in the QC setting we may have to reshift the error vector.
				if constexpr(config.DOOM) {
					const uint32_t DOOM_index = npos[3];
					static_assert(n/2 == k);
					if (DOOM_index != 0) {
						mzd_t *tmpe = mzd_init(1, n);

						// circular left shift
						for (int i = 0; i < k; ++i) {
							mzd_write_bit(tmpe, 0, i, mzd_read_bit(e, 0, ((i+DOOM_index)%k)));
						}
						for (int i = 0; i < k; ++i) {
							mzd_write_bit(tmpe, 0, k+i, mzd_read_bit(e, 0, k+((i+DOOM_index)%k)));
						}

						mzd_copy_row(e, 0, tmpe, 0);
						mzd_free(tmpe);
					}
				}
			}
#if NUMBER_THREADS != 1
			}
#endif

	}

	void run(){

		// count the loops we iterated
		uint64_t loops = 0;

		// we have to reset this value, so It's possible to rerun the algo more often
		not_found = true;
		while (not_found && loops < config.loops) {
			matrix_create_random_permutation(work_matrix_H, work_matrix_H_T, permutation);
			matrix_echelonize_partial_plusfix(work_matrix_H, config.m4ri_k, n-k-l, this->matrix_data, 0, n-k-l, 0, this->permutation);
			mzd_submatrix(H, work_matrix_H, 0, n - k - l, n - k, n - c + DOOM_nr);
			matrix_transpose(HT, H);

			Matrix_T<mzd_t *> HH((mzd_t *) H);
			if constexpr(!config.DOOM) {
				target.column_from_m4ri(work_matrix_H, n-c-config.LOWWEIGHT, 0);
			}

			{
				const uint32_t tid = NUMBER_THREADS != 1 ? omp_get_thread_num() : 0;
				const uint64_t b_tid = lsize2 / threads;
				const uint64_t s_tid = tid * b_tid;
				const uint64_t e_tid = ((tid == threads - 1) ? L2.size() : s_tid + b_tid);
				IndexType npos[npos_size] = {IndexType(s_tid)};
				IndexType pos1;
				LoadType load1 = 0;
				ArgumentLimbType data;
				LabelContainerType label, label2;
				uint64_t *Lptr;

				for (int j = 0; j < npos_size; ++j) { npos[j] = s_tid; }

				BJMM_fill_decoding_lists(L1, L2, cL1, cL2, HT, tid);

				hm->reset(tid);
#if NUMBER_THREADS != 1
#pragma omp barrier
#endif

				hm->hash(L1, tid);
#if NUMBER_THREADS != 1
#pragma omp barrier
#endif
				hm->sort(tid);
#if NUMBER_THREADS != 1
#pragma omp barrier
#endif
				ASSERT(hm->check_sorted());


				if constexpr(config.DOOM) {
					Lptr = (uint64_t *) DOOM_S_View->rows[0] + (s_tid * llimbs_a);
				} else {
					Lptr = (uint64_t *) L2.data_label() + (s_tid * llimbs_a);
				}


				uint64_t upper_limit;
				if constexpr(config.DOOM) {
					ASSERT(threads == 1);
					upper_limit = n - k;
				} else {
					upper_limit = e_tid;
				}

				for (; npos[1] < upper_limit; ++npos[1], Lptr += BaseList4Inc) {
					if constexpr(config.DOOM) {
						//data = add_bjmm_oe(Lptr);
					} else {
						data = MemAccess::add_bjmm(Lptr, target.ptr());
						ASSERT(hm->check_label(data, L2, npos[1]));
					}

					pos1 = hm->find(data, load1);

					if constexpr(!config.DOOM) {
						LabelContainerType::add(label, L2.data_label(npos[1]).data(), target);
					}

					while (pos1 < load1) {
						npos[0] = hm->__buckets[pos1].second[0];
						pos1 += 1;

						uint32_t weight;

						if constexpr(!config.DOOM) {
							weight = LabelContainerType::add_weight(label2, label, L1.data_label(npos[0]).data(), 0, n-k-l);
						} else {
							weight = LabelContainerType::add_weight(label2.data().data(), L1.data_label(npos[0]).data().data().data(), Lptr);
						}


						if (weight <= config.weight_threshhold){
							check_final_list(label2, npos, weight, 0);
						}
					}
				}
			}

#if NUMBER_THREADS != 1
#pragma omp barrier
#endif

#if (defined(NUMBER_OUTER_THREADS) && NUMBER_OUTER_THREADS > 1)
				if ((unlikely(loops % config.exit_loops) == 0)) {
					if (finished.load()) {
						return loops;
					}
				}
#endif
				loops += 1;
		}

	}
};


#endif //SMALLSECRETLWE_DUMER_H
