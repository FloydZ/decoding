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

struct ConfigDumer {
public:
	const uint32_t n, k, w;                 // Instance parameters
	const uint32_t p, l, number_buckets;    // Optimize parameters

	// Number of threads working in parallel on the tree
	const uint32_t threads = 1;

	// number of bits we allow the two baselists to overlap
	const uint32_t epsilon = 0;

	// number of columns we cutoff, e.g. number of zeros we are guessing in the error.
	const uint32_t c = 0;

	// hashmap configuration
	const uint32_t number_bucket = l;   // log number of buckets used
	const uint32_t size_bucket = 1;     // number of elements each bucket can hold

	// Define the exact weight threshold. If the algorithm finds an element
	// which weight is <= the threshold we found the solution
	const uint32_t weight_threshold;

	// exit the program after X loops, regardless if a solution was found or not.
	#ifdef USE_LOOPS
	const uint64_t loops        = USE_LOOPS;
	#else
	const uint64_t loops        = uint64_t(-1);
	#endif

	// check every X loop if another thread already found the solution.
	const uint64_t exit_loops = 10000;

	// print after X loops.
	const uint64_t print_loops = 10000;

	// Number of rows we precompute in the M4RI step per block.
	const uint32_t m4ri_k = matrix_opt_k(n - k, MATRIX_AVX_PADDING(n));

	// allow the base lists to enumerate over the full length k+l and not
	// just (k+l)/2
	const bool Baselist_Full_Length = false;

	// enable decode one out of many instance
	const bool DOOM                 = false;

	// enable the low weight challenge code.
	const bool LOWWEIGHT            = false;

	// append `H1^n = w`
	const bool TrivialAppendRow     = false;

	// reduce the memory consumption of the program, by not save the values of the labels.
	const bool no_values            = false;

	// TODO alle HM flags
	const bool HM1_STDBINARYSEARCH_SWITCH       = true;
	const bool HM1_INTERPOLATIONSEARCH_SWITCH   = false;
	const bool HM1_LINEAREARCH_SWITCH           = false;
	const bool HM1_USE_LOAD_IN_FIND_SWITCH      = true;
	const bool HM1_SAVE_FULL_128BIT_SWITCH      = false;
	const bool HM1_USE_PREFETCH                 = false;
	const bool HM1_USE_ATOMIC_LOAD              = false;
	const bool HM1_USE_PACKED                   = true;

	// constructor
	constexpr ConfigDumer(const uint32_t n,         // code length
	                      const uint32_t k,         // code dimension
	                      const uint32_t w,         // code weight
	                      const uint32_t p,         // weight in the base lists
	                      const uint32_t l,         // bits to match on
	                      const uint32_t buckets,   // number of elements each bucket can hold
	                      const uint32_t thresh)    //
	    : n(n), k(k), w(w), p(p), l(l), number_buckets(buckets), weight_threshold(thresh)
	{}
	// prints information about the problem instance.
	void print() const {
		std::cout << "n: " << n
		          << ", k: " << k
		          << ", c: " << c
		          << ", p: " << p
		          << ", l: " << l
		          << ", DOOM: " << DOOM
		          << ", log(#buckets1): " << number_bucket
		          << ", size_bucket1: " << size_bucket
		          << ", threads: " << threads
		          << ", m4ri_k: " << m4ri_k
		          << ", weight_threshold: " << weight_threshold
		          << ", loops: " << loops
		          << ", print_loops: " << print_loops
		          << ", exit_loops: " << exit_loops
		          << ", epsilon: " << epsilon
		          << ", Baselist_Full_Length: " << Baselist_Full_Length
		          << ", TrivialAppendRow: " << TrivialAppendRow
		          << ", no_values: " << no_values
		          << ", HM1_STDBINARYSEARCH_SWITCH: " << HM1_STDBINARYSEARCH_SWITCH
		          << ", HM1_INTERPOLATIONSEARCH_SWITCH: " << HM1_INTERPOLATIONSEARCH_SWITCH
		          << ", HM1_LINEAREARCH_SWITCH: " << HM1_LINEAREARCH_SWITCH
		          << ", HM1_USE_LOAD_IN_FIND_SWITCH: " << HM1_USE_LOAD_IN_FIND_SWITCH
		          << ", HM1_SAVE_FULL_128BIT_SWITCH: " << HM1_SAVE_FULL_128BIT_SWITCH
		          << ", HM1_USE_PREFETCH: " << HM1_USE_PREFETCH
		          << ", HM1_USE_ATOMIC_LOAD: " << HM1_USE_ATOMIC_LOAD
		          << ", HM1_USE_PACKED: " << HM1_USE_PACKED
		          << "\n";
	}
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
	// │             I_n-k-l         │                  H                      │ │  s1  │
	// │                             │                                         │ │      │
	// │                             │                                         │ │      │
	// ├─────────────────────────────┼                                         ┤ ├──────┤ n-k-l
	// │              0              │                                         │ │  s2  │
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
	constexpr static uint32_t n = config.n;// code length
	constexpr static uint32_t k = config.k -config.TrivialAppendRow;
	constexpr static uint32_t w = config.w;// code weight
	constexpr static uint32_t p = config.p;// NOTE: from now on p is the baselist p not the full p.
	constexpr static uint32_t l = config.l;// number of bits to match on the first (and only) level.
	constexpr static uint32_t c = 0;       // number of columns to cut of the working matrix.
	constexpr static uint32_t nkl = n - config.k - l;

	// base type for holding `l` bits.
	using ArgumentLimbType = LogTypeTemplate<l>;
	using IndexType = TypeTemplate<(uint64_t(1) << config.number_bucket) * config.size_bucket>;
	using LoadType = IndexType;

	using DecodingValue = Value_T<BinaryContainer<k + l - c>>;                      // type to hold `e2`
	using DecodingLabel = Label_T<BinaryContainer<n-config.k>>;                     // type to hold `s`, the syndrome
	using DecodingMatrix = mzd_t *;                                                 // type to hold `H`
	using DecodingElement = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;// type of the tuple (He, e)
	using DecodingList = Parallel_List_T<DecodingElement>;                          // type of a list of tuples (He, 2)
	using ChangeList = std::vector<std::pair<uint16_t, uint16_t>>;                  // chase sequence

	// some internal abstract types
	typedef typename DecodingList::ValueContainerType ValueContainerType;
	typedef typename DecodingList::LabelContainerType LabelContainerType;
	typedef typename DecodingList::ValueContainerLimbType ValueContainerLimbType;
	typedef typename DecodingList::LabelContainerLimbType LabelContainerLimbType;

	// `threads` describes the number of threads that ar concurrently on the same tree.
	constexpr static uint32_t threads = config.threads;

	constexpr static uint32_t llimbs = LabelContainerType::limbs();      // number of `uint64_t` limbs needed to represent a `label`
	constexpr static uint32_t llimbs_a = LabelContainerType::bytes() / 8;// number of `uint64_t` limbs + alignment needed to hold a `label`

	constexpr static uint32_t lVCLTBits = sizeof(ValueContainerLimbType) * 8;// TODO simplify with ::limbs
	constexpr static uint32_t loffset = nkl / lVCLTBits;                     // limb in which the `l` bits are
	constexpr static uint32_t lshift = nkl - (loffset * lVCLTBits);          // number of bits to shift the `loffset` limb down
	                                                                         // to have the `l` bit in a register on bit 0.
	// precompute the list sizes depending on the given configuration.
	constexpr static size_t
	        lsize1 = config.Baselist_Full_Length ?
	                                             bc(k + l - c, p) :
	                                             bc(config.epsilon + ((k + l - c) / 2), p),
	        lsize2 = (config.Baselist_Full_Length & !config.DOOM) ?
	                                             bc(k + l - c - config.LOWWEIGHT, p) :
	                                             bc(config.epsilon + ((k + l - c - config.LOWWEIGHT) - (k + l - c) / 2), p);

	// list per thread size.
	constexpr static uint32_t tL1len = lsize1 / threads;
	constexpr static uint32_t tL2len = lsize2 / threads;

	mzd_t   *wH, // working matrix, e.g. the full n-k \times n matrix
	        *wHT,// working matrix transposed
	        *sT, // syndrome transposed, e.g. in row form
	        *H,  // n-k \time k+l submatrix of the working matrix.
	        *HT; // `H` transposed

	// current permutation applied to working matrix.
	mzp_t *permutation;
	// helper data needed to perform the gaussian elimination.
	customMatrixData *matrix_data;

	mzd_t *e;      // output: vector
	const mzd_t *s;// input: syndrome
	const mzd_t *A;// input: parity check matrix.

	// number of limbs between two `labels` in the list.
	constexpr static size_t BaseList4Inc = config.DOOM ? MATRIX_AVX_PADDING(n - k) / 64 : llimbs_a;
	constexpr static uint32_t DOOM_nr = config.DOOM ? k : 0;
	mzd_t *DOOM_S;     // if DOOM activated: matrix to all syndrome shifts
	mzd_t *DOOM_S_View;// submatrix of H containing all shifts of the syndrome

	mzd_t *trivial_row_row;

	// Changelist.
	ChangeList cL1, cL2;

	// Baselists. Holding values and labels, with label=H*value. The values are constant whereas the labels must
	// recompute for every permutation of the working matrix.
	DecodingList L1{lsize1, threads, tL1len},// left list
	             L2{lsize2, threads, tL2len};     // right list

	// this is a helper declaration. I heavily use this to reuse function declared in the BJMM base class
	constexpr static ConfigBJMM bjmmconfig{n, config.k, w, p, l, l, 1, 1, l, l, w - 4, threads, 1, !config.TrivialAppendRow /*reuse the DOOM field*/, false, 0, false, 0, 0, 1.0, config.no_values};

	// hashmap configuration
	constexpr static ConfigParallelBucketSort chm{0,                       // low bit to hash
	                                              0 + config.number_bucket,// high bit to hash
	                                              0 + l,                   // total number of bits in the hashmap
	                                              config.size_bucket,      // number of elements per bucket
	                                              uint64_t(1) << config.number_bucket,
	                                              threads, 1, n - config.k - l, l, 0, 0,// some helpers
	                                              true, false, false, true};

	// helper class which greatly simplifies the extracting l bits from a label.
	using Extractor = WindowExtractor<DecodingLabel, ArgumentLimbType>;

	// actual extractor function
	static inline ArgumentLimbType extractor(const DecodingLabel &label) noexcept {
		return extractor_ptr(label.ptr());
	};

	static inline ArgumentLimbType extractor_ptr(const uint64_t *label) noexcept {
		if constexpr(config.DOOM) {
			return Extractor::template extract<n-config.k-l, n-config.k>(label);
		}

		if constexpr (config.HM1_SAVE_FULL_128BIT_SWITCH) {
			return Extractor::template extract<n-config.k-128, n-config.k, n-config.k-l>(label);
		} else {
			return Extractor::template extract<n-config.k-l, n-config.k>(label);
		}
	};

	using HMType = ParallelBucketSort<chm, DecodingList, ArgumentLimbType, IndexType, &BJMM<bjmmconfig>::template Hash<0, config.l>>;
	using HMBucketIndexType = typename HMType::BucketIndexType;
	HMType *hm;

	// target, this is the syndrome we want to match on.
	DecodingLabel target;
	ArgumentLimbType iTarget;
	bool not_found = true;

	// number of indices we need to recompute a solutions
	constexpr static uint32_t npos_size = 2;

	uint64_t loops = 0;

	// measures the time without alle the preprocessing and allocating
	double internal_time;

	// constructor
	Dumer(mzd_t *e, const mzd_t *const s, const mzd_t *const A,
	      const uint32_t ext_tid = 0) noexcept
	    : e(e), s(s), A(A) {
		static_assert(n > k, "wrong dimension");
		static_assert(config.number_bucket <= l, "wrong hm1 #bucket");
		static_assert(threads <= config.size_bucket, "wrong #threads");
		static_assert(config.size_bucket % threads == 0, "wrong #threads");
		static_assert((config.DOOM + config.LOWWEIGHT) < 2, "DOOM AND LOWWEIGHT are not valid.");
		static_assert(((c != 0) + config.LOWWEIGHT) < 2, "CUTOFF AND LOWWEIGHT are not valid.");
		static_assert(!config.DOOM, "currently not implemted\n");

#if !defined(NO_LOGGING)
		config.print();
		bjmmconfig.print();
#endif

		// reset the `is a solution already found` flag
		not_found = true;

		// init the rng.
		srand(ext_tid + ext_tid * time(nullptr));
		random_seed(ext_tid + rand() * time(nullptr));

		// Ok this is ridiculous.
		// Apparently m4ris mzd_init is not thread safe. Cool.
		#pragma omp critical
		{
			// Transpose the input syndrome
			sT = mzd_init(s->ncols, s->nrows);
			mzd_transpose(sT, s);

			// DOOM stuff
			if constexpr (config.DOOM) {
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

				static_assert(n / 2 == k);
				DOOM_S = mzd_init(k, k);

				// Generate the DOOM shifts
				for (int i = 0; i < DOOM_nr; ++i) {
					matrix_down_shift_into_matrix(DOOM_S, sT, i, i);
				}

				wH = matrix_init(n - k, n + DOOM_nr);
				wHT = mzd_init(wH->ncols, wH->nrows);
				matrix_concat(wH, A, DOOM_S);

				H = mzd_init(n-config.k, k + l - c + DOOM_nr);
				HT = matrix_init(H->ncols, H->nrows);
				DOOM_S_View = mzd_init_window(HT, k + l - c, 0, k + l + DOOM_nr - c, HT->ncols);
			} else {
				wH = matrix_init(n-k, n + 1);
				wHT = mzd_init(wH->ncols, wH->nrows);
				mzd_t *tmp = matrix_concat(nullptr, A, sT);
				mzd_copy(wH, tmp);
				mzd_free(tmp);

				H = mzd_init(n-config.k, k + l - c);
				HT = matrix_init(H->ncols, H->nrows);
			}

			if constexpr (config.TrivialAppendRow) {
				wH = BJMM<bjmmconfig>::append_trivial_rows(wH);
			}

			// init the helper struct for the gaussian elimination and permutation data structs.
			matrix_data = init_matrix_data(wH->ncols);
			permutation = mzp_init(n - c);

			// Check if the working matrices are all allocated
			if ((wH == nullptr) || (wHT == nullptr) || (sT == nullptr) ||
			    (H == nullptr) || (HT == nullptr) || (permutation == nullptr) || (matrix_data == nullptr)) {
				std::cout << "ExtTID: " << ext_tid << ", alloc error2\n";
				exit(-1);
			}
		}

		// Init the target.
		target.zero();

		// init the hashmap
		hm = new HMType();

		// precalculate the values in both base lists
		if constexpr (config.Baselist_Full_Length) {
			static_assert(config.Baselist_Full_Length && (config.epsilon == 0));
			// TODO prepare_baselist_fulllength_mitm_with_chase2();
		} else {
			BJMM<bjmmconfig>::BJMM_prepare_generate_base_mitm2_with_chase2(L1, L2, cL1, cL2);
		}

		ASSERT(L1.size() == cL1.size());
	}


	/// TODO generalze BJMM version of this
	/// if this functions is called we found the solution.
	/// Because of this, this function is forced to not be inligned to reduce the
	/// instruction cache misses.
	/// \param label	final label computed in the tree.
	/// \param npos		positions of the elements in the baselists summing up to the label
	/// \param weight	weight of the final label
	/// \param DOOM_index2 helper value if DOOM activated.
	__attribute__((noinline))
	void check_final_list(LabelContainerType &label,
	                                                IndexType npos[npos_size],
	                                                const uint32_t weight,
	                                                const uint32_t DOOM_index2) noexcept {
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
				if constexpr (config.no_values == false) {
					ValueContainerType::add_withoutasm(value, L1.data_value(npos[0]).data(), L2.data_value(npos[1]).data());
				} else {

					value.zero();
					uint32_t P[p] = {0};

					// TODO doom
					for (uint32_t i = 0; i < 2; ++i) {
						BJMM<bjmmconfig>::get_bits_set(P, npos[i], i%2, cL1, cL2);
						for (uint32_t j = 0; j < p; ++j) {
							value.flip_bit(P[j]);
						}
					}

					std::cout << value << "\n";
				}

				// recompute the error vector by first setting the label and value at the correct
				// position and then apply the back permutation.
				for (int j = 0; j < n - c; ++j) {
					uint32_t bit;
					constexpr uint32_t limit = n - k - l;
					if (j < limit) {
						bit = label[j];
					} else {
						bit = value[j - limit];
					}
					mzd_write_bit(e, 0, permutation->values[j], bit);
				}


#if !defined(BENCHMARK) && !defined(NO_LOGGING)
				std::cout << " pre perm \n";
				std::cout << "weight input: " << weight << "\n";
				std::cout << "loops:" << loops << " found\n";

				std::cout << target << " target\n";
				std::cout << label << " label\n";
				std::cout << L1.data_label(npos[0]).data() << " npos[0]:" << unsigned(npos[0]) << "\n";
				std::cout << L2.data_label(npos[1]).data() << " npos[1]:" << unsigned(npos[1]) << "\n";

				std::cout << "\n"
				          << value << " value\n";
				std::cout << L1.data_value(npos[0]).data() << " npos[0]:" << unsigned(npos[0]) << "\n";
				std::cout << L2.data_value(npos[1]).data() << " npos[1]:" << unsigned(npos[1]) << "\n";
#endif

				mzd_print(e);
			}
#if NUMBER_THREADS != 1
		}
#endif
	}

	/// actual function which executes the algorithm
	/// \return the number of loops needed to find the solution
	uint64_t __attribute__((noinline))
	run() noexcept {
		// count the loops we iterated
		loops = 0;

		// we have to reset this value, so It's possible to rerun the algo more often
		not_found = true;

		// start the timer:
		internal_time = clock();

		while (not_found && loops < config.loops) {
			// start of the gaussian elemination phase
			matrix_create_random_permutation(wH, wHT, permutation);
			matrix_echelonize_partial_plusfix(wH, config.m4ri_k, n - k - l, this->matrix_data, 0, n - k - l, 0, this->permutation);
			mzd_submatrix(H, wH, config.TrivialAppendRow, n-k-l, n-k, n-c+DOOM_nr);
			matrix_transpose(HT, H);
			// end of the gaussian elimination phase

			// in the case we are not running DOOM we want to extract
			// the current syndrome from the parity check matrix.
			if constexpr (!config.DOOM) {
				target.data().column_from_m4ri(wH, n - c - config.LOWWEIGHT, 0);
				iTarget = extractor(target);
			}

			//TODO parallel phase
			{
				const uint32_t tid = threads != 1 ? omp_get_thread_num() : 0;

				// TODO liste soll das berechnen
				const uint64_t b_tid = lsize2 / threads;
				const uint64_t s_tid = tid * b_tid;
				const uint64_t e_tid = ((tid == threads - 1) ? L2.size() : s_tid + b_tid);
				IndexType npos[npos_size] = {IndexType(s_tid)};
				ArgumentLimbType data;
				LabelContainerType label, label2;
				uint64_t *Lptr;

				// init the
				for (int j = 0; j < npos_size; ++j) { npos[j] = s_tid; }

				// after everything was initializes fill the two baselists following the precomputed chase sequence.
				BJMM<bjmmconfig>::BJMM_fill_decoding_lists(L1, L2, cL1, cL2, HT, tid);

				// if needed, reset (overwrite everything with zeros) the hashmap
				hm->reset(tid);
				OMP_BARRIER

				// hash every element of the left base list into the hashmap.
				hm->hash(L1, tid);
				OMP_BARRIER

				// if needed sort the elements in the buckets.
				hm->sort(tid);
				OMP_BARRIER

				// sanity check if still is everything alright.
				ASSERT(hm->check_sorted());

				// set the base ptr of the right base list according to the
				// current thread number. This means we are splitting the base list
				// into #`threads` disjunctive parts where each part is worked through by
				// a different thread.
				if constexpr (config.DOOM) {
					Lptr = (uint64_t *) DOOM_S_View->rows[0] + (s_tid * llimbs_a);
				} else {
					Lptr = (uint64_t *) L2.data_label() + (s_tid * llimbs_a);
				}


				// set the maximum number of elements each thread can work on.
				uint64_t upper_limit;
				if constexpr (config.DOOM) {
					ASSERT(threads == 1);
					upper_limit = n - k;
				} else {
					upper_limit = e_tid;
				}

				// start the matching routine
				for (; npos[1] < upper_limit; ++npos[1], Lptr += BaseList4Inc) {
					// extract the l part of the current element of the right base list.
					data = extractor_ptr(Lptr);
					ASSERT((hm->check_label(data, L2, npos[1])));
					data ^= iTarget;

					// find a match in the left base list (in form of the hashmap)
					LoadType load1 = 0;// total number of element in the bucket we may find a collision
					IndexType pos1 = hm->find(data, load1);

					// If we are not in the DOOM case we can recompute the element on the full lenght
					// by simply adding the current element of the right base list with the target (=syndrome).
					// So this means if we found a match we can simply recompute the full final label (on which we have to perform the weight check on)
					// by adding the label of the left base list element into an temporary label. This saves one add per match.
					if constexpr (!config.DOOM) {
						LabelContainerType::add(label, L2.data_label(npos[1]).data(), target.data());
					}

					// for every match in the bucket.
					while (pos1 < load1) {
						// extract the index of the element in the left base list the match corresponds to.
						npos[0] = hm->__buckets[pos1].second[0];
						pos1 += 1;

						uint32_t weight;

						// calculate the weight of the final label in the tree.
						if constexpr (!config.DOOM) {
							weight = LabelContainerType::add_weight(label2, label, L1.data_label(npos[0]).data(), 0, n - k - l);
						} else {
							weight = LabelContainerType::add_weight(label2.data().data(), L1.data_label(npos[0]).data().data().data(), Lptr);
						}

						// check if we pass the weight check.
						if (unlikely(weight <= config.weight_threshold)) {
							check_final_list(label2, npos, weight, 0);
						}
					}
				}
			}

			// print onetime information
			if (unlikely(loops == 0)) {
				info();
			}

			// print periodic information
			if (((loops % config.print_loops) == 0)) {
				periodic_info();
			}

			// check if another thread found the solution already.
			OUTER_MULTITHREADED_WRITE(
					if ((unlikely(loops % config.exit_loops) == 0)) {
						if (finished.load()) {
							return loops;
						}
					})

			loops += 1;
		}

		return loops;
	}

	/// \param nn code length
	/// \param cc dimensions to cut off.
	/// \return the expected numbers of loops needed in log
	constexpr static double LogLoops(const uint64_t nn, const uint64_t cc = 0) noexcept {
		// note that p is here only the base p.
		// binom(n, w)/(binom(n-k-l, w-2p) * binom((k+l)/2, p)**2)
		const double nn_ = double(nn);
		const double k_ = double(k) / nn_;
		const double l_ = double(l) / nn_;
		const double p_ = double(p) / nn_;
		const double wp_ = double(w - (2. * p)) / nn_;
		const double w_ = double(w) / nn_;

		const double ccc = cc == 0 ? 0. : double(cc) / nn_ / 2.;

		const double t1 = double(k_ + l_) / 2. - ccc;
		const double t2 = double(1. - k_ - l_);

		const double t1_ = HH(p_ / t1);
		const double t2_ = HH(wp_ / t2);

		const double t11 = t1 * t1_;
		const double t22 = t2 * t2_;

		const double t3 = HH(w_) - 2. * t11 - t22;
		const double shift = t3 * nn;
		return shift;
	}

	/// same `LogLoops` only returns the total number of loops not in log form
	constexpr static double Loops(const uint64_t nn = n, const uint64_t cc = 0) noexcept {
		return pow(2, LogLoops(nn, cc));
	}

	/// returns the expected size of the baselists and the out list.
	constexpr static std::array<size_t, 2> ListSizes() noexcept {
		size_t S1 = (lsize2 * lsize2) >> l;
		std::array<size_t, 2> ret{lsize2, S1};
		return ret;
	}


	/// print one time information about the state of the programm.
	void __attribute__((noinline))
	info() noexcept {
#if !defined(BENCHMARK) && !defined(NO_LOGGING)
		std::cout << "BJMMF: tid:" << omp_get_thread_num() << ", loops: " << loops << "\n";
		std::cout << "log(inner_loops): " << LogLoops(n, c) << ", inner_loops: " << Loops(n, c);

		std::cout << "|L_1|: " << this->L1.size() << "\n";
		std::cout << "|L_1|: " << this->L1.bytes() / (1 << 20) << "MB\n";
		std::cout << "|L_1|+|L_2|: " << (this->L1.bytes() + this->L2.bytes()) / (1 << 20) << "MB\n";
		double ctime = ((double) clock() - internal_time) / CLOCKS_PER_SEC;
		std::cout << "Time: " << ctime << ", clock Time: " << ctime / config.threads << "\n";
		hm->print();
		auto LSizes = ListSizes();
		std::cout << "Exp |hm1|=" << LSizes[0] << ", Exp |out|=" << LSizes[1] << "\n"
		          << std::flush;
#endif
	}

	/// prints current loops information like: Hashmap usage, time, loops, ...
	/// This function is intentionally not inlined to reduce the pressure on the instruction cache.
	void __attribute__((noinline))
	periodic_info() noexcept {
#if !defined(NO_LOGGING)
		double ctime = (((double) clock()) - internal_time) / CLOCKS_PER_SEC;
		std::cout << "\rcurrently at " << loops << " loops, " << ctime << "s, " << loops / ctime << "lps" << std::flush;
#endif
	}
};

#endif //SMALLSECRETLWE_DUMER_H
