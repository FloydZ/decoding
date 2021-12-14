#ifndef SMALLSECRETLWE_BJMM_H
#define SMALLSECRETLWE_BJMM_H

#include "helper.h"
#include "glue_m4ri.h"
#include "custom_matrix.h"
#include "combinations.h"
#include "list.h"
#include "sort.h"

#include <type_traits>
#include <algorithm>
#include <omp.h>

// TODO remove
double inittime=0;

struct ConfigBJMM {
public:
	const uint32_t n, k, w;     // instance Parameters
	// this is the weight on the baselists.
	const uint32_t baselist_p;
	// Stern/Dumer window
	const uint32_t l, l1;

	// at what weight on the first n-k-l coordinates do we classify a label as a solution.
	const uint32_t weight_threshhold = w - 4 * baselist_p;

	const int m4ri_k; // opt parameter for faster gaus elimination
					  // calculated on the fly. You normally don't have to touch this.

	const uint32_t nr_threads;

	// Overlapping of the two baselists halves
	// BaseList size tuning parameter
	const uint32_t epsilon = 0;

	// how many iterations
	const uint32_t intermediate_target_loops = 1;

	// Scaling factor for the different sorting/searching datastructures
	const float scale_bucket = 1.0;

	// Number of Elements to store in each bucket within each hash map.
	const uint64_t size_bucket1;
	const uint64_t size_bucket2;

	// Number of buckets in each hashmap.
	const uint64_t number_bucket1;
	const uint64_t number_bucket2;

	const bool ClassicalTree = false;

	// Decode One Out of Many.
	// Only usable if you want to break QuasiCyclic code.
	// Look at the BJMM constructor to see a detailed graphic explaining everything.
	// This enables:
	//  - all shifts of the syndrome are appended to the working matrix.
	//  - No target will be added to the baselists
	//  - No intermediate target will be used
	const bool DOOM = false;
	// Decode One Out of Many special Tree,
	// See `BJMM::special_doom_tree()` for code and detailed graphs.
	const bool DOOM_Alternative_Tree = false;

	// Instead of an MITM approach, enumerate the baselists on the full length k+l
	const bool Baselist_Full_Length = false;

	// Enable this flag if you want to solve Low Weight challenges: https://decodingchallenge.org/low-weight
	// This will disable:
	//  - Intermediate Target
	// This will enable:
	//  - Last bit of e will be always one
	const bool LOWWEIGHT = false;

	// Cutoff parameter `c`. if this parameter is set to anything else as zero, this many coordinates will be cutoff
	// from the working matrix. E.g. zero coordinates will be guessed in e.
	// This decreases the list size, by decreasing the error vector size.
	// Not that a special `_outer()` function must be called instead of `BJMM()`. This function will take care of all
	// the randomisation and mem-copies needed for this,
	const uint32_t c;
	const uint32_t c_inner_loops = 1000; // Loops(n-c)

	// CURRENTLY, UNUSED
	// Depth Parameter, is only active if d==3
	const uint32_t d = 0;


	// Indyk Motwani Nearest Neighbor Approach. Well technically its only NN in the last level so: May-Ozerov
	// See the class `BJMMNN` for implementation details and graphs explaining what's going on.
	// if this value is set to anything else as zero, this many hashmaps (holding `l2` bits per HM) will be used in
	// the last/final level of the tree.
	const uint32_t IM_nr_views;
	const uint32_t l2 = 0;

	// TODO describe
	const uint64_t number_bucket3 = l2;
	const uint64_t size_bucket3 = 10;//1000;

	// further HM flags
	const bool HM1_STDBINARYSEARCH_SWITCH       = true;
	const bool HM1_INTERPOLATIONSEARCH_SWITCH   = false;
	const bool HM1_LINEAREARCH_SWITCH           = false;
	const bool HM1_USE_LOAD_IN_FIND_SWITCH      = true;
	const bool HM2_STDBINARYSEARCH_SWITCH       = true;
	const bool HM2_INTERPOLATIONSEARCH_SWITCH   = false;
	const bool HM2_LINEAREARCH_SWITCH           = false;
	const bool HM2_USE_LOAD_IN_FIND_SWITCH      = true;

	// If this flag is set to true, the tree will save in the hashmaps the whole 64 bits regardless of the given hashmap
	// bucket size. This allows the tree to check fast if an element in the last level exceeds the weight threshold or not.
	const bool SAVE_FULL_128BIT                 =  (baselist_p == 3);

	// This flag configures the internal hashmaps. If set to `true` the data container within the hashmaps, normally
	// saving the lpart and the indices, get a third data container holding at least 128bit of the label.
	const bool EXTEND_TO_TRIPLE                 = false;

	// Set 0 zero to disable.
	// See `BJMM::append_trivial_rows`
	// If this value is set to anything else than zero, this number of rows will be appended to the working matrix wH.
	// The goal is to decrease the code rate and therefore increase the performance of the algorithms.
	const uint32_t TrivialAppendRows = ClassicalTree ? 0 : (d == 3 ? 0 : (DOOM ? 0 : (IM_nr_views != 0 ? 0 : (Baselist_Full_Length ? 0 : (c != 0 ? 0 : 1)))));

	// Stop the outer loop after `loops` steps. Useful for benchmarking.
#ifdef USE_LOOPS
	const uint64_t loops        = USE_LOOPS;
#else
	const uint64_t loops        = uint64_t(-1);
#endif
#ifdef PRINT_LOOPS
	const uint64_t print_loops  = PRINT_LOOPS;
#else
	const uint64_t print_loops  = 1;
#endif

	const uint64_t exit_loops   = 10000;

	// custom settable seed. If set to any other value than `0` this will be used to seed the internal prng.
	const uint64_t seed = 0;

	// Constructor
	constexpr ConfigBJMM(const uint32_t n, const uint32_t k, const uint32_t w, const uint32_t baselist_p,
	                     const uint32_t l, const uint32_t l1,
	                     const uint64_t sb1 = 100, const uint64_t sb2 = 100,    // Size of the buckets
	                     const uint64_t nb1 = 5,   const uint64_t nb2 = 5,      // logarithmic number of buckets
	                     const uint32_t threshold = 4,                          // threshold
	                     const uint32_t nt = 1,                                 // number of threads
	                     const bool DOOM = false,                               // Decode one out of many? (Only valid for BIKE instances)
	                     const bool DOOM_Alternative_Tree = false,
	                     const bool Baselist_Full_Length = false,
	                     const uint32_t c = 0,
	                     const bool LOWWEIGHT=false,
	                     const uint32_t l2 = 0,
						 const uint32_t IM_nr_views=0,
						 bool ClassicalTree=false,
	                     const bool HM1_STDBINARYSEARCH_SWITCH=true, const bool HM2_STDBINARYSEARCH_SWITCH=true,
	                     const bool HM1_INTERPOLATIONSEARCH_SWITCH=false, const bool HM2_INTERPOLATIONSEARCH_SWITCH=false,
	                     const bool HM1_LINEAREARCH_SWITCH=false, const bool HM2_LINEAREARCH_SWITCH=false,
	                     const bool HM1_USE_LOAD_IN_FIND_SWITCH=true, const bool HM2_USE_LOAD_IN_FIND_SWITCH=true
	                     ) :
			n(n), k(k), w(w), baselist_p(baselist_p), l(l), l1(l1),
			weight_threshhold(threshold), m4ri_k(matrix_opt_k(n - k, MATRIX_AVX_PADDING(n))), nr_threads(nt),
			size_bucket1(sb1), size_bucket2(sb2),
			number_bucket1(nb1), number_bucket2(nb2),
			ClassicalTree(ClassicalTree),
			DOOM(DOOM), DOOM_Alternative_Tree(DOOM_Alternative_Tree),
			Baselist_Full_Length(Baselist_Full_Length), LOWWEIGHT(LOWWEIGHT), c(c), IM_nr_views(IM_nr_views), l2(l2),
			HM1_STDBINARYSEARCH_SWITCH(HM1_STDBINARYSEARCH_SWITCH),
			HM1_INTERPOLATIONSEARCH_SWITCH(HM1_INTERPOLATIONSEARCH_SWITCH),
			HM1_LINEAREARCH_SWITCH(HM1_LINEAREARCH_SWITCH),
			HM1_USE_LOAD_IN_FIND_SWITCH(HM1_USE_LOAD_IN_FIND_SWITCH),
			HM2_STDBINARYSEARCH_SWITCH(HM2_STDBINARYSEARCH_SWITCH),
			HM2_INTERPOLATIONSEARCH_SWITCH(HM2_INTERPOLATIONSEARCH_SWITCH),
			HM2_LINEAREARCH_SWITCH(HM2_LINEAREARCH_SWITCH),
			HM2_USE_LOAD_IN_FIND_SWITCH(HM2_USE_LOAD_IN_FIND_SWITCH) {};

	// prints information about the problem instance.
	void print() const {
		std::cout << "n: " << n << ", k: " << k <<  ", c: " << c << ", p: " << baselist_p << ", l: " << l << ", l1: " << l1
				  << ", DOOM: " << DOOM << ", DOOM_Alternative_Tree: " << DOOM_Alternative_Tree
		          << ", log(#buckets1): " << number_bucket1 << ", log(#buckets2): " << number_bucket2
		          << ", size_bucket1: " << size_bucket1 << ", size_bucket2: " << size_bucket2
		          << ", threads: " << nr_threads << ", m4ri_k: " << m4ri_k << ", weight_threshhold: "
		          << weight_threshhold << ", loops: " << loops << ", print_loops: " << print_loops << ", exit_loops: " << exit_loops
		          << ", intermediate_target_loops:" << intermediate_target_loops << ", epsilon: " << epsilon
		          << ", Baselist_Full_Length: " << Baselist_Full_Length
		          << ", TrivialAppendRows: " << TrivialAppendRows << "\n";
	}
};

template<const ConfigBJMM &config>
class BJMM {
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
	//                                          │       │   │       │          |L|**2 /  l1
	//                                          │       │   │       │                /  2
	//                                          │       │   │       │
	//                                          │       │   │       │
	//                                          │       │   │       │
	//                                          └──┬────┘   └───┬───┘          Match on l1
	//                                     ┌───────┴─┐         ┌┴────────┐
	//                                 ┌───┼───┐ ┌───┴───┐ ┌───┴───┐ ┌───┴───┐
	//                                 │   │   │ │       │ │       │ │       │      k+l
	//                                 │ L1│   │ │  L2   │ │  L3   │ │   L4  │      ---
	//                                 │   │   │ │       │ │       │ │       │       2
	//                                 │   │   │ │       │ │       │ │       │
	//                                 │   │   │ │       │ │       │ │       │    -------
	//                                 │   │   │ │       │ │       │ │       │     basep
	//                                 └───┴───┘ └───────┘ └───────┘ └───────┘
	//
	//      Memory Layout
	//      Addr:   0    63|64 127|128   n-k
	//      Label:  [limb0 |limb1 |limb2... ]
	//
	//      LimbLayout:
	//      limb0:  [     ]
	//      Addr:   0     63
	// constants from the problem instance
	constexpr static uint32_t n   = config.n;
	constexpr static uint32_t k   = config.k - config.TrivialAppendRows;  // See the explanation of the function `TrivialAppendRows`
	constexpr static uint32_t w   = config.w;
	constexpr static uint32_t p   = config.baselist_p;                    // NOTE: from now on p is the baselist p not the full p.
	constexpr static uint32_t l   = config.l;
	constexpr static uint32_t l1  = config.l1;
	constexpr static uint32_t l2  = config.l2;
	constexpr static uint32_t nkl = n - config.k - l;
	constexpr static uint32_t c   = config.c;
	constexpr static uint32_t AlignmentLabel = 32;

	using DecodingValue     = Value_T<BinaryContainer<k+l-c>>;
	using DecodingLabel     = Label_T<BinaryContainer<n - config.k>>;
	using DecodingMatrix    = mzd_t *;
	using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList      = Parallel_List_T<DecodingElement>;
	using ChangeList        = std::vector<std::pair<uint16_t, uint16_t>>;

	typedef typename DecodingList::ValueContainerType ValueContainerType;
	typedef typename DecodingList::LabelContainerType LabelContainerType;
	typedef typename DecodingList::ValueContainerLimbType ValueContainerLimbType;
	typedef typename DecodingList::LabelContainerLimbType LabelContainerLimbType;

	// options for the matrix split
	constexpr static uint32_t nkl_align = 128;
	constexpr static uint32_t nkl_alignment = 0;//((nkl+nkl_align-1)/nkl_align)*nkl_align;
	constexpr static uint32_t nkl_c = 0;//nkl_alignment - nkl;


	// constant variables for the tree merge algorithm
	constexpr static uint32_t threads = config.nr_threads;

	// precalculate the expected size of the baselists. For the sake of simplicity we assume that all lists within the
	// tree structure is of the same size. Additionally, we precalculate the size of each part a single thread is working on.
	constexpr static uint64_t
			lsize1 = config.Baselist_Full_Length ?
					bc(k+l-c, p) :
					bc(config.epsilon + ((k + l - c) / 2), p),
			lsize2 = (config.Baselist_Full_Length & !config.DOOM) ?
					bc(k+l-c-config.LOWWEIGHT, p) :
					bc(config.epsilon + ((k + l - c -config.LOWWEIGHT) - (k + l - c) / 2), p);
	constexpr static uint64_t thread_size_lists1 = lsize1 / threads, thread_size_lists2 = lsize2 / threads;

	// list per thread size.
	constexpr static uint32_t tL1len = lsize1 / threads;
	constexpr static uint32_t tL2len = lsize2 / threads;

	// we choose the datatype depending on the size of l.
	// container of the `l` part if the label
	constexpr static uint32_t ArgumentLimbType_l = config.SAVE_FULL_128BIT ? 128 : (config.IM_nr_views == 0 ? l : std::max(l, l2));
	using ArgumentLimbType  = LogTypeTemplate<ArgumentLimbType_l>;
	// container of an index in the changelist and the hashmaps.
	using IndexType         = TypeTemplate<std::max(
												std::max(size_t((uint64_t(1u) << config.number_bucket1) * config.size_bucket1),
														 size_t((uint64_t(1u) << config.number_bucket2) * config.size_bucket2)),
														 size_t((uint64_t(1u) << config.number_bucket3) * config.size_bucket2))>;
	using LoadType          = IndexType; // container which hold the maximum number of
										 // elements we have to iterate over the hash map. Can be smaller than `IndexType`
										 // To reduce the memory consumption, each HashMap derives its own `LoadType`,
										 // which can be smaller than this. So this `LoadType` can still be used in general
										 // over both hashmaps without the loss of information.

	// b0, b1, b2 must be given as zero aligned. NOT n-k-l aligned.
	// That's because we shift every element to its correct position before hashing them into the hashmap.
	constexpr static uint32_t bucket_offset = config.d == 3 ? l - l2-l1 : 0;
	constexpr static uint32_t number_bucket2 = config.number_bucket2;
	constexpr static uint32_t b10 = 0;
	constexpr static uint32_t b11 = 0 + config.number_bucket1;
	constexpr static uint32_t b12 = 0 + l1;
	constexpr static uint32_t b20 = 0 + l1;
	constexpr static uint32_t b21 = 0 + l1 + number_bucket2;
	// if d == 3:
	//      tree depth 3
	// else tree depth 2
	//   if l2 != 0:
	//      doing NN search
	//   else
	//      normal algorithm
	//   endif
	// endif
	constexpr static uint32_t b22 = config.d == 3 ? (l1+l2) : config.l2 != 0 ? (l1+l2) : l;

	constexpr static ConfigParallelBucketSort chm1{b10, b11, b12, config.size_bucket1,
	                                               uint64_t(1) << config.number_bucket1, threads, 1, n - config.k - l, l, 0, 0,
	                                               config.HM1_STDBINARYSEARCH_SWITCH, config.HM1_INTERPOLATIONSEARCH_SWITCH,
												   config.HM1_LINEAREARCH_SWITCH, config.HM1_USE_LOAD_IN_FIND_SWITCH,
												   config.SAVE_FULL_128BIT, config.EXTEND_TO_TRIPLE};
	constexpr static ConfigParallelBucketSort chm2{b20, b21, b22, config.size_bucket2,
	                                               uint64_t(1) << config.number_bucket2, threads, 2, n - config.k - l, l, 0, 0,
	                                               config.HM2_STDBINARYSEARCH_SWITCH, config.HM2_INTERPOLATIONSEARCH_SWITCH,
												   config.HM2_LINEAREARCH_SWITCH, config.HM2_USE_LOAD_IN_FIND_SWITCH,
												   config.SAVE_FULL_128BIT, config.EXTEND_TO_TRIPLE};

	// we do not have to care about the possibility that the l part can be spun about 2 limbs. because we do not touch the l part.
	constexpr static uint32_t llimbs    = LabelContainerType::limbs();              // number of limbs=uint64_t which are processed of a label
	constexpr static uint32_t llimbs_a  = LabelContainerType::bytes()/8;
	constexpr static uint32_t lVCLTBits = sizeof(ValueContainerLimbType)*8;
	constexpr static uint32_t loffset   = nkl / lVCLTBits;
	constexpr static uint32_t lshift    = nkl - (loffset * lVCLTBits);


	// masks and limbs of the weight calculations of the final label
	constexpr static uint32_t               lupper = LabelContainerType::round_down_to_limb(nkl - 1);
	constexpr static ValueContainerLimbType lumask = LabelContainerType::lower_mask2(nkl);

	// M4Ri matrix structures
	mzd_t *work_matrix_H, *work_matrix_H_T, *sT, *H, *HT;
	mzd_t *outer_matrix_H, *outer_matrix_HT;
	mzp_t *permutation;

	// Additional outer permutation if we choose to cut of c coordinates from the challenge. This is the same as
	// guessing zero positions.
	mzp_t *c_permutation;
	customMatrixData *matrix_data;

	// Instance parameters
	mzd_t *e;
	const mzd_t *s;
	const mzd_t *A;

	// DOOM (Decode one out of many):
	constexpr static uint32_t DOOM_nr = config.DOOM ? k : 0;
	mzd_t * DOOM_S;
	mzd_t * DOOM_S_View;

	// trivial append row
	mzd_t *trivial_row_row2;
	mzd_t *trivial_row_row3;

	// Instead of iterating each element in the baselist, we increment the pointer to the next element by the length
	// of one element, aligned to the lpart. And depending on if DOOM is activated or not this value changes,
	constexpr static uint64_t BaseList4Inc = config.DOOM ? MATRIX_AVX_PADDING(n - k) / 64 : llimbs_a;

	// Changelists. Holding the bit position of the two change bits in grey code. Used for efficient calculations of the
	// labels corresponding to values.
	ChangeList cL1, cL2;

	// Baselists. Holding values and labels, with label=H*value. The values are constant whereas the labels must
	// recompute for every permutation of the working matrix.
	DecodingList L1{lsize1, threads, thread_size_lists1}, L2{lsize2, threads, thread_size_lists2};

	// Hashmaps. Fast (constant) insert and look up time.
	template<const uint32_t l, const uint32_t h>
	static ArgumentLimbType Hash(uint64_t a) {
		static_assert(l < h);
		static_assert(h < 64);
		constexpr uint64_t mask = (~((uint64_t(1u) << l) - 1u)) &
		                          ((uint64_t(1u) << h) - 1u);
		return (uint64_t(a) & mask) >> l;
	}

	using HM1Type = ParallelBucketSort<chm1, DecodingList, ArgumentLimbType, IndexType, &Hash<b10, b11>>;
	using HM2Type = ParallelBucketSort<chm2, DecodingList, ArgumentLimbType, IndexType, &Hash<b20, b21>>;
	using HM1BucketIndexType = typename HM1Type::BucketIndexType;
	using HM2BucketIndexType = typename HM2Type::BucketIndexType;
	HM1Type *hm1; HM2Type *hm2;

	using Extractor = WindowExtractor<DecodingLabel, ArgumentLimbType>;

	// Equivalent to the number of baselists.
	constexpr static uint32_t npos_size = config.d == 3 ? 8 : 4;

	// needed to randomize the baselists.
	alignas(AlignmentLabel) LabelContainerType iT1, target;

	// Test/Bench parameter
	uint32_t ext_tid = 0;
	bool not_found = true;

	// if set to true the internal hashmaps weill be allocated.
	bool init = true;

	// just a helper value to count the elements which survive to the last level.
	uint64_t last_list_counter = 0;

	// measures the time without alle the preprocessing and allocating
	double internal_time;

	/// \param e instance parameter
	/// \param s instance parameter
	/// \param A instance parameter
	/// \param ext_tid 	if this constructor is called within different threads, pass a unique id via this flag.
	///					This Flag is then passed to the internal rng.
	/// \param not_init	Do not init the internal data structures like hashmap etc. Useful if a derived class
	///			constructor is called.
	BJMM(mzd_t *e, const mzd_t *const s, const mzd_t *const A, const uint32_t ext_tid = 0, const bool init=true)
			: e(e), s(s), A(A), ext_tid(ext_tid), init(init) {
		static_assert(n > k, "wrong dimension");
		static_assert(config.number_bucket1 <= l1, "wrong hm1 #bucket");
		if constexpr(l2 == 0) {
			static_assert(config.number_bucket2 <= l, "wrong hm2 #bucket");
		}

		static_assert((threads <= config.size_bucket1) && (threads <= config.size_bucket2), "wrong #threads");
		static_assert((config.size_bucket1 % threads == 0) && (config.size_bucket2 % threads == 0), "wrong #threads");
		static_assert((config.DOOM + config.LOWWEIGHT) < 2, "DOOM AND LOWWEIGHT are not valid.");
		static_assert(((config.c != 0) + config.LOWWEIGHT) < 2, "CUTOFF AND LOWWEIGHT are not valid.");
		if constexpr(config.SAVE_FULL_128BIT) {
			static_assert(n-k >= 128);
		}

		// reset the found indicator variable.
		not_found = true;

		// Make sure that some internal details are only accessed by the omp master thread.
#pragma omp master
		{
#if !defined(BENCHMARK) && !defined(NO_LOGGING)
			if (config.loops != uint64_t(-1)) {
				std::cout << "IMPORTANT: TESTMODE: only " << config.loops << " permutations are tested\n";
			}

			chm1.print();
			chm2.print();
			config.print();
		}

		// Seed the internal prng.
		if (config.seed != 0) {
			random_seed(config.seed + ext_tid);
		} else {
			srand(ext_tid + time(nullptr));
			random_seed(ext_tid + rand() * time(nullptr));
		}

		// Ok this is ridiculous.
		// Apparently m4ris mzd_init is not thread safe. Cool.
#pragma omp critical
		{
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

				static_assert(n / 2 == k);
				DOOM_S = mzd_init(k, k);

				// Generate the DOOM shifts
				for (int i = 0; i < DOOM_nr; ++i) {
					matrix_down_shift_into_matrix(DOOM_S, sT, i, i);
				}

				if constexpr(config.c == 0) {
					work_matrix_H = matrix_init(n - k, n + DOOM_nr);
					work_matrix_H_T = mzd_init(work_matrix_H->ncols, work_matrix_H->nrows);
					matrix_concat(work_matrix_H, A, DOOM_S);
				} else {
					outer_matrix_H = matrix_init(n - k, n + 1);
					outer_matrix_HT = mzd_init(outer_matrix_H->ncols, outer_matrix_H->nrows);
					matrix_concat(outer_matrix_H, A, sT);

					work_matrix_H = matrix_init(n - k, n - c + DOOM_nr + 1);
					work_matrix_H_T = mzd_init(work_matrix_H->ncols, work_matrix_H->nrows);
				}

				H = mzd_init(n - k, k + l - c + DOOM_nr);
				HT = matrix_init(H->ncols, H->nrows);
				DOOM_S_View = mzd_init_window(HT, k + l - c, 0, k + l + DOOM_nr - c, HT->ncols);
			} else {
				if constexpr(config.c == 0) {
					work_matrix_H = matrix_init(n - k, n + 1);
					work_matrix_H_T = mzd_init(work_matrix_H->ncols, work_matrix_H->nrows);
					mzd_t *tmp = matrix_concat(nullptr, A, sT);
					mzd_copy(work_matrix_H, tmp);
					mzd_free(tmp);

					//mzd_print(work_matrix_H);
					if (config.TrivialAppendRows != 0)
						work_matrix_H = append_trivial_rows(work_matrix_H);
					//mzd_print(work_matrix_H);
				} else {
					// init all matrix structures
					outer_matrix_H = matrix_init(n - k, n + 1);
					outer_matrix_HT = mzd_init(outer_matrix_H->ncols, outer_matrix_H->nrows);
					matrix_concat(outer_matrix_H, A, sT);

					work_matrix_H = matrix_init(n - k, n - c + 1);
					work_matrix_H_T = mzd_init(work_matrix_H->ncols, work_matrix_H->nrows);
				}

				H = mzd_init(n - k, k + l - c);
				HT = matrix_init(H->ncols, H->nrows);
			}

			// init the helper struct for the gaussian eleimination and permutation data structs.
			matrix_data = init_matrix_data(work_matrix_H->ncols);
			permutation = mzp_init(n - c);
			if (c != 0)
				c_permutation = mzp_init(n);

			// Check if the working matrices are all allocated
			if ((work_matrix_H == nullptr) || (work_matrix_H_T == nullptr) || (sT == nullptr) ||
			    (H == nullptr) || (HT == nullptr) || (permutation == nullptr) || (matrix_data == nullptr)) {
				std::cout << "ExtTID: " << ext_tid << ", alloc error2\n";
				exit(-1);
			}
		}

		// Init the target.
		target.zero();
		iT1.zero();

		// set to false if the constructor is called from a derived class
		if (init) {
			// initialize non constant variables: HashMaps
			hm1 = new HM1Type();
#ifdef USE_MO
#if USE_MO == 0
			hm2 = new HM2Type();
#endif
#else
			hm2 = new HM2Type();
#endif
		}

		auto list_precompute_time = (double)clock();
		if constexpr(config.Baselist_Full_Length) {
			static_assert(config.Baselist_Full_Length && (config.epsilon == 0));
			prepare_baselist_fulllength_mitm_with_chase2();
		} else {
			BJMM_prepare_generate_base_mitm2_with_chase2(L1, L2, cL1, cL2);
		}

#if !defined(BENCHMARK) && !defined(NO_LOGGING)
		std::cout << "Precomputation took: " << ((double)clock() - list_precompute_time) / (CLOCKS_PER_SEC) << " s\n";
		std::cout << "Init took: " << ((double)clock() - inittime) / (CLOCKS_PER_SEC) << " s\n";
#endif

		ASSERT(L1.size() == cL1.size());
	}

	~BJMM() {
		// free memory
		if (init) {
			delete (hm1);
			delete (hm2);
		}

#if defined(NUMBER_OUTER_THREADS) && NUMBER_OUTER_THREADS != 1
		// I have no idea why.
#pragma omp critical
		{
#endif
			mzd_free(work_matrix_H);
			mzd_free(work_matrix_H_T);
			mzd_free(sT);
			mzd_free(H);
			mzd_free(HT);
			mzp_free(permutation);

			free_matrix_data(matrix_data);

			if (config.DOOM) {
				mzd_free(DOOM_S);
				mzd_free(DOOM_S_View);
			}

			if constexpr(c != 0) {
				mzd_free(outer_matrix_H);
				mzd_free(outer_matrix_HT);
				mzp_free(c_permutation);
			}

			if constexpr (config.TrivialAppendRows > 1) {
				mzd_free(trivial_row_row2);
				mzd_free(trivial_row_row3);
			}

#if defined(NUMBER_OUTER_THREADS) && NUMBER_OUTER_THREADS != 1
	}
#endif
	}

	/// IMPORTANT Notes:
	///		- does not resize L1 or L2, so they need to set already
	///		- epsilon is implemented
	/// \param bL1
	/// \param bL2
	/// \param diff_list1
	/// \param diff_list2
	template<typename DecodingList=DecodingList>
	void BJMM_prepare_generate_base_mitm2_with_chase2(DecodingList &bL1, DecodingList &bL2,
	                                                  ChangeList &cL1, ChangeList &cL2) noexcept {
		typedef typename DecodingList::ValueType Value;
		typedef typename DecodingList::ValueContainerType ValueContainerType;
		typedef typename DecodingList::ValueContainerType::ContainerLimbType VCLT;

		Value e11{}, e12{}, e21{}, e22{};
		e11.zero(); e12.zero(); e21.zero(); e22.zero();

		// Note: if TrivialAppendRows != 0: config.k  != k
		constexpr uint32_t n_full = config.k + config.l - config.c - config.TrivialAppendRows;
		constexpr uint32_t n = n_full / 2;
		constexpr uint32_t p = config.baselist_p;
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

	//
	// generate the chase sequence.
	void prepare_baselist_fulllength_mitm_with_chase2() noexcept {
		DecodingValue e11{}, e12{}, e21{}, e22{};
		e11.zero(); e12.zero(), e21.zero(); e22.zero();

		constexpr uint32_t n_full = config.k + config.l - config.c - config.TrivialAppendRows;
		constexpr uint32_t p = config.baselist_p;
		constexpr uint32_t limbs = ValueContainerType::limbs();

		// resize the data.
		// if we are in the doom setting, we want to enumerate L1 and L3 on the full length but L2 only on one half.
		constexpr bool l2_halfsize = config.DOOM;
		constexpr uint64_t lsize1 = bc(n_full, p);
		constexpr uint64_t lsize2 = l2_halfsize ? bc(((k + l - c) - (k + l - c) / 2), p) : lsize1;
		ASSERT(lsize1 == this->lsize1 && lsize2 == this->lsize2);

		cL1.resize(lsize1);
		cL2.resize(lsize2);

		Combinations_Chase_Binary<ValueContainerLimbType> ccb_l1{n_full, p, 0};
		Combinations_Chase_Binary<ValueContainerLimbType> ccb_l2{n_full, p, 0};
		ccb_l1.left_init(e11.data().data().data());
		ccb_l1.left_step(e11.data().data().data(), true);
		ccb_l2.left_init(e21.data().data().data());
		ccb_l2.left_step(e21.data().data().data(), true);

		L1.data_value(0) = e11;
		L2.data_value(0) = e21;

		uint16_t pos1, pos2;

		auto diff_index = [&](const DecodingValue &a, const DecodingValue &b) {
			Combinations_Chase_Binary<ValueContainerLimbType>::diff(a.data().data().data(),
			                                      b.data().data().data(),
			                                      limbs, &pos1, &pos2);
		};

		// Because of the flag `l2_halfsize` we have to enumerate the two lists seperatly.
		uint64_t c1 = 1, c2 = 1;
		while (c1 < lsize1) {
			e12 = e11;
			ccb_l1.left_step(e11.data().data().data());
			diff_index(e11, e12);
			L1.data_value(c1) = e11;
			cL1[c1 - 1] = std::pair<uint32_t, uint32_t>(pos1, pos2);

			// std::cout << L1.data_value(c2) << ": " << cL1[c1 - 1].first << " " << cL1[c1 - 1].second << "\n";
			c1 += 1;
		}
		while (c2 < lsize2) {
			// List L2
			e22 = e21;
			ccb_l2.left_step(e21.data().data().data());
			diff_index(e21, e22);
			L2.data_value(c2) = e21;
			cL2[c2 - 1] = std::pair<uint32_t, uint32_t>(pos1, pos2);

			// std::cout << L2.data_value(c2) << ": " << cL2[c2 - 1].first << " " << cL2[c2 - 1].second << "\n";
			c2 += 1;
		}
	}

	/// The input matrix will be appended by a few rows.
	/// Note that the input matrix must have in its last colum the syndrom
	/// Append the following trivial rows
	///	[ 1^n 						| 0]	row1
	/// \param in input matrix.
	/// \return newly created matrix with a few additional rows.
	static mzd_t* append_trivial_rows(mzd_t *in) noexcept {
		assert(config.DOOM_Alternative_Tree == false);
		assert(config.Baselist_Full_Length == false);

		const uint32_t cols = in->ncols;
		const uint32_t rows = in->nrows;

		// Note the use of TrivialAppendRows
		mzd_t *ret  = in;
		mzd_t *row1 = mzd_init(1, cols);

		// write the rows
		for (uint32_t i = 0; i < n; ++i) {
			mzd_write_bit(row1, 0, i, 1);
		}

		// write the syndrom
		mzd_write_bit(row1, 0, n, w%2);
		mzd_copy_row(ret, rows-config.TrivialAppendRows, row1, 0);
		mzd_free(row1);
		return ret;
	}

	/// TODO remove. helper function
	/// \tparam Label
	/// \param a
	/// \param b
	template<class Label>
	constexpr inline void internal_xor_helper(DecodingLabel &a, const word *b) noexcept {
		#pragma unroll
		for (uint32_t i = 0; i < a.data().limbs(); ++i) {
			a.data().data()[i] ^= b[i];
		}
	}

	/// This function fills up two given list, in which the `values` are set according to some `Chase` sequence and sets the correct `label`.
	/// This difference between two `values` (difference = bit positons in which two bit strings differs) __MUST__
	/// passed to this function in `v1` and `v2`.
	/// By this approach we avoid that we have to calculate a matrix-vector multiplication for each `value`.
	/// \param L1	Input/Output List
	/// \param L2	Input/Output List
	/// \param v1	Input DiffList
	/// \param v2	Input DiffList
	/// \param HT	Input current permuted and gaussian elimination applied matrix. __MUST__ be transposed
	/// \param tid	Thread index
	template<class DecodingList>
	void BJMM_fill_decoding_lists(DecodingList &L1, DecodingList &L2,
	                         ChangeList &cL1, ChangeList &cL2,
	                         const mzd_t *HT, const uint32_t tid,
							 const bool add_target=false, const DecodingLabel *iTarget=nullptr) noexcept {
		const size_t start1 = tid * tL1len;   // starting index within the list L1 of each thread
		const size_t start2 = tid * tL2len;   // starting index within the list L2 of each thread
		const size_t end1 = tid == (this->threads - 1) ? this->lsize1 : start1 +
		                                                                  this->tL1len;  // ending index of each thread within the list L1,
		const size_t end2 = tid == (this->threads - 1) ? this->lsize2 : start2 +
		                                                                  this->tL2len;  // exepct for the last thread. This needs
		uint32_t P1[p] = {0};
		uint32_t P2[p] = {0};

		// extract the bits currently set in the value
		L1.data_value(start1).data().get_bits_set(P1, p);
		L2.data_value(start2).data().get_bits_set(P2, p);

		// prepare the first element
		L1.data_label(start1).zero();
		L2.data_label(start2).zero();
		for (uint32_t i = 0; i < p; ++i) {
			internal_xor_helper<DecodingLabel>(L1.data_label(start1), HT->rows[P1[i]]);
			internal_xor_helper<DecodingLabel>(L2.data_label(start2), HT->rows[P2[i]]);
		}

		if (add_target) {
			// TODO not implemented
			ASSERT(0);

			LabelContainerType::add(L2.data_label(start2).data(),
			                        L2.data_label(start2).data(),
			                        iTarget->data(),
					// TODO right limits?
					                config.n-config.k-config.l+config.l1, config.n-config.k);
		}

		for (size_t i = MIN(start1 + 1, start2 + 1);
		     i < MAX(end1, end2); i++) {
			if ((i >= start1 + 1) && (i < end1)) {
				L1.data_label(i) = L1.data_label(i-1);
				internal_xor_helper<DecodingLabel>(L1.data_label(i), HT->rows[cL1[i - 1].first]);
				internal_xor_helper<DecodingLabel>(L1.data_label(i), HT->rows[cL1[i - 1].second]);
			}
			if ((i >= start2 + 1) && (i < end2)) {
				L2.data_label(i) = L2.data_label(i-1);
				internal_xor_helper<DecodingLabel>(L2.data_label(i), HT->rows[cL2[i - 1].first]);
				internal_xor_helper<DecodingLabel>(L2.data_label(i), HT->rows[cL2[i - 1].second]);
			}
		}
	}

	/// this functions
	/// \param label
	/// \param npos
	/// \param weight
	/// \param DOOM_index2
	void __attribute__ ((noinline))
	check_final_list(LabelContainerType &label,
					   IndexType npos[4],
					   const uint32_t weight,
					   const uint32_t DOOM_index2) noexcept {
		// make sure that only one thread can access this area at a given time.
		#pragma omp critical
		{
			MULTITHREADED_WRITE(finished.store(true);)
			// make really sure that only one thread every runs this code.
			if (not_found) {
				#pragma omp atomic write
				not_found = false;
				#pragma omp flush(not_found)

				// tmp variable to recompute the solution.
				ValueContainerType value;

				if constexpr(!config.DOOM_Alternative_Tree) {
					// A little optimisation. Because we only need to compute the value once for the
					// golden element we can do it here.
					ValueContainerType::add_withoutasm(value, L1.data_value(npos[0]).data(), L1.data_value(npos[2]).data());
					ValueContainerType::add_withoutasm(value, value, L2.data_value(npos[1]).data());
					if constexpr(!config.DOOM) {
						ValueContainerType::add_withoutasm(value, value, L2.data_value(npos[3]).data());
					}
				} else {
					ValueContainerType::add_withoutasm
							(value, L1.data_value(npos[0]).data(),
									L2.data_value(npos[1]).data());
				}

				if constexpr(config.d == 3) {
					ValueContainerType::add_withoutasm
							(value, value, L1.data_value(npos[4]).data());
					ValueContainerType::add_withoutasm
							(value, value, L1.data_value(npos[6]).data());
					ValueContainerType::add_withoutasm
							(value, value, L2.data_value(npos[5]).data());
					ValueContainerType::add_withoutasm
							(value, value, L2.data_value(npos[7]).data());
				}

				if constexpr(config.IM_nr_views != 0) {
					// NN case: add the target on the l window if it was canceled out.
					std::cout << label << " label3\n";
					if (npos[3] > npos[1])
						LabelContainerType::add(label, label, target, n-k-config.l1-(config.IM_nr_views*config.l2), n-k);
					std::cout << label << " label3\n";
				}

				uint64_t ctr1 = 0;
				uint64_t ctr2 = 0;


				// recompute the error vector by first setting the label and value at the correct
				// position and then apply the back permutation.
				for (uint32_t j = config.TrivialAppendRows; j < n - c; ++j) {
					uint32_t bit = 0;
					constexpr uint32_t limit = n - k - l;
					if (j < limit) {
						bit = label[j-config.TrivialAppendRows];
						ctr1 += bit;
					} else {
						bit = value[j - limit];
						ctr2 += bit;
					}

					if constexpr(config.LOWWEIGHT) {
						if (j == n-1)
							bit = 1;
					}
					mzd_write_bit(e, 0, permutation->values[j], bit);
				}

				// NOTE: See `config.TrivialAppendRows` if you want to know what this option does.
				//  Here we fix the trick by dropping the first row of the main matrix.
				if(config.TrivialAppendRows == 1) {
					auto bit = 0;
					for (uint32_t i = n-k-l; i < n; ++i) {
						if (value[i-(n-k-l)] == 1) {
							bit ^= mzd_read_bit(work_matrix_H, 0, i);
						}
					}

					bit ^= mzd_read_bit(work_matrix_H, 0, n-c);
					mzd_write_bit(e, 0, permutation->values[0], bit);
				}

#ifndef BENCHMARK
				std::cout << " pre perm \n";
				std::cout << "weight n-k-l:" << ctr1 << "\n";
				std::cout << "weight tree: " << ctr2 << "\n";
				std::cout << "weight input: " << weight << "\n";
				std::cout << "hashmap " << DOOM_index2 << " found\n";

				std::cout << iT1 << " iT\n";
				std::cout << target << " target\n";
				std::cout << label << " label\n";
				std::cout << L1.data_label(npos[0]).data() << " npos[0]:" << npos[0] << "\n";
				std::cout << L2.data_label(npos[1]).data() << " npos[1]:" << npos[1] << "\n";
				std::cout << L1.data_label(npos[2]).data() << " npos[2]:" << npos[2] << "\n";
				if constexpr(!config.DOOM) {
					std::cout << L2.data_label(npos[3]).data() << " npos[3]:" << npos[3] << "\n";
				}
				if constexpr(config.d == 3) {
					std::cout << L1.data_label(npos[4]).data() << " npos[4]\n";
					std::cout << L2.data_label(npos[5]).data() << " npos[5]\n";
					std::cout << L1.data_label(npos[6]).data() << " npos[6]\n";
					std::cout << L2.data_label(npos[7]).data() << " npos[7]\n";
				}

				std::cout << "\n" << value << " value\n";
				std::cout << L1.data_value(npos[0]).data() << " npos[0]:" << npos[0] << "\n";
				std::cout << L2.data_value(npos[1]).data() << " npos[1]:" << npos[1] << "\n";
				std::cout << L1.data_value(npos[2]).data() << " npos[2]:" << npos[2] << "\n";
				if constexpr(!config.DOOM) {
					std::cout << L2.data_value(npos[3]).data() << " npos[3]:" << npos[3] << "\n";
				}
#endif
				// Apply the back permutation of the outer loop if necessary.
				if constexpr(config.c != 0) {
					mzd_t *tmpe = mzd_init(1, n);

					for (uint32_t j = 0; j < n; ++j) {
						mzd_write_bit(tmpe, 0, c_permutation->values[j], mzd_read_bit(e, 0, j));
					}

//					print_matrix("pre", e);
//					print_matrix("aft", tmpe);
					mzd_copy_row(e, 0, tmpe, 0);
					mzd_free(tmpe);
				}

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

				if constexpr(config.LOWWEIGHT) {
					std::cout << "LOWWEIGHT : Weight: " << hamming_weight(e) << "\n";
				}
			}
		}
	}

	/// IMPORTANT: This code is only valid if the parameter c is unequal ot 0.
	uint64_t __attribute__ ((noinline)) BJMMF_Outer() noexcept {
		uint64_t loops = 0;
		uint64_t outer_loops = 0;
		while (not_found) {
			matrix_create_random_permutation(outer_matrix_H, outer_matrix_HT, c_permutation);
			if constexpr (config.DOOM) {
				mzd_submatrix(work_matrix_H, outer_matrix_H, 0, 0, n-k, n-c);
				mzd_append(work_matrix_H, DOOM_S, 0, n-c);
			} else {
				mzd_submatrix(work_matrix_H, outer_matrix_H, 0, 0, n-k, n-c);
			}

			// reset permutation
			for (uint32_t i = 0; i < n - c; ++i) {
				permutation->values[i] = i;
			}


			// write the syndrom
			for (uint32_t i = 0; i < n - k; ++i) {
				mzd_write_bit(work_matrix_H, i, n - c, mzd_read_bit(outer_matrix_H, i, n));
			}
			loops += BJMMF();
			outer_loops += 1;
		}

#if !defined(BENCHMARK) && !defined(NO_LOGGING)
		std::cout << "Outer Loops: " << outer_loops << "\n";
#endif
		return loops;
	}

	// runs the algorithm and return the number of iterations needed
	uint64_t __attribute__ ((noinline)) BJMMF() noexcept {
		// count the loops we iterated
		uint64_t loops = 0;

		// we have to reset this value, so It's possible to rerun the algo more often
		not_found = true;

		// measure the intern clocks wihtout all the preprocessing
		internal_time = clock();

		// start the whole thing
		while (not_found && loops < config.loops) {
			// If we cut off some coordinates from the main matrix.
			// Exit this inner loop after the expected number of loops.
			if constexpr(c != 0) {
				if (loops >= config.c_inner_loops)
					return loops;
			}

			matrix_create_random_permutation(work_matrix_H, work_matrix_H_T, permutation);
			matrix_echelonize_partial_plusfix(work_matrix_H, config.m4ri_k, n-k-l, this->matrix_data, 0, n-k-l, 0, this->permutation);

			// Extract the sub-matrices. the length of the matrix is only n-c + DOOM_nr but we need to copy everything.
			// TODO is "config.TrivialAppendRows" correct?
            mzd_submatrix(H, work_matrix_H, config.TrivialAppendRows, n - k - l, n - k, n - c + DOOM_nr);
			matrix_transpose(HT, H);

			//mzd_print(HT);

			Matrix_T<mzd_t *> HH((mzd_t *) H);
			if constexpr(!config.DOOM) {
				target.column_from_m4ri(work_matrix_H, n-c-config.LOWWEIGHT, config.TrivialAppendRows);
			}

			for (uint32_t itloop = 0; not_found && (itloop < config.intermediate_target_loops); ++itloop) {
				iT1.random();


#if	NUMBER_THREADS != 1
#pragma omp parallel default(none) shared(std::cout, L1, L2, cL1, cL2, HH, H, HT, hm1, hm2, iT1, target, not_found, loops) num_threads(threads)
#endif
				{
					// Tree: From now on everything is multithreading.
					// The number at the end of a variable indicates the level its used in.
					const uint32_t tid   = NUMBER_THREADS == 1 ? 0 : omp_get_thread_num();
					const uint64_t b_tid = lsize2 / threads;    // blocksize of each thread. Note that we use `lsize2`,
					// because we only iterate over L2.
					const uint64_t s_tid = tid * b_tid;         // start position of each thread;
					const uint64_t e_tid = ((tid == threads - 1) ? L2.size() : s_tid + b_tid);
					alignas(AlignmentLabel) IndexType npos[npos_size] = {IndexType(s_tid)};     // Array of loop indices. Which happen to be also the indices that
																// are saved in the bucket/hash map structure.
																// The idea is that we already set one position in the `indices array` of the hashmap
																// which needs to be copied into the hashmap if a match was found.
					IndexType pos1, pos2;   // Helper variable. -1 indicating that no match was found. Otherwise the position
											// within the `hm1->__buckets[pos]` array is returned.
					LoadType load1 = 0, load2 = 0;
					ArgumentLimbType data, data1;   // tmp variable. Depending on `l` this is rather a 64bit or 128bit wide value.
					alignas(AlignmentLabel) LabelContainerType label, label2, label3;

					// Pointer to the internal array of labels. Not that this pointer needs an offset depending on the thread_id.
					// Instead of access this internal array we increment the
					uint64_t *Lptr = (uint64_t *) L2.data_label() + (s_tid * llimbs_a);

					// set the initial values of the `npos` array. Somehow this is not done by OpenMP
					for (uint64_t j = 0; j < npos_size; ++j) { npos[j] = s_tid; }


					// we need to first init the baselist L1.
					// Note that `work_matrix_H_T` is already transposed in the `matrix_create_random_permutation` call.
					// Additionally, we have to keep a hash map of L1 for the whole inner loop.
					BJMM_fill_decoding_lists(L1, L2, cL1, cL2, HT, tid);
					OMP_BARRIER

					//ASSERT(check_list(L1, HH, tid));
					//ASSERT(check_list(L2, HH, tid));

					// initialize the buckets with -1 and the load array with zero. This needs to be done for all buckets.
					hm1->reset(tid);
					hm2->reset(tid);
					OMP_BARRIER

					auto extractor = [](const DecodingLabel &label) -> ArgumentLimbType {
						if constexpr (config.SAVE_FULL_128BIT) {
							return Extractor::template extract<n-config.k-128, n-config.k, n-config.k-l>(label);
						} else {
							return Extractor::template extract<n-config.k-l, n-config.k>(label);
						}
					};

					//  0                                           n-k
					// [      n-k-l n-k-l+b0    n-k-l+b0+b1          ]
					hm1->hash1(L1, L1.size(tid), tid, extractor);
					//hm1->hash(L1, tid);
					OMP_BARRIER
					hm1->sort(tid);
					OMP_BARRIER

					ASSERT(hm1->check_sorted());


					// image of a bucket element in the first level
					//          L1      L2
					//           \      /
					//   HM1 [l, [i1, i2]], with l = L1[i1] xor L2[i2]
					// do the first list join between L1 and L2. Save the results in iL1.
					for (; npos[1] < e_tid; ++npos[1], Lptr += llimbs_a) {
						// data has now the following form
						//   0              127/63 bits
						//  [xxxxxxxx|0000000]
						//  n-k-l...n-k    label position.
						if constexpr(config.DOOM) {
							data = Extractor::template extract<n-config.k-l, n-config.k>(Lptr);
						} else {
							if constexpr (config.SAVE_FULL_128BIT) {
								data = Extractor::template add<n-config.k-128, n-config.k, n-config.k-l>(Lptr, iT1.ptr());
							} else {
								data = Extractor::template add<n-config.k-l, n-config.k>(Lptr, iT1.ptr());
								ASSERT((hm1->check_label(data, L2, npos[1])));
							}
						}

						pos1 = hm1->find1(data, load1);
						if constexpr (chm1.b2 == chm1.b1) {
							// if a solution exists, we know that every element in this bucket given is a solution
							while (pos1 < load1) {
								npos[0] = hm1->__buckets[pos1].second[0];
								data1 = data ^ hm1->__buckets[pos1].first;
//
//								hm1->printbinary(hm1->__buckets[pos1].first);
//								std::cout << '\n';
//								hm1->printbinary(data);
//								std::cout << '\n';
//								hm1->printbinary(data1);
//								std::cout << '\n';
//
//								std::cout << L1.data_label(npos[0]).data() << " npos[0]:" << npos[0] << "\n";
//								std::cout << L2.data_label(npos[1]).data() << " npos[1]:" << npos[1] << "\n";
//								std::cout << iT1 << "\n";
								hm2->insert1(data1, npos, tid);
								pos1 += 1;
							}
						} else {
							// Fall back to the generic case
							while (HM1BucketIndexType(pos1) != HM1BucketIndexType(-1)) {
								data1 = hm1->template traverse<0, 1>(data, pos1, npos, load1);
								hm2->insert1(data1, npos, tid);
							}
						}
					}

					OMP_BARRIER
					// after everything was inserted into the second hash map, it needs to be sorted on the bits
					// [l1 + log(hm2.nr_buckets), ..., l)
					hm2->sort(tid);
					OMP_BARRIER
					ASSERT(hm1->check_sorted());
					ASSERT(hm2->check_sorted());
					OMP_BARRIER

					if constexpr(config.DOOM) {
						Lptr = (uint64_t *) DOOM_S_View->rows[0] + (s_tid * llimbs_a);
					} else {
						// reset tmp variables to be reusable in the second list join.
						Lptr = (uint64_t *) L2.data_label() + (s_tid * llimbs_a);
					}


					uint64_t upper_limit;
					if constexpr(config.DOOM) {
						// TODO das ist noch nicht 100% richitg, da der start Value noch nicht richtig berechnet wird. Momentan nur richtig für nrt = 1
						ASSERT(threads == 1);
						upper_limit = n - k;
					} else {
						upper_limit = e_tid;
					}

					// do the second lst join between L3, L4.
					for (; npos[3] < upper_limit; ++npos[3], Lptr += BaseList4Inc) {
						if constexpr(config.DOOM) {
							data = Extractor::template extract<n-config.k-l, n-config.k>(Lptr);
						} else {
							if constexpr (config.SAVE_FULL_128BIT) {
								data = Extractor::template add<n-config.k-128, n-config.k, n-config.k-l>(Lptr, target.ptr());
							} else {
								data = Extractor::template add<n-config.k-l, n-config.k>(Lptr, target.ptr());
							}
							ASSERT((hm1->check_label(data, L2, npos[3])));
						}
						pos1 = hm1->find1(data, load1);

						if constexpr(!config.DOOM  && !config.SAVE_FULL_128BIT) {
							LabelContainerType::add(label, L2.data_label(npos[3]).data(), target);
						}

						if constexpr ((chm1.b2 == chm1.b1) && (chm2.b2 == chm2.b1)) {
							// if a solution exists, we know that every element in this bucket given is a solution
							while (pos1 < load1) {
								npos[2] = hm1->__buckets[pos1].second[0];
								data1 = data ^ hm1->__buckets[pos1].first;
								pos1 += 1;

								pos2 = hm2->find1(data1, load2);
								if (pos2 != IndexType(-1)) {
									if constexpr(!config.DOOM) {
										// only precompute the label of the two right baselists if we do not have 128
										// bits in our hashmap to check the threshold.
										if constexpr (!config.SAVE_FULL_128BIT) {
											LabelContainerType::add(label2, label, L1.data_label(npos[2]).data());
										}
									} else {
										LabelContainerType::add(label2.data().data(), L1.data_label(npos[2]).data().data().data(), Lptr);
									}
								}

								while (pos2 < load2) {
									npos[0] = hm2->__buckets[pos2].second[0];
									npos[1] = hm2->__buckets[pos2].second[1];

									pos2 += 1;
									DEBUG_MACRO(last_list_counter +=1;)
//									hm2->printbinary(data1);
//									std::cout << " data 1\n";
//									hm2->printbinary(hm2->__buckets[pos2].first);
//									std::cout << " hm2 data\n";
//									hm2->printbinary(data1 ^ hm2->__buckets[pos2].first);
//									std::cout << " data ^ hm2 data\n";
//
//
//									std::cout << L1.data_label(npos[0]).data() << " npos[0]:" << npos[0] << "\n";
//									std::cout << L2.data_label(npos[1]).data() << " npos[1]:" << npos[1] << "\n";
//									std::cout << L1.data_label(npos[2]).data() << " npos[2]:" << npos[2] << "\n";
//									std::cout << L2.data_label(npos[3]).data() << " npos[3]:" << npos[3] << "\n";
//
//									std::cout << target << " target\n";
//									std::cout << iT1 << " iT1\n";

									// if activated this is an early exit of the last level of the tree.
									if constexpr (config.SAVE_FULL_128BIT) {
										const ArgumentLimbType data2 = data1 ^ hm2->__buckets[pos2-1].first;
										const uint32_t iweight = __builtin_popcountll(data2 >> 64u);

										if (likely(iweight > config.weight_threshhold)) {
											continue;
										}

										if (likely((__builtin_popcountll(data2) + iweight) > config.weight_threshhold)) {
											continue;
										}

										LabelContainerType::add(label3, L2.data_label(npos[1]).data(), L1.data_label(npos[0]).data());
										LabelContainerType::add(label2, L2.data_label(npos[3]).data(), L1.data_label(npos[2]).data());
										LabelContainerType::add(label2, label2, target);
									}

									if constexpr (!config.SAVE_FULL_128BIT) {
										LabelContainerType::add(label3, L2.data_label(npos[1]).data(), L1.data_label(npos[0]).data());
									}

									uint32_t weight;
									if constexpr(!config.DOOM) {
										#ifdef DEBUG
										weight = LabelContainerType::template add_only_upper_weight_partly_withoutasm<lupper, lumask>(label3, label3, label2);

										LabelContainerType ltmp;
										LabelContainerType::add(ltmp, label3, target);
										for (uint32_t i = n-config.k-l; i < n-config.k; ++i) {
											if ((label3[i] != 0 ) && (ltmp[i] != 0)) {
												std::cout << target << " target\n";
												std::cout << label3 << " error label3\n";
												std::cout << ltmp << " error ltmp\n";
												ASSERT(false);
											}
										}
										#else
										weight = LabelContainerType::template add_only_upper_weight_partly_withoutasm_earlyexit<lupper, lumask, config.weight_threshhold>(label3, label3, label2);
										#endif
									} else {
										weight = LabelContainerType::add_weight(label3.data().data(), label3.data().data(),label2.data().data());
									}

									if (likely(weight > config.weight_threshhold)) {
										continue;
									}
									check_final_list(label3, npos, weight, npos[3]);
								}
							}
						} else if constexpr(chm1.b2 == chm1.b1) {
							while (pos1 < load1) {
								npos[2] = hm1->__buckets[pos1].second[0];
								data1 = data ^ hm1->__buckets[pos1].first;
								pos1 += 1;

								pos2 = hm2->find1(data1, load2);
								if (pos2 != IndexType(-1)) {
									if constexpr(!config.DOOM) {
										//LabelContainerType::add(label, L1.data_label(npos[2]).data(), L2.data_label(npos[3]).data());
										//LabelContainerType::add(label, label, target);
										LabelContainerType::add(label2, label, L1.data_label(npos[2]).data());
									} else {
										LabelContainerType::add(label2.data().data(), L1.data_label(npos[2]).data().data().data(), Lptr);
									}
								}

								while (pos2 != IndexType(-1)) {
									hm2->template traverse_drop<0, 2>(data1, pos2, npos, load2);

									LabelContainerType::add(label3, L2.data_label(npos[1]).data(), L1.data_label(npos[0]).data());

									DEBUG_MACRO(last_list_counter +=1;)

									uint32_t weight;
									if constexpr(!config.DOOM) {
										weight = LabelContainerType::template add_only_upper_weight_partly_withoutasm_earlyexit<lupper, lumask, config.weight_threshhold>(label3, label3, label2);
									} else {
										weight = LabelContainerType::add_weight(label3.data().data(), label3.data().data(),label2.data().data());
									}

									if (weight > config.weight_threshhold) {
										continue;
									}
									check_final_list(label3, npos, weight, 0);
								}
							}
						} else {
							// Fallback code if the two hashmaps are not fully sorted via buckets
							while (HM1BucketIndexType(pos1) != HM1BucketIndexType(-1)) {
								data1 = hm1->template traverse<2, 1>(data, pos1, npos, load1);
								pos2 = hm2->find1(data1, load2);

								if constexpr(!config.DOOM) {
									LabelContainerType::add(label2, label, L1.data_label(npos[2]).data());
								} else {
									LabelContainerType::add(label2.data().data(), L1.data_label(npos[2]).data().data().data(), Lptr);
								}

								while (HM2BucketIndexType(pos2) != HM2BucketIndexType(-1)) {
									hm2->template traverse_drop<0, 2>(data1, pos2, npos, load2);
									LabelContainerType::add(label3, L2.data_label(npos[1]).data(), L1.data_label(npos[0]).data());

									uint32_t weight;
									if constexpr(!config.DOOM) {
										weight = LabelContainerType::template add_only_upper_weight_partly_withoutasm_earlyexit<lupper, lumask, config.weight_threshhold>(label3, label3, label2);
									} else {
										weight = LabelContainerType::add_weight(label3.data().data(), label3.data().data(),label2.data().data());
									}

									if (weight > config.weight_threshhold) {
										continue;
									}
									check_final_list(label3, npos, weight, 0);
								}
							}
						}
					}
				}

				// NOTE this can be removed?
				// OMP_BARRIER
				print_info(loops);

				//  update the global loop counter
				OUTER_MULTITHREADED_WRITE(
				if ((unlikely(loops % config.exit_loops) == 0)) {
					if (finished.load()) {
						return loops;
					}
				})

				// and last but least update the loop counter
				loops += 1;
			}
		}

#if !defined(BENCHMARK) && !defined(NO_LOGGING)
		double offset_loops = (double(loops) / double(Loops())) * double(100);
		std::cout << "loops/expected: " << loops << "/" << Loops() << " " << offset_loops << "%\n" << std::flush;
#endif
		return loops;
	}

	uint64_t run() {
		if constexpr(config.c == 0) {
			return BJMMF();
		} else {
			return BJMMF_Outer();
		}
	}


	/// checks every element in the list if the label matches to the given value
	/// \tparam DecodingList type of the list to check
	/// \param L	input list
	/// \param H	corresponding matrix s.t. label=Hvalue
	/// \param tid	thread id (unused)
	/// \param low	lower coordinate of the label to check, set to -1 to set check everything
	/// \param high higher coordinate of the label to check, set to -1 to set check everything
	/// \parma size set to a different value than -1 ti use this value as the listsize. usefulle if the list doesnt know its load.
	/// \return true/false
	template<class DecodingList>
	bool check_list(const std::vector<DecodingList> &L, const Matrix_T<mzd_t *> &H, const uint32_t tid,
	                const uint32_t low=-1, const uint32_t high=-1, size_t size=-1) {
		typedef typename DecodingList::ValueType ValueType;
		typedef typename DecodingList::LabelType LabelType;

		LabelType tmp;

		const uint32_t k_lower = low   == uint32_t(-1) ? 0 : low;
		const uint32_t k_higher = high == uint32_t(-1) ? LabelType::size() : high;
		for (auto &LL: L) {
			const size_t listsize = size == size_t(-1) ? LL.size() : size;
			for (size_t i = 0; i < listsize; ++i) {
				new_vector_matrix_product<LabelType, ValueType, mzd_t *>(tmp, LL.data_value(i), H);
				if (tmp.data().is_zero()) {
					continue;
				}
//				std::cout << LL.data_label(i) << "\n";
//				std::cout << LL.data_value(i) << "\n" << std::flush;

				if (!tmp.is_equal(LL.data_label(i), k_lower, k_higher)) {
					std::cout << tmp << "  should vs it at pos:" << i << ", tid:" << tid << "\n";
					std::cout << LL.data_label(i) << "\n";
					std::cout << LL.data_value(i) << "\n" << std::flush;
					//print_matrix("HT", this->HT);
					return false;
				}

				if (tmp.data().is_zero() || LL.data_value(i).data().is_zero() || LL.data_label(i).data().is_zero()) {
					std::cout << tmp << " any element is zero:" << i << "\n";
					std::cout << LL.data_label(i) << " label in table\n";
					std::cout << LL.data_value(i) << " value in table\n" << std::flush;
					//print_matrix("HT", this->HT);
					return false;
				}
			}
		}

		return true;
	}


	template<class DecodingList>
	bool check_list(const DecodingList &L1, const Matrix_T<mzd_t *> &H, const uint32_t tid,
					const uint32_t low=-1, const uint32_t high=-1, size_t size=-1) {
		bool ret = true;
#pragma omp barrier

		if (tid == 0) {
#pragma omp master
			{
				std::vector<DecodingList> L{L1};
				ret = check_list<DecodingList>(L, H, tid, low, high, size);
			}
		}
#pragma omp barrier

		return ret;
	}

	// TODO remove
	// Only valid if config.d == 3 specified.
	constexpr static uint64_t S3() { return (S1() * S1()) >> l2; }

	// Expected size of the intermediate list.
	constexpr static uint64_t S1() { return (lsize2 * lsize2) >> l1; }

	// list size of the output list.
	constexpr static uint64_t S0() {
		if constexpr(config.d == 3) {
			return (S3() * S3()) >> (l - l2 - l1);
		} else {
			return (S1() * S1()) >> (l - l1);
		}
	}

	// Outer loops if c != 0
	constexpr static double LogOuterLoops(const uint64_t nn, const uint64_t cc) {
		// log(binomial(n, w)/binomial(n-c, w), 2).n()
		const double nn_ = double(nn);
		const double w_ = double(w) / double(nn);
		const double cc_ = double(cc) / nn_;

		const double shift = HH(w_) - (1 - cc_) * HH(w_);
		return shift;
	}

	constexpr static double OuterLoops(const uint64_t nn, const uint64_t cc) {
		return pow(2, LogOuterLoops(nn, cc));
	}

	/// returns the expected numbers of loops needed
	/// \param nn code length
	/// \param cc dimensions to cut off.
	/// \return
	constexpr static double LogLoops(const uint64_t nn, const uint64_t cc=0) noexcept {
		// note that p is here only the base p.
		// binom(n, w)/(binom(n-k-l, w-4p) * binom((k+l)/2 - cc/2, 2*p)**2)
		// log((binomial(n, w) / (binomial(n-k-l, w-4*p) * binomial((k+l)/2, 2*p)**2)), 2).n()
		// log((binomial(n-c, w) / (binomial(n-k-l, w-4*p) * binomial((k+l)/2 - c//2, 2*p)**2)) * binomial(n, w)/binomial(n-c, w), 2).n()
		const double nn_ = double(nn);
		const double k_ = double(k) / nn_;
		const double l_ = double(l) / nn_;
		const double p_ = (double(2*p)) / nn_;
		const double wp_ = double(w - (4 * p)) / nn_;
		const double w_ = double(w) / nn_;

		const double ccc = cc == 0 ? 0. : double(cc) / nn_ / 2.;

		const double t1 = double(k_ + l_)/2. - ccc;
		const double t2 = double(1. - k_ - l_);

		const double t1_ = HH(p_ / t1);
		const double t2_ = HH(wp_ / t2);

		const double t11 = t1 * t1_;
		const double t22 = t2 * t2_;

		const double t3 = HH(w_) - 2. * t11 - t22;
		const double shift = t3 * nn;
		return shift;
	}

	constexpr static double Loops(const uint64_t nn, const uint64_t cc=0) noexcept {
		return pow(2, LogLoops(nn, cc));
	}

	static double Loops() {
		return Loops(n);
	}

	/// prints current loops information like: Hashmap usage,..
	/// \param loops
	void __attribute__ ((noinline))
	print_info(uint64_t loops) {
#if !defined(BENCHMARK) && !defined(NO_LOGGING)
		if (unlikely((loops % config.print_loops) == 0)) {
			std::cout << "BJMMF: tid:" << omp_get_thread_num() << ", loops: " << loops << "\n";
			std::cout << "log(inner_loops): " << LogLoops(n, c) << ", inner_loops: " << Loops(n, c) ;
			if constexpr(c != 0) {
				std::cout << ", log(outer_loops): " << LogOuterLoops(n, c) << ", outer_loops: " << OuterLoops(n, c) ;
			}
			double ctime = ((double)clock()-internal_time)/CLOCKS_PER_SEC;
			std::cout << "Time: " << ctime << ", clock Time: " << ctime/config.nr_threads << "\n";
#ifndef CHALLENGE
			hm1->print();
			hm2->print();
			std::cout << "Exp |hm1|=" << lsize1 << ", Exp |hm2|=" << S1() << ", Exp |out|=" << S0()
			#ifdef DEBUG
			<< ":" << last_list_counter/MAX(1, loops) << "\n\n" << std::flush;
			#else
			<< "\n\n" << std::flush;
			#endif
#endif
		}
#endif
	}
};

#endif //SMALLSECRETLWE_BJMM_H
#endif