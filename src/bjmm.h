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

	// number of threads work in parallel on the same tree
	const uint32_t nr_threads;

	// number of instances work in parallel on different parallel
	const uint32_t nr_outer_threads;

	// Overlapping of the two baselists halves
	// BaseList size tuning parameter
	const uint32_t epsilon = 0;

	// how many iterations
	const uint32_t intermediate_target_loops = 1;//1u << 7u;

	// Scaling factor for the different sorting/searching datastructures
	const float scale_bucket = 1.0;

	// Number of Elements to store in each bucket within each hash map.
	const uint64_t size_bucket1;
	const uint64_t size_bucket2;

	// Number of buckets in each hashmap.
	const uint64_t number_bucket1;
	const uint64_t number_bucket2;

	// DO NOT USE, deprecated
	const bool ClassicalTree = false;

	// do not generate or save the values in the base lists, but regenerate on the fly.
	const bool no_values = false;

	// Decode One Out of Many.
	// Only usable if you want to break QuasiCyclic code.
	// Look at the BJMM constructor to see a detailed graphic explaining everything.
	// This enables:
	//  - all shifts of the syndrome are appended to the working matrix.
	//  - No target will be added to the baselists
	//  - No intermediate target will be used
	const bool DOOM = false;

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

	// If this flag is set to true, the tree will save in the hashmaps the whole 128 bits regardless of the given hashmap
	// bucket size. This allows the tree to check fast if an element in the last level exceeds the weight threshold or not.
	const bool HM1_SAVE_FULL_128BIT_SWITCH      = false;
	const bool HM2_SAVE_FULL_128BIT_SWITCH      = false;

	// This flag configures the internal hashmaps. If set to `true` the data container within the hashmaps, normally
	// saving the lpart and the indices, get a third data container holding at least 128bit of the label.
	const uint8_t HM1_EXTEND_TO_TRIPLE_SWITCH   = 0;
	const uint8_t HM2_EXTEND_TO_TRIPLE_SWITCH   = 0;

	// enforces the hashmap to prefetch data on a successful find
	const bool HM1_USE_PREFETCH                 = false;
	const bool HM2_USE_PREFETCH                 = false;

	// This flag enforces the hashmaps to use a `atomic::uint32_t` instead of  `uint32_t`
	// as the core type of the load factors. This enables multiple threads to work
	// on the same array (and not on disjunct parts). This allows for reduced size of the
	// buckets in the multiple thread setting.
	const bool HM1_USE_ATOMIC_LOAD              = false;
	const bool HM2_USE_ATOMIC_LOAD              = false;

	// This flag enforces the hashmap to use a data structure which is packed (e.g.
	// the different fields of the struct are not aligned).
	const bool HM1_USE_PACKED                   = true;
	const bool HM2_USE_PACKED                   = true;

	// Set 0 zero to disable.
	// See `BJMM::append_trivial_rows`
	// If this value is set to anything else than zero, this number of rows will be appended to the working matrix wH.
	// The goal is to decrease the code rate and therefore increase the performance of the algorithms.
	const uint32_t TrivialAppendRows = ClassicalTree ? 0 : (d == 3 ? 0 : (DOOM ? 0 : (c != 0 ? 0 : 1)));

	// If this flag is set, the algorithm transform into its high weight variant.
	// This means that for `finding` matches in the hashmaps we will not use any load factor
	// anymore. The load factor is only used for inserting now. This reduces the cache
	// misses for a find from 2 to 1.
	bool HighWeightVariant  = false;

	// if set to `true` an optimized m4ri version is used. This optimized version tracks along which
	// unity vectors are permuted into/out the information set. Base on this, it re-permutes this vectors
	// s.t. the gaussian elimination does not need to work through n-k-l vectors, but only those, which
	// are not systemizes
	bool OptM4RI            = true;

	// If this options is set != 1 and 'OptM4RI' is true, not a random permutation is chosen, but
	// only 'gaus_c' columns are permuted. NOTE: in this case an adapted runtime analyse must be used.
	uint32_t gaus_c         = 0;

// Stop the outer loop after `loops` steps. Useful for benchmarking.
#ifdef USE_LOOPS
	const uint64_t loops        = USE_LOOPS;
#else
	const uint64_t loops        = uint64_t(-1);
#endif

	// print some useful information about the current state of the program every X loops
#ifdef PRINT_LOOPS
	const uint64_t print_loops  = PRINT_LOOPS;
#else
	const uint64_t print_loops  = 1000;
#endif

#ifdef EXIT_LOOPS
	const uint64_t exit_loops  = EXIT_LOOPS;
#else
	const uint64_t exit_loops  = 10000;
#endif

	// custom settable seed. If set to any other value than `0` this will be used to seed the internal prng.
	const uint64_t seed = 0;

	// Constructor
	constexpr ConfigBJMM(const uint32_t n, const uint32_t k, const uint32_t w, const uint32_t baselist_p,
	                     const uint32_t l, const uint32_t l1,
	                     const uint64_t sb1 = 100, const uint64_t sb2 = 100,    // Size of the buckets
	                     const uint64_t nb1 = 5,   const uint64_t nb2 = 5,      // logarithmic number of buckets
	                     const uint32_t threshold = 4,                          // threshold
	                     const uint32_t nt = 1,                                 // number of threads
	                     const uint32_t nr_outer_threads = 1,                   // number of outer threads
	                     const bool DOOM = false,                               // Decode one out of many? (Only valid for BIKE instances)
	                     const bool Baselist_Full_Length = false,               // instead, an MITM enumeration in the baselist, enumrate the weight p on every coordinate
	                     const uint32_t c = 0,                                  // cutoff dimension
	                     const bool LOWWEIGHT=false,                            // if set to true, the tree will try to match on zero, faking a syndrome != 0^n-k
	                     const uint32_t l2 = 0,                                 // Indyk Motwani coordinates to apply NN on
						 const uint32_t IM_nr_views=0,                          // number of l2 windows we apply a NN search on
	                     const double iFactor=1.0,                              // scaling factor of the list
	                     const bool no_values=false,                            // do not calculate the values and do not allocate space for them.
	                     const bool HM1_STDBINARYSEARCH_SWITCH=true, const bool HM2_STDBINARYSEARCH_SWITCH=true,            // if b1 != b2 meaning we do not match on very coordinate in the hashmaps, we need to apply a search on it. And the best one is the std search
	                     const bool HM1_INTERPOLATIONSEARCH_SWITCH=false, const bool HM2_INTERPOLATIONSEARCH_SWITCH=false,  // same as std::search, but a custom implementation of an interpolation search
	                     const bool HM1_LINEAREARCH_SWITCH=false, const bool HM2_LINEAREARCH_SWITCH=false,                  // same as std::search, but a linear search is done
	                     const bool HM1_USE_LOAD_IN_FIND_SWITCH=true, const bool HM2_USE_LOAD_IN_FIND_SWITCH=true,          // if set to true, the hashmap tries to encode the load of the bucket into the elements itself.
	                     const bool HM1_SAVE_FULL_128BIT_SWITCH=false, const bool HM2_SAVE_FULL_128BIT_SWITCH=false,        // if set to true, the hashmap saves 128 bit regardless of b2
	                     const uint8_t HM1_EXTEND_TO_TRIPLE_SWITCH=0, const uint8_t HM2_EXTEND_TO_TRIPLE_SWITCH=0,          // if set to true, the hashmap gets one extra value in each element.
	                     const bool HM1_USE_PREFETCH=false, const bool HM2_USE_PREFETCH=false,                              //
	                     const bool HM1_USE_ATOMIC_LOAD=false, const bool HM2_USE_ATOMIC_LOAD=false,                        //
	                     const bool HM1_USE_PACKED=true, const bool HM2_USE_PACKED=true,                                    //
	                     const bool high_weight=false,                                                                      //
	                     const uint32_t intermediate_target_loops=1,                                                        //
	                     const uint32_t seed=0,                                                                             //
	                     const uint32_t gaus_c=0,                                                                           //
	                     const bool gaus_opt=true                                                                           //
	) :
							 n(n),
	                         k(k),
	                         w(w),
	                         baselist_p(baselist_p),
	                         l(l),
	                         l1(l1),
	                         weight_threshhold(threshold),
	                         m4ri_k(matrix_opt_k(n - k, MATRIX_AVX_PADDING(n))),
	                         nr_threads(nt),
	                         nr_outer_threads(nr_outer_threads),
	                         intermediate_target_loops(intermediate_target_loops),
	                         scale_bucket(iFactor),
							 size_bucket1(sb1), size_bucket2(sb2),
							 number_bucket1(nb1), number_bucket2(nb2),
	                         no_values(no_values),
							 DOOM(DOOM),
							 Baselist_Full_Length(Baselist_Full_Length),
	                         LOWWEIGHT(LOWWEIGHT),
	                         c(c),
	                         IM_nr_views(IM_nr_views),
	                         l2(l2),
							 HM1_STDBINARYSEARCH_SWITCH(HM1_STDBINARYSEARCH_SWITCH),
							 HM1_INTERPOLATIONSEARCH_SWITCH(HM1_INTERPOLATIONSEARCH_SWITCH),
							 HM1_LINEAREARCH_SWITCH(HM1_LINEAREARCH_SWITCH),
							 HM1_USE_LOAD_IN_FIND_SWITCH(HM1_USE_LOAD_IN_FIND_SWITCH),
							 HM2_STDBINARYSEARCH_SWITCH(HM2_STDBINARYSEARCH_SWITCH),
							 HM2_INTERPOLATIONSEARCH_SWITCH(HM2_INTERPOLATIONSEARCH_SWITCH),
							 HM2_LINEAREARCH_SWITCH(HM2_LINEAREARCH_SWITCH),
							 HM2_USE_LOAD_IN_FIND_SWITCH(HM2_USE_LOAD_IN_FIND_SWITCH),
	        				 HM1_SAVE_FULL_128BIT_SWITCH(HM1_SAVE_FULL_128BIT_SWITCH),
	        				 HM2_SAVE_FULL_128BIT_SWITCH(HM2_SAVE_FULL_128BIT_SWITCH),
	        				 HM1_EXTEND_TO_TRIPLE_SWITCH(HM1_EXTEND_TO_TRIPLE_SWITCH),
	        				 HM2_EXTEND_TO_TRIPLE_SWITCH(HM2_EXTEND_TO_TRIPLE_SWITCH),
							 HM1_USE_PREFETCH(HM1_USE_PREFETCH), HM2_USE_PREFETCH(HM2_USE_PREFETCH),
	                         HM1_USE_ATOMIC_LOAD(HM1_USE_ATOMIC_LOAD), HM2_USE_ATOMIC_LOAD(HM2_USE_ATOMIC_LOAD),
	                         HM1_USE_PACKED(HM1_USE_PACKED), HM2_USE_PACKED(HM2_USE_PACKED),
	                         HighWeightVariant(high_weight),
	                         OptM4RI(gaus_opt),
							 gaus_c(gaus_c),
	                         seed(seed)
	{};

	// prints information about the problem instance.
	void print() const {
		std::cout << "n: " << n
		          << ", k: " << k
		          << ", c: " << c
		          << ", p: " << baselist_p
		          << ", l: " << l
		          << ", l1: " << l1
				  << ", IM_nr_views: " << IM_nr_views
				  << ", l2: " << l2
				  << ", DOOM: " << DOOM
		          << ", log(#buckets1): " << number_bucket1 << ", log(#buckets2): " << number_bucket2
		          << ", size_bucket1: " << size_bucket1 << ", size_bucket2: " << size_bucket2
		          << ", threads: " << nr_threads
				  << ", outer_threads: " << nr_outer_threads
		          << ", m4ri_k: " << m4ri_k
		          << ", weight_threshhold: " << weight_threshhold
		          << ", loops: " << loops
		          << ", print_loops: " << print_loops
		          << ", exit_loops: " << exit_loops
		          << ", intermediate_target_loops:" << intermediate_target_loops
		          << ", epsilon: " << epsilon
		          << ", Baselist_Full_Length: " << Baselist_Full_Length
		          << ", TrivialAppendRows: " << TrivialAppendRows
				  << ", no_values: " << no_values
				  << ", scale_bucket: " << scale_bucket
		          << ", intermediate_target_loops: " << intermediate_target_loops
				  << ", OptM4RI: " << OptM4RI
		          << ", gaus_c: " << gaus_c
				  << ", HM1_STDBINARYSEARCH_SWITCH: " << HM1_STDBINARYSEARCH_SWITCH
				  << ", HM1_INTERPOLATIONSEARCH_SWITCH: " << HM1_INTERPOLATIONSEARCH_SWITCH
				  << ", HM1_LINEAREARCH_SWITCH: " << HM1_LINEAREARCH_SWITCH
				  << ", HM1_USE_LOAD_IN_FIND_SWITCH: " << HM1_USE_LOAD_IN_FIND_SWITCH
				  << ", HM2_STDBINARYSEARCH_SWITCH: " << HM2_STDBINARYSEARCH_SWITCH
				  << ", HM2_INTERPOLATIONSEARCH_SWITCH: " << HM2_INTERPOLATIONSEARCH_SWITCH
				  << ", HM2_LINEAREARCH_SWITCH: " << HM2_LINEAREARCH_SWITCH
				  << ", HM2_USE_LOAD_IN_FIND_SWITCH: " << HM2_USE_LOAD_IN_FIND_SWITCH
				  << ", HM1_SAVE_FULL_128BIT_SWITCH: " << HM1_SAVE_FULL_128BIT_SWITCH
				  << ", HM2_SAVE_FULL_128BIT_SWITCH: " << HM2_SAVE_FULL_128BIT_SWITCH
				  << ", HM1_EXTEND_TO_TRIPLE_SWITCH: " << int(HM1_EXTEND_TO_TRIPLE_SWITCH)
				  << ", HM2_EXTEND_TO_TRIPLE_SWITCH: " << int(HM2_EXTEND_TO_TRIPLE_SWITCH)
				  << ", HM1_USE_PREFETCH: " << HM1_USE_PREFETCH
		          << ", HM2_USE_PREFETCH: " << HM2_USE_PREFETCH
				  << ", HM1_USE_ATOMIC_LOAD: " << HM1_USE_ATOMIC_LOAD
				  << ", HM2_USE_ATOMIC_LOAD: " << HM2_USE_ATOMIC_LOAD
		          << ", HM1_USE_PACKED: " << HM1_USE_PACKED
		          << ", HM2_USE_PACKED: " << HM2_USE_PACKED
				  << ", HighWeightVariant: " << HighWeightVariant
		          << "\n";
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
	constexpr static uint32_t n   = config.n;   // code length
	constexpr static uint32_t k   = config.k - config.TrivialAppendRows;  // See the explanation of the function `TrivialAppendRows`
	constexpr static uint32_t w   = config.w;   // code weigh
	constexpr static uint32_t p   = config.baselist_p; // NOTE: from now on p is the baselist p not the full p.
	constexpr static uint32_t l   = config.l;   // number of bits to match in total
	constexpr static uint32_t l1  = config.l1;  // number of bits to match in the first level
	constexpr static uint32_t l2  = config.l2;  // number of bits to match in the second level (NOT used by the normal bjmm, only by d3 and MO)
	constexpr static uint32_t nkl = n - config.k - l;   // little helper
	constexpr static uint32_t c   = config.c;   // number of columns to cut of the working matrix.

	using DecodingValue     = Value_T<BinaryContainer<k+l-c>>;
	using DecodingLabel     = Label_T<BinaryContainer<n-config.k>>;
	using DecodingMatrix    = mzd_t *;
	using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList      = Parallel_List_T<DecodingElement>;
	using ChangeElement     = std::pair<uint16_t, uint16_t>;
	using ChangeList        = std::vector<ChangeElement>;

	typedef typename DecodingList::ValueContainerType ValueContainerType;
	typedef typename DecodingList::LabelContainerType LabelContainerType;
	typedef typename DecodingList::ValueContainerLimbType ValueContainerLimbType;
	typedef typename DecodingList::LabelContainerLimbType LabelContainerLimbType;

	// constant variables for the tree merge algorithm
	constexpr static uint32_t threads = config.nr_threads;

	// precalculate the expected size of the baselists. For the sake of simplicity we assume that all lists within the
	// tree structure is of the same size. Additionally, we precalculate the size of each part a single thread is working on.
	constexpr static uint64_t
			lsize1 = config.Baselist_Full_Length ?
					bc(k+l-c, p) :
					bc(config.epsilon + ((k + l - c) / 2), p),
			lsize2 = config.Baselist_Full_Length ?
					bc(k+l-c-config.LOWWEIGHT, p) :
					bc(config.epsilon + ((k + l - c -config.LOWWEIGHT) - (k + l - c) / 2), p);
	constexpr static uint64_t thread_size_lists1 = lsize1 / threads, thread_size_lists2 = lsize2 / threads;

	// list per thread size.
	constexpr static uint32_t tL1len = lsize1 / threads; // TODO remove listen aufgabe
	constexpr static uint32_t tL2len = lsize2 / threads;

	// we choose the datatype depending on the size of l.
	// container of the `l` part if the label
	constexpr static uint32_t ArgumentLimbType_l = config.HM1_SAVE_FULL_128BIT_SWITCH ? 128 : (config.IM_nr_views == 0 ? l : std::max(l, l2));
	using ArgumentLimbType  = LogTypeTemplate<ArgumentLimbType_l+l2>; // TODO oskd;sjdf;
	// container of an index in the changelist and the hashmaps.
	using IndexType         = TypeTemplate<std::max(
												std::max(size_t((uint64_t(1u) << config.number_bucket1) * config.size_bucket1),
														 size_t((uint64_t(1u) << config.number_bucket2) * config.size_bucket2)),
														 size_t((uint64_t(1u) << config.number_bucket3) * config.size_bucket2))>; // TODO correct?
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
												   config.HM1_SAVE_FULL_128BIT_SWITCH, config.HM1_EXTEND_TO_TRIPLE_SWITCH,
	                                               config.HM1_USE_PREFETCH, config.HM1_USE_ATOMIC_LOAD,
	                                               config.HighWeightVariant,
	                                               config.HM1_USE_PACKED
	};
	constexpr static ConfigParallelBucketSort chm2{b20, b21, b22, config.size_bucket2,
	                                               uint64_t(1) << config.number_bucket2, threads, 2, n-config.k-l, l, 0, 0,
	                                               config.HM2_STDBINARYSEARCH_SWITCH, config.HM2_INTERPOLATIONSEARCH_SWITCH,
												   config.HM2_LINEAREARCH_SWITCH, config.HM2_USE_LOAD_IN_FIND_SWITCH,
												   config.HM2_SAVE_FULL_128BIT_SWITCH, config.HM2_EXTEND_TO_TRIPLE_SWITCH,
	                                               config.HM2_USE_PREFETCH, config.HM2_USE_ATOMIC_LOAD,
	                                               config.HighWeightVariant,
	                                               config.HM2_USE_PACKED
	};

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
	DecodingList L1{lsize1, threads, thread_size_lists1, config.no_values},
	             L2{lsize2, threads, thread_size_lists2, config.no_values};

	// Hashmaps. Fast (constant) insert and look up time.
	template<const uint32_t l, const uint32_t h>
	inline static ArgumentLimbType Hash(uint64_t a) noexcept {
		static_assert(l < h);
		static_assert(h < 64);
		constexpr uint64_t mask = (~((uint64_t(1u) << l) - 1u)) &
		                          ((uint64_t(1u) << h) - 1u);
		return (uint64_t(a) & mask) >> l;
	}

	// Types of the hasmpas.
	using HM1Type = ParallelBucketSort<chm1, DecodingList, ArgumentLimbType, IndexType, &Hash<b10, b11>>;
	using HM2Type = ParallelBucketSort<chm2, DecodingList, ArgumentLimbType, IndexType, &Hash<b20, b21>>;
	using HM1BucketIndexType = typename HM1Type::BucketIndexType;
	using HM2BucketIndexType = typename HM2Type::BucketIndexType;
	HM1Type *hm1; HM2Type *hm2;

	// crucial class, to extract arbitrary bits from a given array of limbs.
	using Extractor = WindowExtractor<DecodingLabel, ArgumentLimbType>;

	// actual extractor function
	static inline ArgumentLimbType extractor(const DecodingLabel &label) noexcept {
		return extractor_ptr(label.ptr());
	};

	static inline ArgumentLimbType extractor_ptr(const uint64_t *__restrict__ label) noexcept {
		if constexpr(config.DOOM) {
			return Extractor::template extract<n-config.k-l, n-config.k>(label);
		}

		if constexpr (config.HM1_SAVE_FULL_128BIT_SWITCH) {
			return Extractor::template extract<n-config.k-128, n-config.k, n-config.k-l>(label);
		} else {
			return Extractor::template extract<n-config.k-l, n-config.k>(label);
		}
	};

	// Equivalent to the number of baselists.
	constexpr static uint32_t npos_size = config.d == 3 ? 8 : 4;

	// needed to randomize the baselists.
	// alignment needed for the special alignment flag
	alignas(32) DecodingLabel target;

	// this is variable hold the `l`/`128` bits shifted down do zero.
	ArgumentLimbType iTarget=0, iTarget_org=0, iT1=0;

	// Test/Bench parameter
	uint32_t ext_tid = 0;
	bool not_found = true;

	// if set to true the internal hashmaps weill be allocated.
	bool init = true;

	// just a helper value to count the elements which survive to the last level.
	DEBUG_MACRO(uint64_t last_list_counter = 0;)

	// measures the time without alle the preprocessing and allocating
	double internal_time;

	// count the loops we iterated
	uint64_t loops = 0;

	DEBUG_MACRO(uint64_t LowWeight_LVL2_Weight_Counter = 0;)

	/// \param e instance parameter
	/// \param s instance parameter
	/// \param A instance parameter
	/// \param ext_tid 	if this constructor is called within different threads, pass a unique id via this flag.
	///					This Flag is then passed to the internal rng.
	/// \param not_init	Do not init the internal data structures like hashmap etc. Useful if a derived class
	///			constructor is called.
	BJMM(mzd_t *__restrict__ e,
	     const mzd_t *__restrict__ const s,
	     const mzd_t *__restrict__ const A,
	     const uint32_t ext_tid = 0,
	     const bool init=true,
	     const bool init_matrix=true)
			noexcept : e(e), s(s), A(A), ext_tid(ext_tid), init(init) {
		static_assert(n > k, "wrong dimension");
		static_assert(config.number_bucket1 <= l1, "wrong hm1 #bucket");
		if constexpr(l2 == 0) {
			static_assert(config.number_bucket2 <= l, "wrong hm2 #bucket");
		}

		if constexpr (!config.HM1_USE_ATOMIC_LOAD && config.HM2_USE_ATOMIC_LOAD) {
			static_assert((config.size_bucket1 % threads == 0) && (config.size_bucket2 % threads == 0), "wrong #threads");
			static_assert((threads <= config.size_bucket1) && (threads <= config.size_bucket2), "wrong #threads");
		}
		static_assert((config.DOOM + config.LOWWEIGHT) < 2, "DOOM AND LOWWEIGHT are not valid.");
		static_assert(((config.c != 0) + config.LOWWEIGHT) < 2, "CUTOFF AND LOWWEIGHT are not valid.");
		if constexpr(config.HM1_SAVE_FULL_128BIT_SWITCH || config.HM2_SAVE_FULL_128BIT_SWITCH) {
			static_assert(n-k >= 128);
		}

		// reset the found indicator variable.
		not_found = true;

		// Make sure that some internal details are only accessed by the omp master thread.
		#pragma omp master
		{
#if !defined(NO_LOGGING)
			if (config.loops != uint64_t(-1)) {
				std::cout << "IMPORTANT: TESTMODE: only " << config.loops << " permutations are tested\n";
			}
			chm1.print();
#if defined(USE_MO) && USE_MO == 0
			chm2.print();
#endif
			config.print();
#endif

		}

		// Seed the internal prng.
		if (config.seed != 0) {
			random_seed(config.seed + ext_tid);
		}

		// Ok this is ridiculous.
		// Apparently m4ris mzd_init is not thread safe. Cool.
		#pragma omp critical
		{
			// Transpose the input syndrome
			sT = mzd_init(s->ncols, s->nrows);
			mzd_transpose(sT, s);

			if (init_matrix) {
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

					if constexpr (config.c == 0) {
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

					H = mzd_init(n - config.k, k + l - c + DOOM_nr);
					HT = matrix_init(H->ncols, H->nrows);
					DOOM_S_View = mzd_init_window(HT, k + l - c, 0, k + l + DOOM_nr - c, HT->ncols);
				} else {
					if constexpr (config.c == 0) {
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

						   H = mzd_init(n - config.k, k + l - c);
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
		}

		// Init the targets, intermediate targets, ...
		target.zero();
		iT1 = 0;
		iTarget = 0;

		// set to false if the constructor is called from a derived class
		if (init) {
			// initialize non-constant variables: HashMaps
			hm1 = new HM1Type();
			// TODO remove, this is stupid
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
		list_precompute_time = (double)clock() - list_precompute_time;

#if !defined(BENCHMARK) && !defined(NO_LOGGING)
		std::cout << "Precomputation took: " << ((double)clock() - list_precompute_time) / (CLOCKS_PER_SEC) << " s\n";
		std::cout << "Init took: " << ((double)clock() - inittime) / (CLOCKS_PER_SEC) << " s\n";
#endif

		ASSERT(L1.size() == cL1.size());
	}

	~BJMM() noexcept {
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


	static bool check_matrix(const mzd_t *A, const uint32_t max_row) {
		const uint32_t limit = A->nrows-config.TrivialAppendRows;

		// check for unity matrix on n-k-l
		for (uint32_t i = 0; i < limit; ++i) {
			for (uint32_t j = 0; j < max_row; ++j) {
				if (i == j) {
					if (mzd_read_bit(A, i, j) != 1) {
						return false;
					}
				} else {
					if (mzd_read_bit(A, i, j) != 0) {
						return false;
					}
				}
			}
		}

		return true;
	}

	///
	/// \param P array containing the the bit positions of the value at position 'pos'
	/// \param pos pos of the element in the value list the set bits need to be calculated
	/// \param side if == 0: L1 is chosen
	///					  1: L2 is chosen
	static void get_bits_set(uint32_t P[p], const size_t pos, const uint32_t side,
	                         ChangeList &cL1, ChangeList &cL2) noexcept {
		// offset between L1 and L2
		constexpr uint32_t n_full = config.k + config.l - config.c - config.TrivialAppendRows;
		
		// first catch the trivial case
		if (pos <= p) {
			for (uint32_t i = 0; i < p; ++i) {
				P[i] = i;
			}

			goto get_bits_set_finish;
		}
		
		if constexpr (p == 1) {
			P[0] = pos;
		} else {
			// calc direction of the chase sequence
			// if positiv -> going from left to right
			// if negative -> going from right to left
			auto calc_direction = [](const ChangeElement &e) -> int32_t {
				return int32_t(e.second) - 	int32_t(e.first);
			};

			// if negativ 1
			// else 0
			auto calc_sign = [](const int32_t a) -> bool {
				return a < 0;
			};

			// current change list
			auto *ccL = side == 0 ? &cL1 : &cL2;

			// set the first bit position
			P[0] = ccL->at(pos).second;

			// set the intial direction of the change sequence
			int32_t direction = calc_direction(ccL->at(pos)),
			        direction2 = direction;

			// number of bits set
			uint32_t cp = 1;

			// current position in the array
			for (size_t cpos = pos-1; cpos > 0; --cpos) {
				direction = calc_direction(ccL->at(cpos));

				// if the sign changed we found the next bit set
				if (calc_sign(direction) != calc_sign(direction2)) {
					P[cp] = ccL->at(cpos).second;
					cp += 1;

					// restart
					direction2 = direction;

					// return if we found enough bits
					if (cp == p)
						return;
				}
			}

			return;
		}
	
		get_bits_set_finish:
		if (side == 1) {
			// fix for the left side (L2)
			for (uint32_t i = 0; i < p; ++i) {
				P[i] += n_full / 2;
			}
		}
	}


	/// IMPORTANT Notes:
	///		- does not resize L1 or L2, so they need to set already
	///		- epsilon is implemented
	/// This function is intentionally quite abstract,to allow all other subclasses to us it.
	/// This function generates a MITM chase sequence and fill the baselists values accordingly
	/// \param bL1			output: left base list
	/// \param bL2			output: right base list
	/// \param diff_list1	output: change list for the left base list
	/// \param diff_list2	output: change list for the right baselist
	template<typename DecodingList=DecodingList>
	static void BJMM_prepare_generate_base_mitm2_with_chase2(DecodingList &bL1, DecodingList &bL2,
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

		if constexpr (config.no_values == false) {
			bL1.data_value(0) = e11;
			bL2.data_value(0) = e21;
		}

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

				if constexpr (config.no_values == false) { bL1.data_value(c1) = e11; }
				cL1[c1 - 1] = std::pair<uint32_t, uint32_t>(pos1, pos2);
				c1 += 1;
			}

			if (c2 < lsize2) {
				e22 = e21;
				ccb_l2.left_step(e21.data().data().data());
				diff_index(e21, e22);

				if constexpr (config.no_values == false) { bL2.data_value(c2) = e21; }
				cL2[c2 - 1] = std::pair<uint32_t, uint32_t>(pos1, pos2);
				c2 += 1;
			}
		}
	}

	/// basically the same as the mitm function, only filling it on the whole length
	constexpr void prepare_baselist_fulllength_mitm_with_chase2() noexcept {
		DecodingValue e11{}, e12{}, e21{}, e22{};
		e11.zero(); e12.zero(), e21.zero(); e22.zero();

		constexpr uint32_t n_full = config.k + config.l - config.c - config.TrivialAppendRows;
		constexpr uint32_t p = config.baselist_p;
		constexpr uint32_t limbs = ValueContainerType::limbs();

		// resize the data.
		// if we are in the doom setting, we want to enumerate L1 and L3 on the full length but L2 only on one half.
		constexpr bool l2_halfsize = config.DOOM;
		//constexpr uint64_t lsize1 = bc(n_full, p);
		//constexpr uint64_t lsize2 = l2_halfsize ? bc(((k + l - c) - (k + l - c) / 2), p) : lsize1;
		//ASSERT(lsize1 == this->lsize1 && lsize2 == this->lsize2);

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
	constexpr static inline void internal_xor_helper(DecodingLabel &a, const word *b) noexcept {
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
	static void BJMM_fill_decoding_lists(DecodingList &L1, DecodingList &L2,
	                         ChangeList &cL1, ChangeList &cL2,
	                         const mzd_t *HT, const uint32_t tid,
							 const bool add_target=false, const DecodingLabel *iTarget=nullptr) noexcept {
		const size_t start1 = tid * tL1len;   // starting index within the list L1 of each thread
		const size_t start2 = tid * tL2len;   // starting index within the list L2 of each thread
		// const size_t end1 = tid == (config.nr_threads - 1) ? L1.size() : start1 + this->tL1len;  // ending index of each thread within the list L1,
		// const size_t end2 = tid == (config.nr_threads - 1) ? L2.size() : start2 + this->tL2len;  // exepct for the last thread. This needs
		const size_t end1 = tid == (config.nr_threads - 1) ? L1.size() : start1 + L1.size(tid);  // ending index of each thread within the list L1,
		const size_t end2 = tid == (config.nr_threads - 1) ? L2.size() : start2 + L2.size(tid);  // exepct for the last thread. This needs

		uint32_t P1[p] = {0};
		uint32_t P2[p] = {0};

		// extract the bits currently set in the value
		if constexpr (config.no_values == false) {
			L1.data_value(start1).data().get_bits_set(P1, p);
			L2.data_value(start2).data().get_bits_set(P2, p);
		} else {
			get_bits_set(P1, start1, 0, cL1, cL2);
			get_bits_set(P2, start2, 1, cL1, cL2);
		}

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

	/// if the algorithm reaches this function, we found the solution. This functions only
	/// recomputes the valid error vector `e`.
	/// \param label	final label the tree reconstructed. For the normal BJMM this label is correct on the first n-k-l coordinates.
	///						the other `l` coordinates are dismissed.
	/// \param npos		array of indices of  the baselists to reconstruct the full label and value. In the normal BJMM this must be an array of length 4. For depth 3 this mus be an array of length 8.
	///						In the QC Setting with activated DOOM, this must be an array of length 3.
	/// \param weight	The weight the tree calculated on the l coordinates.
	/// \param DOOM_index2	Only used in the NN/DOOM Setting.
	///							DOOM: index of the syndrome we found a match to. Must be < k
	///							NN: Index og the hashmap found the solution.
	template<const uint32_t l_limit=config.l>
	void __attribute__ ((noinline))
	check_final_list(LabelContainerType &label,
					   IndexType npos[4],
					   const uint32_t weight,
					   const uint32_t DOOM_index2) noexcept {
		// make sure that only one thread can access this area at a given time.
		#pragma omp critical
		{
			OUTER_MULTITHREADED_WRITE(finished.store(true);)
			// make really sure that only one thread every runs this code.
			if (not_found) {
				#pragma omp atomic write
				not_found = false;
				#pragma omp flush(not_found)

				// tmp variable to recompute the solution.
				ValueContainerType value;

				if constexpr (config.no_values == false) {
					ValueContainerType::add_withoutasm
							(value, L1.data_value(npos[0]).data(),
									L2.data_value(npos[1]).data());
					ValueContainerType::add_withoutasm(value, value, L1.data_value(npos[2]).data());
					if (!config.DOOM) {
						ValueContainerType::add_withoutasm(value, value, L2.data_value(npos[3]).data());
					}
				} else {
					value.zero();
					uint32_t P[p] = {0};

					// TODO doom
					for (uint32_t i = 0; i < 4; ++i) {
						get_bits_set(P, npos[i], i%2, cL1, cL2);
						for (uint32_t j = 0; j < p; ++j) {
							value.flip_bit(P[j]);
						}
					}

					std::cout << value << "\n";
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
						LabelContainerType::add(label, label, target.data(), n-k-config.l1-(config.IM_nr_views*config.l2), n-k);
					std::cout << label << " label3\n";
				}

				uint32_t ctr1 = 0;
				uint32_t ctr2 = 0;

				// recompute the error vector by first setting the label and value at the correct
				// position and then apply the back permutation.
				for (uint32_t j = config.TrivialAppendRows; j < n - c; ++j) {
					uint32_t bit = 0;
					constexpr uint32_t limit = n - k - l_limit;
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
					for (uint32_t i = n-k-l_limit; i < n; ++i) {
						if (value[i-(n-k-l_limit)] == 1) {
							bit ^= mzd_read_bit(work_matrix_H, 0, i);
						}
					}

					bit ^= mzd_read_bit(work_matrix_H, 0, n-c);
					mzd_write_bit(e, 0, permutation->values[0], bit);
				}

#if !defined(NO_LOGGING)
				std::cout << " pre perm \n";
				std::cout << "weight n-k-l:" << ctr1 << "\n";
				std::cout << "weight tree: " << ctr2 << "\n";
				std::cout << "weight input: " << weight << "\n";
				std::cout << "hashmap/syndrome " << DOOM_index2 << " found\n";

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

				if constexpr (config.no_values == false) {
					std::cout << "\n"
					          << value << " value\n";
					std::cout << L1.data_value(npos[0]).data() << " npos[0]:" << npos[0] << "\n";
					std::cout << L2.data_value(npos[1]).data() << " npos[1]:" << npos[1] << "\n";
					std::cout << L1.data_value(npos[2]).data() << " npos[2]:" << npos[2] << "\n";
					if constexpr (!config.DOOM) {
						std::cout << L2.data_value(npos[3]).data() << " npos[3]:" << npos[3] << "\n";
					}
				}
#endif
				// Apply the back permutation of the outer loop if necessary.
				if constexpr(config.c != 0) {
					mzd_t *tmpe = mzd_init(1, n);

					for (uint32_t j = 0; j < n; ++j) {
						mzd_write_bit(tmpe, 0, c_permutation->values[j], mzd_read_bit(e, 0, j));
					}

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
					DEBUG_MACRO(std::cout << "LOWWEIGHT : errors: " << LowWeight_LVL2_Weight_Counter << "\n";)
				}

				// the only thing we print if `NO_LOGGING` is activated
				mzd_print(e);
			}
		}
	}

	/// IMPORTANT: This code is only valid if the parameter c is unequal ot 0.
	uint64_t __attribute__ ((noinline))
	BJMMF_Outer() noexcept {
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

		std::cout << "Outer Loops: " << outer_loops << "\n";
		return loops;
	}

	// runs the algorithm and return the number of iterations needed
	uint64_t __attribute__ ((noinline))
	BJMMF() noexcept {
		// we have to reset this value, so It's possible to rerun the algo more often
		not_found = true;

		// measure the intern clocks without all the preprocessing
		internal_time = clock();

		// reset loop counter, to allow multiple runs.
		loops = 0;

		// stupid test, i wanted to test this during compile time. but thats not possible
		// because the constructor is also called from MO
		if constexpr (config.HM1_EXTEND_TO_TRIPLE_SWITCH || config.HM2_EXTEND_TO_TRIPLE_SWITCH) {
			std::cout << "ett not implemented for BJMM\n";
			return 0;
		}

		// start the whole thing
		while (not_found && loops < config.loops) {
			// If we cut off some coordinates from the main matrix.
			// Exit this inner loop after the expected number of loops.
			if constexpr(c != 0) {
				if (loops >= config.c_inner_loops)
					return loops;
			}

			if constexpr (config.OptM4RI) {
				if constexpr (config.gaus_c == 0) {
					matrix_echelonize_partial_plusfix_opt<n, k, l>
					        (work_matrix_H, work_matrix_H_T, config.m4ri_k, n-k-l, this->matrix_data, permutation);
				} else {
					matrix_echelonize_partial_plusfix_opt_onlyc<n, k, l, config.gaus_c>
					        (work_matrix_H, work_matrix_H_T, config.m4ri_k, n - k - l, this->matrix_data, permutation);
				}
			} else {
				matrix_create_random_permutation(work_matrix_H, work_matrix_H_T, permutation);
				matrix_echelonize_partial_plusfix(work_matrix_H, config.m4ri_k, n - k - l, this->matrix_data, 0, n - k - l, 0, this->permutation);
			}

			ASSERT(check_matrix(work_matrix_H, n-k-l));

			// Extract the sub-matrices. the length of the matrix is only n-c + DOOM_nr but we need to copy everything.
            mzd_submatrix(H, work_matrix_H, config.TrivialAppendRows, n-k-l, n-k, n-c+DOOM_nr);
			matrix_transpose(HT, H);

			//mzd_print(work_matrix_H);

			// helper structure only needed for debugging
			Matrix_T<mzd_t *> HH((mzd_t *) H);

			// extract the syndrome as the last column of the parity check matrix.
			if constexpr(!config.DOOM) {
				target.data().column_from_m4ri(work_matrix_H, n-c-config.LOWWEIGHT, config.TrivialAppendRows);

				// also extract only the l part of the target
				iTarget_org = extractor(target);
			}

#if	NUMBER_THREADS != 1
#pragma omp parallel default(none) shared(std::cout, L1, L2, cL1, cL2, HH, H, HT, hm1, hm2, iT1, target, not_found, loops) num_threads(threads)
#endif
				{
					// Tree: From now on everything is multithreading.
					// The number at the end of a variable indicates the level its used in.
					const uint32_t tid   = threads == 1 ? 0 : omp_get_thread_num();
					const uint64_t b_tid = lsize2 / threads;    // block size of each thread. Note that we use `lsize2`,

				    // TODO todo remove das muss die liste berechnen
				    // because we only iterate over L2.
					const uint64_t s_tid = tid * b_tid;         // start position of each thread;
					const uint64_t e_tid = ((tid == threads - 1) ? L2.size() : s_tid + b_tid);
				    IndexType npos[npos_size] = {IndexType(s_tid)}; // Array of loop indices. Which happen to be also the indices that
																	// are saved in the bucket/hash map structure.
				                                                    // The idea is that we already set one position in the `indices array` of the hashmap
																	// which needs to be copied into the hashmap if a match was found.
					IndexType pos1, pos2;   // Helper variable. -1 indicating that no match was found. Otherwise the position
											// within the `hm1->__buckets[pos]` array is returned.
					LoadType load1 = 0, load2 = 0;
					ArgumentLimbType data, data1;   // tmp variable. Depending on `l` this is rather a 64bit or 128bit wide value.
				    alignas(32) LabelContainerType label, label2, label3;


					// we need to first init the baselist L1.
					// Note that `work_matrix_H_T` is already transposed in the `matrix_create_random_permutation` call.
					// Additionally, we have to keep a hash map of L1 for the whole inner loop.
					BJMM_fill_decoding_lists(L1, L2, cL1, cL2, HT, tid);
					OMP_BARRIER
					ASSERT(check_list(L1, HH, tid));
					ASSERT(check_list(L2, HH, tid));

					// initialize the buckets with -1 and the load array with zero. This needs to be done for all buckets.
					hm1->reset(tid);
					OMP_BARRIER

					// This is only the normal configuration:
					//  0                                           n-k
					// [      n-k-l n-k-l+b0    n-k-l+b0+b1          ]
					hm1->hash(L1, L1.size(tid), tid, &extractor);
					OMP_BARRIER
					hm1->sort(tid);
					OMP_BARRIER
					//ASSERT(hm1->check_sorted());

				    for (uint32_t interloops = 0; interloops < config.intermediate_target_loops; ++interloops) {
					// Pointer to the internal array of labels. Not that this pointer needs an offset depending on the thread_id.
					// Instead of access this internal array we increment the
					uint64_t *Lptr = (uint64_t *) L2.data_label() + (s_tid * llimbs_a);

					// set the initial values of the `npos` array. Somehow this is not done by OpenMP
					for (uint64_t j = 0; j < npos_size; ++j) { npos[j] = s_tid; }
					// generate a random intermediate target
					iT1 = fastrandombits<ArgumentLimbType, l>();
					iTarget = iTarget_org ^ iT1;

					hm2->reset(tid);


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
						data = extractor_ptr(Lptr);
						ASSERT((hm1->check_label(data, L2, npos[1])));
						data ^= iT1;

						if constexpr (!config.HighWeightVariant) {
							pos1 = hm1->find(data, load1);
							if constexpr (chm1.b2 == chm1.b1) {
								// if a solution exists, we know that every element in this bucket given is a solution
								while (pos1 < load1) {
									npos[0] = hm1->__buckets[pos1].second[0];
									data1 = data ^ hm1->__buckets[pos1].first;
									hm2->insert(data1, npos, tid);
									pos1 += 1;
								}
							} else {
								// Fall back to the generic case
								while (HM1BucketIndexType(pos1) != HM1BucketIndexType(-1)) {
									data1 = hm1->template traverse<0, 1>(data, pos1, npos, load1);
									hm2->insert(data1, npos, tid);
								}
							}
						} else { // high weight variant
							if constexpr (chm1.b2 == chm1.b1) {
								pos1 = hm1->traverse_find(data);
								const size_t limit1 = pos1 + config.size_bucket1;

								// for every match
								while (pos1 < limit1 && hm1->__buckets[pos1].first != HM1Type::zero_element) {
									npos[0] = hm1->__buckets[pos1].second[0];
									data1 = data ^ hm1->__buckets[pos1].first;
									hm2->insert(data1, npos, tid);
									pos1 += 1;
								}
							} else {
								assert(0 && "not implemented");
							}
						}
					}

					OMP_BARRIER
					// after everything was inserted into the second hash map, it needs to be sorted on the bits
					// [l1 + log(hm2.nr_buckets), ..., l)
					hm2->sort(tid);
					OMP_BARRIER
					//ASSERT(hm1->check_sorted());
					//ASSERT(hm2->check_sorted());
					OMP_BARRIER

					// reset tmp variables to be reusable in the second list join.
					if constexpr(config.DOOM) {
					    // in the DOOM setting we throw away the last list L4 and enumerate the
					    // list of all syndromes
						Lptr = (uint64_t *) DOOM_S_View->rows[0] + (s_tid * llimbs_a);
					} else {
						Lptr = (uint64_t *) L2.data_label() + (s_tid * llimbs_a);
					}

					uint64_t upper_limit;
					if constexpr(config.DOOM) {
						// NOTE: this is a technical limitation of the implementation in the DOOM
					    // configuration. Somehow it would be quite useless to work with multiple
					    // threads on `n` syndromes in the last list.
						ASSERT(threads == 1);
						upper_limit = n - k;
					} else {
						upper_limit = e_tid;
					}

					// do the second lst join between L3, L4.
					if constexpr (!config.HighWeightVariant) {
					    // Normal variant.
					    for (; npos[3] < upper_limit; ++npos[3], Lptr += BaseList4Inc) {
						    data = extractor_ptr(Lptr);
						    // ASSERT((hm1->check_label(data, L2, npos[3])));
						    data ^= iTarget;
						    pos1 = hm1->find(data, load1);

						    if constexpr (!config.DOOM && !config.HM1_SAVE_FULL_128BIT_SWITCH) {
							    LabelContainerType::add(label, L2.data_label(npos[3]).data(), target.data());
						    }

						    if constexpr ((chm1.b2 == chm1.b1) && (chm2.b2 == chm2.b1)) {
							    // if a solution exists, we know that every element in this bucket given is a solution
							    while (pos1 < load1) {
								    npos[2] = hm1->__buckets[pos1].second[0];
								    data1 = data ^ hm1->__buckets[pos1].first;
								    pos1 += 1;

								    pos2 = hm2->find(data1, load2);
								    if (pos2 != IndexType(-1)) {
									    if constexpr (!config.DOOM) {
										    // only precompute the label of the two right baselists if we do not have 128
										    // bits in our hashmap to check the threshold.
										    if constexpr (!config.HM1_SAVE_FULL_128BIT_SWITCH) {
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
									    DEBUG_MACRO(last_list_counter += 1;)

									    // if activated this is an early exit of the last level of the tree.
									    if constexpr (config.HM1_SAVE_FULL_128BIT_SWITCH) {
										    const ArgumentLimbType data2 = data1 ^ hm2->__buckets[pos2 - 1].first;
										    const uint32_t iweight = __builtin_popcountll(data2 >> 64u);

										    if (likely(iweight > config.weight_threshhold)) {
											    continue;
										    }

										    if (likely((__builtin_popcountll(data2) + iweight) > config.weight_threshhold)) {
											    continue;
										    }

										    LabelContainerType::add(label3, L2.data_label(npos[1]).data(), L1.data_label(npos[0]).data());
										    LabelContainerType::add(label2, L2.data_label(npos[3]).data(), L1.data_label(npos[2]).data());
										    LabelContainerType::add(label2, label2, target.data());
									    }

									    if constexpr (!config.HM1_SAVE_FULL_128BIT_SWITCH) {
										    LabelContainerType::add(label3, L2.data_label(npos[1]).data(), L1.data_label(npos[0]).data());
									    }

									    uint32_t weight;
									    if constexpr (!config.DOOM) {
#ifdef DEBUG
										    weight = LabelContainerType::template add_only_upper_weight_partly_withoutasm<lupper, lumask>(label3, label3, label2);

										    LabelContainerType ltmp;
										    LabelContainerType::add(ltmp, label3, target.data());
										    for (uint32_t i = n - config.k - l; i < n - config.k; ++i) {
											    if ((label3[i] != 0) && (ltmp[i] != 0)) {
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
										    weight = LabelContainerType::add_weight(label3.data().data(),
											                                        label3.data().data(),
											                                        label2.data().data());
#ifdef DEBUG
										    for (uint32_t i = n - config.k - l; i < n - config.k; ++i) {
												//std::cout << label3 << "\n";
											    if (label3[i] != 0) {
												    std::cout << label3 << " error label3\n";
												    ASSERT(false);
											    }
										    }
#endif
									    }

									    // DEBUG: this is stupid
									    if constexpr (config.LOWWEIGHT) {
										    if (unlikely(weight < 4)) {
											    DEBUG_MACRO(LowWeight_LVL2_Weight_Counter += 1;)
											    continue;
										    }
									    }

									    // check if we found the solution
									    if (likely(weight > config.weight_threshhold)) {
										    continue;
									    }

									    check_final_list(label3, npos, weight, npos[3]);
								    }
							    }
						    } else if constexpr (chm1.b2 == chm1.b1) {
							    while (pos1 < load1) {
								    npos[2] = hm1->__buckets[pos1].second[0];
								    data1 = data ^ hm1->__buckets[pos1].first;
								    pos1 += 1;

								    pos2 = hm2->find(data1, load2);
								    if (pos2 != IndexType(-1)) {
									    if constexpr (!config.DOOM) {
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

									    DEBUG_MACRO(last_list_counter += 1;)

									    uint32_t weight;
									    if constexpr (!config.DOOM) {
										    weight = LabelContainerType::template add_only_upper_weight_partly_withoutasm_earlyexit<lupper, lumask, config.weight_threshhold>(label3, label3, label2);
									    } else {
										    weight = LabelContainerType::add_weight(label3.data().data(), label3.data().data(), label2.data().data());
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
								    pos2 = hm2->find(data1, load2);

								    if constexpr (!config.DOOM) {
									    LabelContainerType::add(label2, label, L1.data_label(npos[2]).data());
								    } else {
									    LabelContainerType::add(label2.data().data(), L1.data_label(npos[2]).data().data().data(), Lptr);
								    }

								    while (HM2BucketIndexType(pos2) != HM2BucketIndexType(-1)) {
									    hm2->template traverse_drop<0, 2>(data1, pos2, npos, load2);
									    LabelContainerType::add(label3, L2.data_label(npos[1]).data(), L1.data_label(npos[0]).data());

									    uint32_t weight;
									    if constexpr (!config.DOOM) {
										    weight = LabelContainerType::template add_only_upper_weight_partly_withoutasm_earlyexit<lupper, lumask, config.weight_threshhold>(label3, label3, label2);
									    } else {
										    weight = LabelContainerType::add_weight(label3.data().data(), label3.data().data(), label2.data().data());
									    }

									    if (weight > config.weight_threshhold) {
										    continue;
									    }
									    check_final_list(label3, npos, weight, 0);
								    }
							    }
						    }
					    }
				    } else { // high weight variant
						for (; npos[3] < upper_limit; ++npos[3], Lptr += BaseList4Inc) {
							data = extractor_ptr(Lptr);
							ASSERT((hm1->check_label(data, L2, npos[3])));
							data ^= iTarget;

							pos1 = hm1->traverse_find(data);
							const size_t limit1 = pos1 + config.size_bucket1;

							if constexpr(!config.DOOM  && !config.HM1_SAVE_FULL_128BIT_SWITCH) {
								LabelContainerType::add(label, L2.data_label(npos[3]).data(), target.data());
							}

							if constexpr ((chm1.b2 == chm1.b1) && (chm2.b2 == chm2.b1)) {
								// if a solution exists, we know that every element in this bucket given is a solution
								while (pos1 < limit1 && hm1->__buckets[pos1].first != HM1Type::zero_element) {
									npos[2] = hm1->__buckets[pos1].second[0];
									data1 = data ^ hm1->__buckets[pos1].first;
									pos1 += 1;

									// find next element
									pos2 = hm2->traverse_find(data1);
									const size_t limit2 = pos2 + config.size_bucket2;

									// vll optimierungs potential
									if constexpr(!config.DOOM) {
										// only precompute the label of the two right baselists if we do not have 128
										// bits in our hashmap to check the threshold.
										if constexpr (!config.HM1_SAVE_FULL_128BIT_SWITCH) {
											LabelContainerType::add(label2, label, L1.data_label(npos[2]).data());
										}
									} else {
										LabelContainerType::add(label2.data().data(), L1.data_label(npos[2]).data().data().data(), Lptr);
									}

									while (pos2 < limit2 && hm2->__buckets[pos2].first != HM2Type::zero_element) {
										npos[0] = hm2->__buckets[pos2].second[0];
										npos[1] = hm2->__buckets[pos2].second[1];

										DEBUG_MACRO(last_list_counter +=1;)
										// increment the counter after the debugging output
										pos2 += 1;

										// if activated this is an early exit of the last level of the tree.
										if constexpr (config.HM1_SAVE_FULL_128BIT_SWITCH) {
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
											LabelContainerType::add(label2, label2, target.data());
										}

										if constexpr (!config.HM1_SAVE_FULL_128BIT_SWITCH) {
											LabelContainerType::add(label3, L2.data_label(npos[1]).data(), L1.data_label(npos[0]).data());
										}

										uint32_t weight;
										if constexpr(!config.DOOM) {
#ifdef DEBUG
											weight = LabelContainerType::template add_only_upper_weight_partly_withoutasm<lupper, lumask>(label3, label3, label2);

											LabelContainerType ltmp;
											LabelContainerType::add(ltmp, label3, target.data());
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

										// DEBUG: this is stupid
										if constexpr (config.LOWWEIGHT) {
											if (unlikely(weight < 4)) {
												DEBUG_MACRO(LowWeight_LVL2_Weight_Counter += 1;)
												continue;
											}
										}

										// check if we found the solution
										if (likely(weight > config.weight_threshhold)) {
											continue;
										}

										check_final_list(label3, npos, weight, npos[3]);
									}
								}
							} else {
								assert(0); // not implemented
							}
						}
					}
				    }
			    } // finish interloops
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

				// and last but least update the loop counter
				loops += 1;

		}

#if !defined(BENCHMARK) && !defined(NO_LOGGING)
		double offset_loops = (double(loops) / double(Loops())) * double(100);
		std::cout << "loops/expected: " << loops << "/" << Loops() << " " << offset_loops << "%\n" << std::flush;
#endif
		return loops;
	}

	// confy wrapper for easier abstraction of the algorithms
	inline uint64_t run() noexcept {
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
	                const uint32_t low=-1, const uint32_t high=-1, size_t size=-1) noexcept {
		typedef typename DecodingList::ValueType ValueType;
		typedef typename DecodingList::LabelType LabelType;

		LabelType tmp;

		// otherwise the whole is not working
		if (config.no_values)
			return true;

		const uint32_t k_lower = low   == uint32_t(-1) ? 0 : low;
		const uint32_t k_higher = high == uint32_t(-1) ? LabelType::size() : high;
		for (auto &LL: L) {
			const size_t listsize = size == size_t(-1) ? LL.size() : size;
			for (size_t i = 0; i < listsize; ++i) {
				new_vector_matrix_product<LabelType, ValueType, mzd_t *>(tmp, LL.data_value(i), H);
				if (tmp.data().is_zero()) {
					continue;
				}

				if (!tmp.is_equal(LL.data_label(i), k_lower, k_higher)) {
					std::cout << tmp << "  should vs it at pos:" << i << ", tid:" << tid << "\n";
					std::cout << LL.data_label(i) << "\n";
					std::cout << LL.data_value(i) << "\n" << std::flush;
					return false;
				}

				if (tmp.data().is_zero() || LL.data_value(i).data().is_zero() || LL.data_label(i).data().is_zero()) {
					std::cout << tmp << " any element is zero:" << i << "\n";
					std::cout << LL.data_label(i) << " label in table\n";
					std::cout << LL.data_value(i) << " value in table\n" << std::flush;
					return false;
				}
			}
		}

		return true;
	}


	template<class DecodingList>
	bool check_list(const DecodingList &L1, const Matrix_T<mzd_t *> &H, const uint32_t tid,
					const uint32_t low=-1, const uint32_t high=-1, size_t size=-1) noexcept {
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

	// Expected size of the intermediate list.
	constexpr static std::array<uint64_t, 3> ListSizes() noexcept {
		uint64_t S1 = (lsize2 * lsize2) >> l1;
		uint64_t S2 = (S1 * S1) >> (l-l1);

		std::array<uint64_t, 3> ret{lsize2, S1, S2};
		return ret;
	}

	/// \param nn total number of coordinates
	/// \param cc number of coordinates to cut off e.g. to guess zero
	/// \return the Outer loops if c != 0
	constexpr static double LogOuterLoops(const uint64_t nn, const uint64_t cc) noexcept {
		if (cc == 0) return 1.0;

		// log(binomial(n, w)/binomial(n-c, w), 2).n()
		const double nn_ = double(nn);
		const double w_ = double(w) / double(nn);
		const double cc_ = double(cc) / nn_;

		const double shift = HH(w_) - (1 - cc_) * HH(w_);
		return shift;
	}

	/// \param nn total number of coordinates
	/// \param cc number of coordinates to cut off
	/// \return same as `LogOuterLoops` only in the exponential form/
	constexpr static double OuterLoops(const uint64_t nn, const uint64_t cc) noexcept {
		if (cc == 0) return 1.0;

		return pow(2, LogOuterLoops(nn, cc));
	}

	/// \param nn code length
	/// \param cc dimensions to cut off.
	/// \return the expected numbers of loops needed in log
	constexpr static double LogLoops(const uint64_t nn, const uint64_t cc=0) noexcept {
		// note that p is here only the base p.
		// binom(n, w)/(binom(n-k-l, w-4p) * binom((k+l)/2 - cc/2, 2*p)**2)
		// log((binomial(n, w) / (binomial(n-k-l, w-4*p) * binomial((k+l)/2, 2*p)**2)), 2).n()
		// log((binomial(n-c, w) / (binomial(n-k-l, w-4*p) * binomial((k+l)/2 - c//2, 2*p)**2)) * binomial(n, w)/binomial(n-c, w), 2).n()
		const double nn_ = double(nn);
		const double k_  = double(k) / nn_;
		const double l_  = double(l) / nn_;
		const double p_  = (double(2*p)) / nn_;
		const double wp_ = double(w - (4 * p)) / nn_;
		const double w_  = double(w) / nn_;

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

	/// same `LogLoops` only returns the total number of loops not in log form
	constexpr static double Loops(const uint64_t nn=n, const uint64_t cc=0) noexcept {
		return pow(2, LogLoops(nn, cc));
	}

	void __attribute__ ((noinline))
	info() noexcept {
#if !defined(BENCHMARK) && !defined(NO_LOGGING)
			std::cout << "BJMMF: tid:" << omp_get_thread_num() << ", loops: " << loops << "\n";
			std::cout << "log(inner_loops): " << LogLoops(n, c) << ", inner_loops: " << Loops(n, c) ;
			if constexpr(c != 0) {
				std::cout << ", log(outer_loops): " << LogOuterLoops(n, c) << ", outer_loops: " << OuterLoops(n, c) ;
			}
			std::cout << "\n|Value|: " << DecodingValue::size() << ", |Label|:" << DecodingLabel::size() << "\n";
		    std::cout << "|L_1|: " << this->L1.size() << "\n";
			std::cout << "|L_1|: " << this->L1.bytes() / (1<<20) << "MB\n";
			std::cout << "|L_1|+|L_2|: " << (this->L1.bytes() + this->L2.bytes()) / (1<<20) << "MB\n";
			double ctime = ((double)clock()-internal_time)/CLOCKS_PER_SEC;
			std::cout << "Time: " << ctime << ", clock Time: " << ctime/config.nr_threads << "\n";
			hm1->print();
			hm2->print();
		    auto LSizes = ListSizes();
			std::cout << "Exp |hm1|=" << lsize1 << ", Exp |hm2|=" << LSizes[1] << ", Exp |out|=" << LSizes[2]
#ifdef DEBUG
            << ":" << last_list_counter/MAX(1, loops) << "\n\n" << std::flush;
#else
			<< "\n\n" << std::flush;
#endif
#endif
	}

	/// prints current loops information like: Hashmap usage, time, loops, ...
	/// This function is intentionally not inlined to reduce the pressure on the instruction cache.
	void __attribute__ ((noinline))
	periodic_info() noexcept {
#if !defined(NO_LOGGING)
		double ctime = (((double)clock())-internal_time)/CLOCKS_PER_SEC;
		std::cout << "\rcurrently at " << loops << " loops, " << ctime << "s, " << loops/ctime << "lps\n" << std::flush;
#endif
	}
};

#endif //SMALLSECRETLWE_BJMM_H