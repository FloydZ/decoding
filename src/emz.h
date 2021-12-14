#ifndef SMALLSECRETLWE_EMZ_H
#define SMALLSECRETLWE_EMZ_H

#include "sort.h"
#include "decoding.h"
#include "list_fill.h"

using DiffList = std::vector<std::vector<std::pair<uint64_t, uint64_t>>>;


///  Schematic View On the Algorithm.
///			   <-----c----->
///			   +---------------+----------------+-------------------------------+  +--------^
///			   |            |  |                |                               |  |       ||
///			   |            |  |                |                               |  |       ||
///			   |            |  |                |                               |  |       ||n-k-l
///			   |   I_{n-k-l}|  |       0        |              H_1              |  |  s_1  ||
///			   |            |  |                |                               |  |       ||
///			   |            |  |                |                               |  |       |v
///			   +----------------------------------------------------------------+  +-------->  ^
///			   |            |  |                |                               |  |       ||  |
///			   |            |  |    I_{l_1}     |                               |  |       ||  |l_1
///			   |            |  |                |                               |  |       ||  +
///			   |       0    |  +----------------+              H_2              |  |  s_2  ||l
///			   |            |  |                |                               |  |       ||  +
///			   |            |  |                |                               |  |       ||  |l_2
///			   |            |  |                |                               |  |       ||  |
///			   +---------------+----------------+-------------------------------+  +--------v  v
///
///			                   <-----------+l+---------->
///			                   <-----------------|  l+k  +---------------------->
///			   +----------------------------------------------------------------+
///			   |              || <---+l_1 +---> |   l_2  | <-------+k+--------> |
///			   |              ||     4*w_1      |    0   |        4*w_2         |
///			   |     e_2      ||      e_d       |        |         e_1          |
///			   +--------------------------------+-----+--+----------------------+
///			   <-------------->                       ^   <-------+   +--------->
///			        n-k-l                             |
///			                                          |
///			                    +---------------------+---------------------+
///			                    |             s_2 on l coord                |
///			             l1/2   |   k/2                                     |
///			           +--------+--------+                         +--------+--------+
///			           |w_1|w_1|0|w_2|w_2|                         |w_1|w_1|0|w_2|w_2|
///			           |   |   | |   |   |                         |   |   | |   |   |
///			           +---+---+++---+---+                         +---+---+^+---+---+
///			           0        |        k                                  |
///			         +----------+----------+                     +----------+----------+
///			         |       T on l_1      |                     |    +T+s_2 on l_1    |
///			  l_1/2  |   k/2        l1/2   |   k-k/2      l_1/2  |   k/2        l_1/2  |   k/2
///			+--------+--------+   +--------+--------+   +--------+--------+   +--------+--------+
///			|w_1| 0 |0|w_2| 0 |   | 0 |w_1|0| 0 |w_2|   |w_1| 0 |0|w_2| 0 |   | 0 |w_1|0| 0 |w_2|
///			|   |   | |   |   |   |   |   | |   |   |   |   |   | |   |   |   |   |   | |   |   |
///			+---+-----+---+---+   +---+---+-----+---+   +---+---+-+---+---+   +---+---+-+---+---+
///			0       ^ l_2     k   0      l_1^       k   0      l_1        k   0        l_2      k
///			        |                       |
///			       l_1                     l_2
/// \tparam n 		Challenge Dimension
/// \tparam c 		cutoff c dimensions
/// \tparam k 		Challenge Dimension
/// \tparam l 		Dumer Window
/// \tparam w 		Challenge Parameter
/// \tparam p 		Challenge Paramet
/// \tparam d 		Tree Depth
/// \tparam l1		Length of e_d
/// \tparam w1		weight on e_d (on l1/2 coordinates) of the baselists
/// \tparam l2		length of (hopefully) zero window
/// \tparam w2		weight on e_1 (on k/2 coordinates)	of the baselists
/// \tparam epsilon			number of bits we allow each side of the k part to overlap
/// \tparam number_buckets	number of buckets used in the bucket sort algorithm
/// \tparam tries			number of tree rebuild with differetn intermediate targets
/// \param e		output: solution error
/// \param s		const input:	goppa code syndrom.
/// \param A		const input: McEliece Matrix
/// \param 			exit_loops terminate the programm after `exit_loops` runs of the outer loops
/// \return	1 on success. 0 on not found.
struct ConfigEMZ2DThread {
public:
	const uint32_t n, k, w, p, l, l1, l2, w1, w2, epsilon, r1, tries, number_bucketsearch_retries;
	uint32_t c = 0;         // number of coordinates to cut off from the left.
	uint32_t nrb1 = l1;     // Number of buckets for the first hash map. In log scale
	uint32_t nrb2 = l2-l1;  // Number rof buckets for the second hash map in log scale.

	// TODO implement
	uint32_t sb1 = 1;       // Size of each Bucket in the first hashmap = number of elements each bucket can hold at max.
	uint32_t sb2 = 1;       // Same as `sb1`
	uint32_t threads = 1;

	uint64_t exit_loops = -1;

	constexpr ConfigEMZ2DThread(uint32_t n, uint32_t k, uint32_t w, uint32_t p, uint32_t l, uint32_t l1, uint32_t l2,
	                            uint32_t w1, uint32_t w2, uint32_t epsilon, uint32_t r1,
	                            uint32_t tries, uint32_t number_bucketsearch_retries) :
			n(n), k(k), w(w), p(p), l(l), l1(l1), l2(l2), w1(w1), w2(w2), epsilon(epsilon), r1(r1), tries(tries),
			number_bucketsearch_retries(number_bucketsearch_retries)
	{}
};

// TODO implement with new hashmap implementation.
template<const ConfigEMZ2DThread &config, class List>
class EMZ {
private:
	typedef typename DecodingList::LabelType DecodingLabel;
	typedef typename DecodingList::ValueType DecodingValue;
	typedef typename DecodingList::ElementType DecodingElement;
	typedef typename DecodingList::MatrixType DecodingMatrix;

	// TODO auf größe von l anpassen
	using ArgumentLimbType  = uint64_t;
	using IndexType         = uint64_t;
	using LoadType          = uint64_t;

	// recalculated lengths if we cut of `c` coordinates
	constexpr static uint64_t n = config.n;
	constexpr static uint64_t k = config.k;
	constexpr static uint64_t n_prime = n-config.c;
	constexpr static uint64_t k_prime = k-config.c;
	constexpr static uint64_t l = config.l;
	constexpr static uint64_t l1 = config.l1;
	constexpr static uint64_t l2 = config.l2;
	constexpr static uint64_t r1 = config.r1;
	constexpr static uint64_t w1 = config.w1;
	constexpr static uint64_t w2 = config.w2;
	constexpr static uint64_t threads = config.threads;

	// TODO größen auf neue hashmap anpassen.
	// expected size of baselists, intermediate lists and the output list
	constexpr static float    size_scale = 1.11;
	constexpr static float    size_scale_inter = 1.07;
	constexpr static uint64_t size_bucket_scale = 1;
	constexpr static uint64_t size_baselist_ = bc(l1/2, w1)*bc((k_prime/2)+config.epsilon,w2);
	constexpr static uint64_t size_intermediate_ = (size_baselist_*size_baselist_) / (1<<(l1+r1));
	constexpr static uint64_t size_out_ = (size_intermediate_*size_intermediate_) / (1<<l2);

	// Do not scale the baselists size, because we know exactly how big they are.
	constexpr static uint64_t size_baselist = size_baselist_*size_scale;
	constexpr static uint64_t size_intermediate = size_intermediate_*size_scale_inter;
	constexpr static uint64_t size_out = size_out_*size_scale;

	// number of rebuilds of the level2 bucket. During each rebuild of the output list of the tree the `window` of
	// guessed zero coordinates is moved to the right by `bucketsearch_step` coordinates. So the following images accrues.
	//         n-k-l      l1 l2
	// [        |       |000|   |   .....   ]   The
	// `window`:[  ]
	// next iter:   [  ]            <- additional tries.
	// next iter:           [  ]    <- normal guessed zero window.
	// In someway this is a Indyk-Motwani NN approach, but not all random selections of coordinates are possible.
	constexpr static uint64_t number_bucketsearch_retries = MIN(config.number_bucketsearch_retries, n-k-l-l2);

	// Limits
	constexpr static uint64_t k_lower1 = n-k-l-r1;
	constexpr static uint64_t k_upper1 = n-k-l+l1;
	constexpr static uint64_t k_lower2 = n-k-l+l1;
	constexpr static uint64_t k_upper2 = n-k;

	constexpr static ConfigParallelBucketSort chm1{0, 0 + config.nrb1, 0 + l1, config.sb1,
	                                               uint64_t(1) << config.nrb1, threads, 1,
	                                               n_prime - k_prime - l, l, 0};
	constexpr static ConfigParallelBucketSort chm2{0+l1, l1 + config.nrb2, 0 + l2, config.sb2,
	                                               uint64_t(1) << config.nrb2, threads, 2,
	                                               n_prime - k_prime - l, l, 0};

	static ArgumentLimbType Hash(ArgumentLimbType) {
		return 0;
	}
	ParallelBucketSort<chm1, DecodingList, ArgumentLimbType, IndexType, &Hash> *hm1;
	ParallelBucketSort<chm2, DecodingList, ArgumentLimbType, IndexType, &Hash> *hm2;

	// Changelistst
	std::vector<std::vector<std::pair<uint64_t, uint64_t>>> changelist11, changelist12, changelist21, changelist22;
	// The baselists are shared between threads
	DecodingList BaseList1{size_baselist},  BaseList2{size_baselist};
	// These lists are not shared between different threads
	DecodingList iL1{size_intermediate}, iL2{size_intermediate}, out{size_out}, List1{size_baselist}, List2{size_baselist};

	mzd_t *e;
	const mzd_t *s;
	const mzd_t *A;

public:
	EMZ(mzd_t *e, const mzd_t *const s, const mzd_t *const A)
			: e(e), s(s), A(A){
		static_assert(size_baselist > 0);
		static_assert(size_intermediate > 0);
		static_assert(size_out > 0);

		prepare_generate_base_mitm2_extended<DecodingList>(BaseList1, changelist11, changelist12, l1, k_prime, l2, w1, w2, false, config.epsilon, true);
		prepare_generate_base_mitm2_extended<DecodingList>(BaseList2, changelist21, changelist22, l1, k_prime, l2, w1, w2, true, config.epsilon, true);

		hm1 = new ParallelBucketSort<chm1, DecodingList, ArgumentLimbType, IndexType, &Hash>();
		hm2 = new ParallelBucketSort<chm2, DecodingList, ArgumentLimbType, IndexType, &Hash>();
	}

	~EMZ() {
		delete (hm1);
		delete (hm2);
	}

	int emz_new_data_structure_d2_thread() {

		// every thread needs its own copy of the working matrix
		mzd_t *work_matrix_H = mzd_init(n - k, n_prime + 1);
		mzd_t *work_matrix_H_T= mzd_init(work_matrix_H->ncols,work_matrix_H->nrows);
		mzd_t *columnTransposed = mzd_transpose(NULL, s);
		mzd_concat(work_matrix_H, A, columnTransposed);

		// helper functions to better access H_1, H_2
		mzd_t *H_prime = mzd_init(n - k, k_prime + l);
		mzd_t *H_prime_T = mzd_init(k_prime + l, n - k);

		mzd_t *working_s = mzd_init(n - k, 1);
		mzd_t *working_s_T = mzd_init(1, n - k);

		// for the matrix permutations
		mzp_t *permutation = mzp_init(n_prime);

		// return value of `check_resultlist`. __MUST__ not be freed anywhere.
		mzd_t *t;
		constexpr static ConfigCheckResultList configCheckResultList(n, k, w, config.c, l, l1);

		uint64_t outer_loops = 0;   // Performance counter.
		int ra = 1;                 // Rang counter.

		const uint64_t m4ri_k = m4ri_opt_k(work_matrix_H->nrows, work_matrix_H->ncols, 0);

		// TODO omp implementieren.
		const uint32_t tid = 0;// omp_get_thread_num();

		// precompute grey code for m4ri
		customMatrixData *matrix_data = init_matrix_data(work_matrix_H->ncols);
		while (ra != 0) {
			matrix_create_random_permutation(work_matrix_H, work_matrix_H_T, permutation);
			ra = matrix_echelonize_partial_plusfix(work_matrix_H, m4ri_k, work_matrix_H->nrows, matrix_data, 0,
			                                       n - k - l, 0, permutation);

			mzd_submatrix(working_s, work_matrix_H, 0, n_prime, n - k, n_prime + 1);
			mzd_transpose(working_s_T, working_s);

			mzd_submatrix(H_prime, work_matrix_H, 0, n - k - l, n - k, n_prime);
			mzd_transpose(H_prime_T, H_prime);

			Matrix_T<mzd_t *> B(H_prime);

			mceliece_d2_fill_decoding_lists<n, k_prime, l, l1, w1, l2, w2, config.epsilon, DecodingList>(List1, List2, changelist11, changelist12, changelist21, changelist22, H_prime_T);
			ASSERT(check_correctness<DecodingList>(List1, B));
			ASSERT(check_correctness<DecodingList>(List2, B));

			DecodingLabel target, zero, iR;;
			target.data().from_m4ri(working_s_T);
			zero.zero();

			const uint32_t tid = omp_get_thread_num();

			for (int try_i = 0; try_i < config.tries; ++try_i) {
				iR.random();    // generate a random intermediate target.

				// TODO now fill the tree.

				out.set_load(0);
				iL1.set_load(0);
				iL2.set_load(0);
			}
		}

		finish:
		std::cout <<  MULTITHREADED_WRITE("Thread: " << std::this_thread::get_id() << " " << ) "number of outer loops: " << outer_loops << "\n";



		mzd_free(work_matrix_H);
		mzd_free(work_matrix_H_T);
		mzd_free(working_s);
		mzd_free(working_s_T);
		mzd_free(H_prime);
		mzd_free(H_prime_T);
		mzd_free(columnTransposed);
		mzp_free(permutation);

		free_matrix_data(matrix_data);

		//return return_value;
		return outer_loops;
	}

};

template<const uint64_t n, const uint64_t c, const uint64_t k, const uint64_t l, const uint64_t w, const uint64_t p, const uint64_t d,
		const uint64_t l1, const uint64_t w1, const uint64_t l2, const uint64_t w2, const uint64_t epsilon, const uint64_t r1=0, const uint64_t number_buckets=13, const uint64_t tries=1, const uint64_t number_bucketsearch_retries_=1, const uint64_t exit_loops=99999, class DecodingList=DecodingList>
int emz_d2_thread(mzd_t *e, const mzd_t *const s, const mzd_t *const A, const DecodingList &BaseList1, const DecodingList &BaseList2, const DiffList &changelist11, const DiffList &changelist12, const DiffList &changelist21, const DiffList &changelist22
#ifdef CHECK_PERM
	, const mzd_t *correct_e
#endif
) {
	typedef typename DecodingList::LabelType DecodingLabel;
	typedef typename DecodingList::ValueType DecodingValue;
	typedef typename DecodingList::ElementType DecodingElement;
	typedef typename DecodingList::MatrixType DecodingMatrix;

#ifdef CHECK_PERM
	uint64_t good_perms=0;
    mzd_t * e_tmp= mzd_init(1,n);
    uint16_t pre_set[6]={0,0,0,0,4*w2-2};
    Value_T<BinaryContainer<n>> cor_e;
#endif
	
	// recalculated lengths if we cut of `c` coordinates
	constexpr uint64_t n_prime = n-c;
	constexpr uint64_t k_prime = k-c;

	// expected size of baselists, intermediate lists and the output list
	constexpr float    size_scale = 1.11;                                       // TODO ACHTUNG ANDRE
	constexpr float    size_scale_inter = 1.07;                                 // TODO ACHTUNG ANDRE

	constexpr uint64_t size_bucket_scale = 1;//2;
	constexpr uint64_t size_baselist_ = bc(l1/2, w1)*bc((k_prime/2)+epsilon,w2);
	constexpr uint64_t size_intermediate_ = (size_baselist_*size_baselist_) / (1<<(l1+r1));
	constexpr uint64_t size_out_ = (size_intermediate_*size_intermediate_) / (1<<l2);
	constexpr uint64_t size_bucket = MAX(10, MAX(size_baselist_/MIN(1<<l1, 1<<number_buckets), size_intermediate_/MIN(1<<l2, 1<<number_buckets)))*size_bucket_scale;

	// Do not scale the baselists size, because we know exactly how big they are.
	constexpr uint64_t size_baselist = size_baselist_*size_scale;       // TODO bug fix für die Tatsache das unsere BaselistGröße nicht korrekt ist, da ja eigenltich \prod über alle \omega berechnet werden müssten.
	constexpr uint64_t size_intermediate = size_intermediate_*size_scale_inter;
	constexpr uint64_t size_out = size_out_*size_scale;
	static_assert(size_baselist > 0, "this is not good");
	static_assert(size_intermediate > 0, "this is again not good");
	static_assert(size_out > 0, "again not good");

	// number of rebuilds of the level2 bucket. During each rebuild of the output list of the tree the `window` of
	// guessed zero coordinates is moved to the right by `bucketsearch_step` coordinates. So the following images accrues.
	//         n-k-l      l1 l2
	// [        |       |000|   |   .....   ]   The
	// `window`:[  ]
	// next iter:   [  ]            <- additonal tries.
	// next iter:           [  ]    <- normal guessed zero window.
	// In someway this is a Indyk-Motwani NN approach, but not all ramdom selections of coordinates are possible.
	constexpr uint64_t number_bucketsearch_retries = MIN(number_bucketsearch_retries_, n-k-l-l2);
	constexpr uint64_t bucketsearch_step = number_bucketsearch_retries == 0 ? n-k-l-l2 : MAX(1, (n-k-l-l2)/MAX(1, number_bucketsearch_retries));

	// Limits
	constexpr uint64_t k_lower1 = n-k-l-r1;
	constexpr uint64_t k_upper1 = n-k-l+l1;
	constexpr uint64_t k_lower2 = n-k-l+l1;
	constexpr uint64_t k_upper2 = n-k;


	// These lists are not shared between different threads
	DecodingList iL1{size_intermediate}, iL2{size_intermediate}, out{size_out}, List1{size_baselist}, List2{size_baselist};

	// Bucket sort for Level 1 and 2.
	auto *bucket_level1 = new Bucket_Sort<DecodingList, size_bucket, k_lower1,
			k_lower1 + MIN(l1+r1, number_buckets), k_upper1>();
	auto *bucket_level2 = new Bucket_Sort<DecodingList, size_bucket, k_lower2,
			k_lower2 + MIN(l2, number_buckets), k_upper2>();

	// every thread needs its own copy of the workingmatrix
	mzd_t *work_matrix_H = mzd_init(n - k, n_prime + 1);
	mzd_t *work_matrix_H_T= mzd_init(work_matrix_H->ncols,work_matrix_H->nrows);
	mzd_t *columnTransposed = mzd_transpose(NULL, s);
	mzd_concat(work_matrix_H, A, columnTransposed);

	// helper functions to better access H_1, H_2
	mzd_t *H_prime = mzd_init(n - k, k_prime + l);
	mzd_t *H_prime_T = mzd_init(k_prime + l, n - k);

	mzd_t *working_s = mzd_init(n - k, 1);
	mzd_t *working_s_T = mzd_init(1, n - k);

	// for the matrix permutations
	mzp_t *P_C = mzp_init(n_prime);

	// return value of `check_resultlist`. __MUST__ not be freed anywhere.
	mzd_t *t;
	constexpr static ConfigCheckResultList configCheckResultList(n, k, w, c, l, l1);

	uint64_t outer_loops = 0;
	int ra = 1;

	// TODO check if optimal?
	const uint64_t m4ri_k = m4ri_opt_k(work_matrix_H->nrows, work_matrix_H->ncols, 0);

	// precompute grey code for m4ri
	customMatrixData *matrix_data = init_matrix_data(work_matrix_H->ncols);
	while (ra != 0) {
		matrix_create_random_permutation(work_matrix_H, work_matrix_H_T, P_C);
		ra = matrix_echelonize_partial(work_matrix_H, m4ri_k, work_matrix_H->nrows, matrix_data, 0);
		if (ra < n - k - l) {
			continue;
		}

#ifdef CHECK_PERM
		//mzd_copy(e_tmp,correct_e);
	    //mzd_apply_p_right(e_tmp,P_C);

        for(rci_t j=0; j < n; ++j)
            mzd_write_bit(e_tmp, 0, j, mzd_read_bit(correct_e, 0, P_C->values[j]));
        cor_e.data().from_m4ri(e_tmp);
        uint16_t l1_1_weight = cor_e.data().weight(n-k-l,n-k-l+l1/2);
        uint16_t l1_2_weight = cor_e.data().weight(n-k-l+l1/2,n-k-l+l1);
        uint16_t l2_1_weight = cor_e.data().weight(n-k-l+l1,n-k-l+l1+l2/2);
        uint16_t l2_2_weight = cor_e.data().weight(n-k-l+l1+l2/2,n-k);
        uint16_t k1_weight = cor_e.data().weight(n-k,n-k+k/2);
        uint16_t k2_weight = cor_e.data().weight(n-k+k/2,n);


        if(l1_1_weight==pre_set[0]&& l1_2_weight==pre_set[1]&& l2_1_weight==pre_set[2]
        && l2_2_weight==pre_set[3]&& k1_weight<=pre_set[4]/2 && k2_weight<=pre_set[4]/2) {
            good_perms++;
            std::cout << "Good Ones " << good_perms << ", weight "<<k1_weight+k2_weight<<"\n";
        }
#endif

		copy_submatrix(working_s, work_matrix_H, 0, n_prime, n - k, n_prime + 1);
		mzd_transpose(working_s_T, working_s);

		copy_submatrix(H_prime, work_matrix_H, 0, n - k - l, n - k, n_prime);
		mzd_transpose(H_prime_T, H_prime);

		Matrix_T<mzd_t *> B(H_prime);

		List1 = BaseList1;
		List2 = BaseList2;

		// MADIVE((void *)List1.data(), List1.get_load() * DecodingElement::bytes(), POSIX_MADV_WILLNEED|MADV_SEQUENTIAL);
		// MADIVE((void *)List2.data(), List2.get_load() * DecodingElement::bytes(), POSIX_MADV_WILLNEED|MADV_SEQUENTIAL);

		// MADIVE((void *)changelist11.data(), changelist11.size() * sizeof(std::pair<uint64_t, uint64_t>), POSIX_MADV_WILLNEED|MADV_SEQUENTIAL);
		// MADIVE((void *)changelist12.data(), changelist12.size() * sizeof(std::pair<uint64_t, uint64_t>), POSIX_MADV_WILLNEED|MADV_SEQUENTIAL);
		// MADIVE((void *)changelist21.data(), changelist21.size() * sizeof(std::pair<uint64_t, uint64_t>), POSIX_MADV_WILLNEED|MADV_SEQUENTIAL);
		// MADIVE((void *)changelist22.data(), changelist22.size() * sizeof(std::pair<uint64_t, uint64_t>), POSIX_MADV_WILLNEED|MADV_SEQUENTIAL);

		mceliece_d2_fill_decoding_lists<n, k_prime, l, l1, w1, l2, w2, epsilon, DecodingList>(List1, List2, changelist11, changelist12, changelist21, changelist22, H_prime_T);
		ASSERT(check_correctness<DecodingList>(List1, B));
		ASSERT(check_correctness<DecodingList>(List2, B));

		DecodingLabel target;
		target.data().from_m4ri(working_s_T);

		// hash the table.
		uint64_t bucket, lower, upper;
		bool found;
		bucket_level1->hash(List1);

		DecodingLabel zero, iR;
		zero.zero();

		// checks if one of the lists is already full. If so quit.
		int full = 1;

		for (int try_i = 0; try_i < tries; ++try_i) {
			full = 1;
			iR.random();

			//MADIVE((void *)iL1.data(), iL1.get_size() * DecodingElement::bytes(), POSIX_MADV_WILLNEED | POSIX_MADV_SEQUENTIAL);
			// MADIVE((void *)iL2.data(), iL2.get_size() * DecodingElement::bytes(), POSIX_MADV_WILLNEED | POSIX_MADV_SEQUENTIAL)

			for (uint64_t i = 0; i < List2.get_load() && full > 0; ++i) {
				// first add the target
				DecodingLabel::add(List2[i].get_label(), List2[i].get_label(), iR, k_lower1, k_upper1);

				// search for collisions
				found = bucket_level1->find(List2[i].get_label(), &bucket, &lower, &upper);
				if (found) {
					for (uint64_t j = lower; j <= upper; ++j) {
						full = iL1.template add_and_append<size_intermediate>(List1[bucket_level1->buckets[bucket][j].first], List2[i]);
						//full = iL1.add_and_append2((uint64_t *)&List1[bucket_level1->buckets[bucket][j].first], (uint64_t *)&List2[i], size_intermediate);
					}
				}

				// add on the full lengt
				DecodingLabel::sub(List2[i].get_label(), List2[i].get_label(), target, 0, k_upper2);

				// search for collisions
				found = bucket_level1->find(List2[i].get_label(), &bucket, &lower, &upper);
				if (found) {
					for (uint64_t j = lower; j <= upper; ++j) {
						full = iL2.template add_and_append<size_intermediate>(List1[bucket_level1->buckets[bucket][j].first], List2[i]);
						//full = iL2.add_and_append2((uint64_t *)&List1[bucket_level1->buckets[bucket][j].first], (uint64_t *)&List2[i], size_intermediate);
					}
				}
			}

			bucket_level2->hash(iL1);
			MADIVE((void *)out.data(), out.get_size() * DecodingElement::bytes(), POSIX_MADV_WILLNEED|POSIX_MADV_SEQUENTIAL);
			full = 1;

			for (uint64_t i = 0; i < iL2.get_load() && full > 0; i++) {
				found = bucket_level2->find(iL2[i].get_label(), &bucket, &lower, &upper);
				if (found) {
					for (uint64_t j = lower; j <= upper; ++j) {
						full = out.template add_and_append<size_out>(iL1[bucket_level2->buckets[bucket][j].first], iL2[i]);
						//full = out.add_and_append2((uint64_t *)&iL1[bucket_level2->buckets[bucket][j].first], (uint64_t *)&iL2[i], size_out);
					}
				}
			}

			t = check_resultlist<configCheckResultList, DecodingList>(out, w, s, P_C, A);
			if ((t != nullptr) MULTITHREADED_WRITE(&& !finished.load())) {
				MULTITHREADED_WRITE(finished.store(true));
				mzd_copy(e, t);
				mzd_free(t);
				std::cout << try_i << " found in initial try" << "\n";
				goto finish;
			}

			for (int v = 0; v < number_bucketsearch_retries; v++) {
				// Reset the out list everytime we try a different l2 zero window.
				out.set_load(0);
				full = 1;

				bucket_level2->hash(iL1, v * bucketsearch_step, l2 + (v * bucketsearch_step));
				for (uint64_t i = 0; i < iL2.get_load() && full > 0; i++) {
					found = bucket_level2->find(iL2[i].get_label(), &bucket, &lower, &upper, v * bucketsearch_step,
					                            l2 + (v * bucketsearch_step));
					if (found) {
						for (uint64_t j = lower; j <= upper; ++j) {
							full = out.template add_and_append<size_out>(iL1[bucket_level2->buckets[bucket][j].first], iL2[i]);
							//full = out.add_and_append2((uint64_t *)&iL1[bucket_level2->buckets[bucket][j].first], (uint64_t *)&iL2[i], size_out);
						}
					}
				}

				t = check_resultlist<configCheckResultList, DecodingList>(out, w, s, P_C, A);
				if ((t != nullptr) MULTITHREADED_WRITE(&& !finished.load())) {
					MULTITHREADED_WRITE(finished.store(true));
					mzd_copy(e, t);
					mzd_free(t);
					std::cout << try_i << " found in retry number" << v + 1 << "\n";
					goto finish;
				}
			}

#ifdef DEBUG
			if (outer_loops == 0 && try_i == 0) {
				const uint64_t mem_buckets      = (sizeof(*bucket_level1)+sizeof(*bucket_level2))/(1024*1024);
				const uint64_t mem_outlist      = NUMBER_THREADS*out.size()*sizeof(DecodingElement)/(1024*1024);
				const uint64_t mem_interlists   = NUMBER_THREADS*(iL2.size()+iL1.size())*sizeof(DecodingElement)/(1024*1024);
				const uint64_t mem_baselists    = NUMBER_THREADS*4*List1.size()*sizeof(DecodingElement)/ (1024*1024);
				const uint64_t mem_changelists  = 4*changelist22.size() * sizeof(std::pair<uint64_t, uint64_t>);    // thats only a upper bound.
				const uint64_t mem_baselists_   = 2*List1.size()*sizeof(DecodingElement)/ (1024*1024);          // the two lists getting shifted in

				const uint64_t outer_diff =  out.size() - out.get_load();
				const uint64_t inter_diff1 =  iL1.size() - iL1.get_load();
				const uint64_t inter_diff2 =  iL2.size() - iL2.get_load();

				bucket_level1->print_stats();
				bucket_level2->print_stats();

				std::cout << "OutDiff:            " << outer_diff << " log:" << log2(outer_diff) << "\n";
				std::cout << "InterDiff:          " << inter_diff1 << " log:" << log2(inter_diff1) << " " << inter_diff2 << " log:" << log2(inter_diff2) << "\n";

#if (defined(NUMBER_THREADS) && NUMBER_THREADS > 1)
				std::cout << "TotalMemChangeLists: " << changelist22.size() << " mem: " << mem_changelists << "MB\n";
				std::cout << "TotalMemBuckets:     " << size_bucket << " log: " << log2(size_bucket) << " mem: " << mem_buckets << "MB\n";
				std::cout << "TotalSizeOutlists:   " << out.get_load() << " log: " << log2(out.get_load()) << " mem: " << mem_outlist << "MB\n";
				std::cout << "TotalSizeInterlists: " << iL1.get_load() << " log: " << log2(iL1.get_load()) << " mem: " << mem_interlists << "MB\n";
				std::cout << "TotalSizeBaselists:  " << List1.get_load() << " log: " << log2(List1.get_load()) << " mem: " << mem_baselists + mem_baselists_ << "MB\n";
				std::cout << "TotalMem:            " << (mem_buckets+mem_outlist+mem_interlists+mem_baselists+mem_baselists_+mem_changelists)  << "MB\n";
#else
				std::cout << "TotalMemChangeLists: " << mem_changelists << "MB\n";
				std::cout << "NumberBuckets:       " << number_buckets << "\n";
				std::cout << "SizeBucket:          " << size_bucket << " log: " << log2(size_bucket) << "\n";
				std::cout << "MemBuckets:          " << mem_buckets << "MB\n";

				std::cout << "SizeOutlists:        " << out.get_load() << " log: " << log2(out.get_load()) << " mem: " << mem_outlist << "MB\n";
				std::cout << "SizeInterlists:      " << iL1.get_load() << " log: " << log2(iL1.get_load()) << " mem: " << mem_interlists << "MB\n";
				std::cout << "SizeBaselists:       " << List1.get_load() << " log: " << log2(List1.get_load()) << " mem: " << mem_baselists+mem_baselists_ << "MB\n";
				std::cout << "TotalMem:            " << (mem_buckets+mem_outlist+mem_interlists+mem_baselists+mem_baselists_+mem_changelists)  << "MB\n";
#endif
			}
#endif

			constexpr uint64_t logging_value = 10000;
			if ((outer_loops % logging_value) == 0 && try_i == 0) {
				MULTITHREADED_WRITE(uint64_t val = outerloops_all.load() + (outer_loops != 0 ? logging_value : 1));
				MULTITHREADED_WRITE(outerloops_all.store(val));

				std::cout << MULTITHREADED_WRITE("Thread: " << std::this_thread::get_id() << " " << ) "outer finished: " << outer_loops << "\n" << std::flush;
				MULTITHREADED_WRITE(std::cout << "Alltogether: " << outerloops_all.load()  << "\n" << std::flush);
			}


			out.set_load(0);
			iL1.set_load(0);
			iL2.set_load(0);
		}

		outer_loops += 1;
		if (((c != 0) && (outer_loops > exit_loops)) MULTITHREADED_WRITE(|| finished.load())){
			goto finish;
		}
	}

	finish:
	std::cout <<  MULTITHREADED_WRITE("Thread: " << std::this_thread::get_id() << " " << ) "number of outer loops: " << outer_loops << "\n";


	delete (bucket_level1);
	delete (bucket_level2);
	mzd_free(work_matrix_H);
	mzd_free(work_matrix_H_T);
	mzd_free(working_s);
	mzd_free(working_s_T);
	mzd_free(H_prime);
	mzd_free(H_prime_T);
	mzd_free(columnTransposed);
	mzp_free(P_C);

	free_matrix_data(matrix_data);

	//return return_value;
	return outer_loops;
}


template<const uint64_t n, const uint64_t c, const uint64_t k, const uint64_t l, const uint64_t w, const uint64_t p, const uint64_t d,
        const uint64_t l1, const uint64_t w1, const uint64_t l2, const uint64_t w2, const uint64_t epsilon, const uint64_t r1=0, const uint64_t number_buckets=13, const uint64_t tries=1, const uint64_t number_bucketsearch_retries_=1, const uint64_t exit_loops=99999, class DecodingList=DecodingList>
int emz_d2(mzd_t *e, const mzd_t *const s, const mzd_t *const A
#ifdef CHECK_PERM
		, const mzd_t *correct_e
#endif
) {
	ASSERT(A!= nullptr && n-c == A->ncols && n-k == A->nrows && n >= k+l && w > p && n-k == s->ncols && 1 == s->nrows && e->nrows == 1 && s->nrows == 1 && c < n-k-l);

	// reset the mutlithreading stuff
	MULTITHREADED_WRITE(finished.store(false);)
	MULTITHREADED_WRITE(outerloops_all.store(0);)


	// recalculated lengths if we cut of `c` coordinates
	constexpr uint64_t n_prime = n-c;
	constexpr uint64_t k_prime = k-c;

	constexpr uint64_t size_baselist = bc(l1/2, w1)*bc((k_prime/2)+epsilon,w2);

	typedef typename DecodingList::LabelType DecodingLabel;
	typedef typename DecodingList::ValueType DecodingValue;
	typedef typename DecodingList::ElementType DecodingElement;
	typedef typename DecodingList::MatrixType DecodingMatrix;

	// Baselists which are shared between threads
	DecodingList BaseList1{size_baselist},  BaseList2{size_baselist};

	/* old list construction, where only one weight was represented in the lists.
	std::vector<std::pair<uint64_t, uint64_t>> changelist11, changelist12, changelist21, changelist22;
	prepare_generate_base_mitm2(BaseList1, changelist11, changelist12, l1, k, l2, w1, w2, false, epsilon, true);
	prepare_generate_base_mitm2(BaseList2, changelist21, changelist22, l1, k, l2, w1, w2, true, epsilon, true);*/

	std::vector<std::vector<std::pair<uint64_t, uint64_t>>> changelist11, changelist12, changelist21, changelist22;
	prepare_generate_base_mitm2_extended<DecodingList>(BaseList1, changelist11, changelist12, l1, k_prime, l2, w1, w2, false, epsilon, true);
	prepare_generate_base_mitm2_extended<DecodingList>(BaseList2, changelist21, changelist22, l1, k_prime, l2, w1, w2, true, epsilon, true);

#if (defined(NUMBER_THREADS) && NUMBER_THREADS > 1)

	int threads_finished = 0;
	std::vector<std::thread> threads{NUMBER_THREADS};
	for( auto & t : threads ) {
		//t = std::thread(mc_eliece_d2_thread<n,c,k,l,w,p,d,l1,w1,l2,w2,epsilon,r1,number_buckets,tries,number_bucketsearch_retries_,exit_loops, DecodingList>, e, s, A, std::ref(BaseList1), std::ref(BaseList2), std::ref(changelist11), std::ref(changelist12), std::ref(changelist21), std::ref(changelist22));
		t = std::thread(mc_eliece_d2_thread<n,c,k,l,w,p,d,l1,w1,l2,w2,epsilon,r1,number_buckets,tries,number_bucketsearch_retries_,exit_loops, DecodingList>, e, s, A, (BaseList1), (BaseList2), (changelist11), (changelist12), (changelist21), (changelist22)
#ifdef CHECK_PERM
		, correct_e
#endif
);
	}
	for (unsigned int i = 0; i < NUMBER_THREADS; i++) {
		threads[i].join();
	}
	std::cout << "number of all perm: " << outerloops_all.load();
#else
	int threads_finished = emz_d2_thread<n, c, k, l, w, p, d, l1, w1, l2, w2, epsilon, r1, number_buckets, tries, number_bucketsearch_retries_, exit_loops, DecodingList>(
			e, s, A, BaseList1, BaseList2, changelist11, changelist12, changelist21, changelist22
#ifdef CHECK_PERM
			, correct_e
#endif
	);
#endif

	return threads_finished;
}

/// Wrapper function around `emz_2d`
template<const uint64_t n, const uint64_t c, const uint64_t k, const uint64_t l, const uint64_t w, const uint64_t p, const uint64_t d,
		const uint64_t l1, const uint64_t w1, const uint64_t l2, const uint64_t w2, const uint64_t epsilon, const uint64_t r1=0, const uint64_t number_buckets=13, const uint64_t tries=1,  const uint64_t number_bucketsearch_retries_=1, const uint64_t exit_loops=99999>
int emz_d2_outer(mzd_t *e, const mzd_t *const s, const mzd_t *const A) {
	constexpr uint64_t n_prime = n-c;
	constexpr uint64_t k_prime = k-c;

#ifdef CHECK_PERM
	mzd_t *correct_e = mzd_from_str(1,n,correct_e_str);
#endif

	// local variables
	mzd_t *e_prime = mzd_init(1, n-c);
	mzd_t *full_e_befor_perm = mzd_init(1, n);
	mzd_t *A_prime = mzd_init(n-k, n_prime);
	mzd_t *A_ = mzd_init(n-k, n);
	mzd_copy(A_, A);
	mzp_t *P_C = mzp_init(n);
	for(rci_t i = 0; i < n; i++) P_C->values[i] = i;

	int ret = 0;
	while (ret == 0) {
		matrix_create_random_permutation(A_, P_C);
		mzd_submatrix(A_prime, A_, 0, c, n-k, n);
		if (A_prime == nullptr)
			return 0;

		ret = emz_d2<n, c, k, l, w, p, d, l1, w1, l2, w2, epsilon, r1, number_buckets, tries, number_bucketsearch_retries_, exit_loops>(
				e_prime, s, A_prime
#ifdef CHECK_PERM
				, correct_e
#endif
		);

	}

	for(rci_t j=0; j < n-c; ++j) {
		mzd_write_bit(full_e_befor_perm, 0,c+j, mzd_read_bit(e_prime, 0, j));
	}

	for(rci_t j=0; j < n; ++j) {
		mzd_write_bit(e, 0, P_C->values[j], mzd_read_bit(full_e_befor_perm, 0, j));
	}

	// free local variables.
	mzd_free(A_prime);
	mzd_free(A_);
	mzd_free(full_e_befor_perm);
	mzd_free(e_prime);
	mzp_free(P_C);
	return ret;
}
#endif //SMALLSECRETLWE_EMZ_H
