#ifndef DECODING_MONOIM
#define DECODING_MONOIM

#include <cstdint>
#include <utility>
#include "helper.h"

#include "isd.h"
#include "simd/simd.h"
#include "math/bc.h"
#include "container/hashmap.h"
#include "nn/nn.h"

class ConfigMO : public ConfigISD{
public:
	const uint32_t l1;
	const uint32_t l2;

	const uint32_t nr_IM = 2;

	// size of the intermediate cache list
	const size_t intermediate_list_max_size = 102*8; // TODO this is needed, to be?

	// if set to true, the intermediate index list will be reset.
	const bool reset_intermediate_list = true;

	/// number of `limbs` to walk down in the algorithm
	constexpr static uint32_t r = 2;

	/// number of leaves on each iteration level
	constexpr static uint32_t N = 100;

	///
	constexpr static uint32_t nnk = 32;

	/// weight to match on each check
	constexpr static uint32_t dk = 12;

	/// allowed offset (unused)
	constexpr static uint32_t epsilon = 0;

	/// bruteforce the remaining list below this threshold
	constexpr static uint32_t bruteforce_threshold = 1024;

	/// expected number of permutations (ignores the representations)
	[[nodiscard]] consteval uint64_t compute_loops() const noexcept {
#ifdef EXPECTED_PERMUTATIONS
		return EXPECTED_PERMUTATIONS;
#else
		return bc(n, w)/(bc(n-k-l, w-4*p) * bc((k+l)/2, 2*p) * bc((k+l)/2, 2*p));
#endif
	}

	// returns the expected number of iterations
	constexpr uint64_t Loops() const {
#ifdef EXPECTED_PERMUTATIONS
		return EXPECTED_PERMUTATIONS;
#else
		return bc(n, w)/( bc(n-k-l, w-2*p) * bc((k+l)/2, p) * bc((k+l)/2, p));
#endif
	}

	// print some information about the configuration
	void print() const {
		std::cout << "{: \"l1\": " << l1
		          << ", \"l2\": " << l2
				  << " }" << std::endl;
	}
};

/// implementing May-Ozerov WITHOUT Indyk Montwani NN
/// \tparam config
template<const ConfigISD &isd, const ConfigMO &config>
class MayOzerov : public ISDInstance<uint64_t, isd>{
public:
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t k = config.k;
	constexpr static uint32_t w = config.w;
	constexpr static uint32_t p = config.p; // NOTE: from now on p is the baselist p not the full p.
	constexpr static uint32_t l1 = config.l1;
	constexpr static uint32_t l2 = config.l2;
	constexpr static uint32_t l = config.l;
	static_assert(l1 + l2 == l);

	/// describes the number of columns to permute, instead of doing the full permutation
	/// if set to 0 the normal permutation and gaussian elimination is applied.
	constexpr static uint32_t gaus_c = 0;

	//
	constexpr static bool init_MO = true;
	constexpr static bool MO = false;
	constexpr static uint32_t kl_half = (k+l1)/2 ;

	// base list shit
	constexpr static size_t baselist_enumeration_length = kl_half;
	constexpr static size_t baselist_enumeration_weigth = p;
	constexpr static size_t baselist_enumeration_size =
		bc(baselist_enumeration_length, baselist_enumeration_weigth);

	constexpr static uint32_t HM1_NRBUCKETS = 1ul << l1;
	// 2* (list_enumeration_size*list_enumeration_size) >> l;
	constexpr static uint32_t HM1_BUCKETSIZE = 106*40; // TODO move to config

	// number of limbs in each row of HT
	constexpr static uint32_t NR_HT_LIMBS = (n-k-l1+63)/64;

	// base datatype of the hashmap
	using listType = uint32_t; 	// DO NOT CHANGE

	/// list entries of the current permutation
	/// if a solution was found.
	listType solutions[2*p] = {0};

	/// internal arrays pointing to
	uint64_t *pHT, *pHTr;   			/// the n-k-l part
	uint32_t *lHT, *lHTr, syndrome;     /// the l part

	/// This implementation buffers up
	/// Maximal list size of the temporary intermediate list
	constexpr static size_t intermediate_list_max_size = config.intermediate_list_max_size;

	/// TODO mach es genauso wie in `stern_mo.h`
	/// NOTE: that we increase the size with about one bucket size to make sure that we can fully insert a bucket
	alignas(64) std::array<listType, intermediate_list_max_size + HM1_BUCKETSIZE> intermediate_L2_left;
	alignas(64) std::array<listType, intermediate_list_max_size + HM1_BUCKETSIZE> intermediate_L2_right;
	alignas(64) std::array<listType, intermediate_list_max_size + HM1_BUCKETSIZE> intermediate_L4_left;
	alignas(64) std::array<listType, intermediate_list_max_size + HM1_BUCKETSIZE> intermediate_L4_right;
	uint32_t intermediate_L2_ctr = 0, intermediate_L4_ctr = 0;

	/// Intermediate lists, computed on the full length of `n-k-\ell`
	uint64_t *L = nullptr, *R = nullptr;
	size_t L_ctr = 0, R_ctr = 0;

	// TODO the value 10
	constexpr static size_t NR_LR = 10 * (baselist_enumeration_size*baselist_enumeration_size) >> l1;
	constexpr static size_t size_LR = roundToAligned<64>(NR_LR * sizeof(uint64_t) * NR_HT_LIMBS);

	/// NN parameters
	/// number of `limbs` to walk down in the algorithm
	constexpr static uint32_t r = config.r;

	/// number of leaves on each iteration level
	constexpr static uint32_t N = config.N;

	///
	constexpr static uint32_t nnk = config.nnk;

	/// weight to match on each check
	constexpr static uint32_t dk = config.dk;

	/// weight to exactly match on
	constexpr static uint32_t d = w - 4;

	/// allowed offset (unused)
	constexpr static uint32_t epsilon = config.epsilon;

	/// bruteforce the remaining list below this threshold
	constexpr static uint32_t bruteforce_threshold = config.bruteforce_threshold;

	/// Define the nearest neighbour struct
	constexpr static NN_Config nn_config{n-k-l1, r, N, nnk, size_LR, dk, d, epsilon, bruteforce_threshold};
	NN<nn_config> nn_algo {};
	using NNElement= typename NN<nn_config>::Element;

	/// internal hash function
	/// \tparam h max bit to return
	/// \param a value to hash
	/// \return lower `h` bits of `a`
	template<const uint32_t h>
	inline static size_t Hash(uint64_t a) noexcept {
		using T = uint64_t;
		static_assert(h < sizeof(T) * 8);
		constexpr T mask = (1u << h) - 1u;
		return a & mask;
	};

	// Baselist
	constexpr static SimpleHashMapConfig simpleHashMapConfig{HM1_BUCKETSIZE, HM1_NRBUCKETS};
	using HM = SimpleHashMap<uint32_t, listType, simpleHashMapConfig, &Hash<l1>>;
	HM *hm = nullptr;

	/// TODO how to choose correctly?
	/// GHZ of the cpu running, needed to compute the total runtime.
	constexpr static uint64_t ghz = 2000000000;

	/// not constant values.
	uint64_t loops = 0;
	bool not_found = true;

	constexpr MayOzerov() noexcept {
		/// malloc the l part sub matrix
		constexpr size_t size_lHT = roundToAligned<64>(sizeof(uint32_t) * (k+l1));
		lHT = (uint32_t *)cryptanalysislib::aligned_alloc(64, size_lHT);
		lHTr = lHT + baselist_enumeration_length;
		assert(lHT);

		/// malloc the n-k-l sub matrix
		constexpr size_t size_pHT = roundToAligned<64>(sizeof(uint64_t) * (k+l1) * NR_HT_LIMBS);
		pHT = (uint64_t *)cryptanalysislib::aligned_alloc(64, size_pHT);
		pHTr = pHT + baselist_enumeration_length*NR_HT_LIMBS;
		assert(pHT);

		/// only init in the base class
		if (init_MO) {
			/// malloc the list for the `n-k-l` part
			L = (uint64_t *) cryptanalysislib::aligned_alloc(64, size_LR);
			R = (uint64_t *) cryptanalysislib::aligned_alloc(64, size_LR);
			assert(L);
			assert(R);
		}

		// hashmap only for the base list
		hm = new HM();
		ASSERT(hm);
	}

	/// Deconstructor, freeing alot of stuff
	constexpr ~MayOzerov() noexcept {
		free(lHT);
		free(pHT);

		// Done by the NN algo
		//free(L);
		//free(R);
		delete hm;
	}


	/// helper function. Checks if HT[i] == value for i in biject(index)
	/// \param index `index`-th element in the baselist
	/// \param value check for equality
	/// \return true/false
	bool check_hashmap(const size_t index, const uint32_t value) noexcept {
		uint16_t rows[p];
		biject<baselist_enumeration_length, p>(index, rows);

		uint32_t value2 = 0;
		for (uint32_t i = 0; i < p; ++i) {
			value2 ^= lHT[rows[i]];
		}

		return value2 == value;
	}

	///
	/// \return
	bool check_LR() noexcept {
		constexpr uint64_t mask = ~((1ull << ((n-k-l1)%64u)) - 1ul);

		for (size_t i = 0; i < L_ctr; i++) {
			const uint64_t  tmp = L[i*NR_HT_LIMBS + (NR_HT_LIMBS-1)];
			if ((tmp & mask) != 0) {
				for (uint32_t j = 0; j < NR_HT_LIMBS; j++) {
					print_binary<uint64_t>(L[i*NR_HT_LIMBS + j]);
				}
				return false;
			}
		}

		for (size_t i = 0; i < R_ctr; i++) {
			if ((R[i*NR_HT_LIMBS + (NR_HT_LIMBS-1)] & mask) != 0) {

				for (uint32_t j = 0; j < NR_HT_LIMBS; j++) {
					print_binary<uint64_t>(R[i*NR_HT_LIMBS + j]);
				}
				return false;
			}
		}

		return true;
	}

#ifdef USE_AVX2
	/// first clears the internal hashmap
	/// hashes the left side into the internal hashmap
	/// NOTE: this is the avx2 version. 8 elements < 32bit are hashed simultaneous
	void fill_hashmap_avx2() noexcept {
		// first clear it
		hm->clear();
		static_assert(l < 32);

		alignas(64) __m256i rows[p];
		__m256i a = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
		const __m256i eight = _mm256_set1_epi32(8);


		/// TODO what happens if the baselist enum ize is not a multiple of 8
		for (size_t i = 0; i < (baselist_enumeration_size)/8; i++) {
			biject_avx<baselist_enumeration_length, p>(a, rows);
			a = _mm256_add_epi32(a, eight);
			alignas(64) __m256i tmp = _mm256_set1_epi32(0);

			#pragma unroll
			for (uint16_t j = 0; j < p; j++) {
				tmp ^= _mm256_i32gather_epi32(lHT, rows[j], 4);
			}

			const U256i u = { tmp };
			for (uint16_t j = 0; j < 8; j++) {
				hm->insert(u.a[j], i*8 + j);
			}
		}
	}
#endif

	/// construct the MO intermediate lists L, R from the cache lists
	/// TODO avx2.
	void construct_LR() noexcept {
		size_t min_ctr = std::min(intermediate_L2_ctr, intermediate_L4_ctr);
		uint16_t rows_l_L2[p], rows_r_L2[p],
				 rows_l_L4[p], rows_r_L4[p];

		for (size_t i = 0; i < min_ctr; i++) {
			ASSERT(intermediate_L2_left[i]  < baselist_enumeration_size);
			ASSERT(intermediate_L2_right[i] < baselist_enumeration_size);
			ASSERT(intermediate_L4_left[i]  < baselist_enumeration_size);
			ASSERT(intermediate_L4_right[i] < baselist_enumeration_size);

			biject<baselist_enumeration_length, p>(intermediate_L2_left[i] , rows_l_L2);
			biject<baselist_enumeration_length, p>(intermediate_L2_right[i], rows_r_L2);
			biject<baselist_enumeration_length, p>(intermediate_L4_left[i],  rows_l_L4);
			biject<baselist_enumeration_length, p>(intermediate_L4_right[i], rows_r_L4);

			for (uint16_t a = 0; a < NR_HT_LIMBS; ++a) {
				uint64_t tmp_L = 0, tmp_R = ws->rows[0][a];

				for (uint16_t j = 0; j < p; j++) {
					ASSERT(rows_l_L2[j] < kl_half);
					ASSERT(rows_r_L2[j] < kl_half);
					ASSERT(rows_l_L4[j] < kl_half);
					ASSERT(rows_r_L4[j] < kl_half);

					tmp_L ^= pHT [rows_l_L2[j]*NR_HT_LIMBS + a];
					tmp_L ^= pHTr[rows_r_L2[j]*NR_HT_LIMBS + a];
					tmp_R ^= pHT [rows_l_L4[j]*NR_HT_LIMBS + a];
					tmp_R ^= pHTr[rows_r_L4[j]*NR_HT_LIMBS + a];
				}

				L[L_ctr*NR_HT_LIMBS + a] = tmp_L;
				R[R_ctr*NR_HT_LIMBS + a] = tmp_R;
			}

			/// increase the counter
			L_ctr += 1;
			R_ctr += 1;
			assert(L_ctr < NR_LR);
			assert(R_ctr < NR_LR);
		}

		// work through the last elements
		uint64_t init_value[NR_HT_LIMBS] = {0};
		uint64_t *T = L;
		size_t *T_ctr = &L_ctr;
		uint32_t *cctr = &intermediate_L2_ctr;
		uint32_t *MO_index_l = intermediate_L2_left.data(),
		         *MO_index_r = intermediate_L2_right.data();
		if (intermediate_L2_ctr < intermediate_L4_ctr) {
			for (uint16_t i = 0; i < NR_HT_LIMBS; ++i) {
				init_value [i] = ws->rows[0][i];
			}

			T = R;

			/// NOTE: this one is the only thing we need to set contra naming
			cctr = &intermediate_L4_ctr;
			T_ctr = &R_ctr;

			MO_index_l = intermediate_L4_left.data();
			MO_index_r = intermediate_L4_right.data();
		}

		for (uint32_t i = min_ctr; i < (*cctr); i++) {
			assert(MO_index_l[i] < baselist_enumeration_size);
			assert(MO_index_r[i] < baselist_enumeration_size);
			biject<baselist_enumeration_length, p>(MO_index_l[i], rows_l_L2);
			biject<baselist_enumeration_length, p>(MO_index_r[i], rows_r_L2);

			for (uint16_t a = 0; a < NR_HT_LIMBS; ++a) {
				uint64_t tmp = init_value[a];

				for (uint16_t j = 0; j < p; j++) {
					ASSERT(rows_l_L2[j] < kl_half);
					ASSERT(rows_r_L2[j] < kl_half);

					tmp ^= pHT [rows_l_L2[j]*NR_HT_LIMBS + a];
					tmp ^= pHTr[rows_r_L2[j]*NR_HT_LIMBS + a];
				}

				T[(*T_ctr)*NR_HT_LIMBS + a] = tmp;
			}

			/// increase the counter
			*T_ctr += 1;
			assert((*T_ctr) < NR_LR);
		}

		if constexpr (config.reset_intermediate_list) {
			/// reset the counters
			intermediate_L2_ctr = 0;
			intermediate_L4_ctr = 0;
		}
	}

	/// NOTE: assumes that the hashmap is already filled_mm256_i32gather_epi64
	/// computes the left an right intermediate list L, R for the may ozerov
	/// algorithm
	void intermediate_lists() noexcept {
		uint16_t rows[p];

		for (size_t right_index = 0; right_index < baselist_enumeration_size; ++right_index) {
			biject<baselist_enumeration_length, p>(right_index, rows);

			// TODO IT
			uint32_t tmpl = 0; uint32_t tmpr;

			for (uint32_t j = 0; j < p; ++j) {
				tmpl ^= lHTr[rows[j]];
			}
			tmpr = tmpl ^ syndrome;


			const size_t hashmap_offset_L2 = tmpl*HM1_BUCKETSIZE;
			const size_t hashmap_offset_L4 = tmpr*HM1_BUCKETSIZE;

			const size_t left_load_L2 = hm->__internal_load_array[tmpl];
			const size_t left_load_L4 = hm->__internal_load_array[tmpr];

			for (uint64_t j = 0; j < left_load_L2; j++) {
				const listType left_index = hm->__internal_hashmap_array[hashmap_offset_L2 + j];

				assert(check_hashmap(left_index, tmpl));
				intermediate_L2_left [intermediate_L2_ctr] = left_index;
				intermediate_L2_right[intermediate_L2_ctr] = right_index;
				intermediate_L2_ctr += 1;
			}

			for (uint64_t j = 0; j < left_load_L4; j++) {
				const listType left_index = hm->__internal_hashmap_array[hashmap_offset_L4 + j];

				assert(check_hashmap(left_index, tmpr));
				intermediate_L4_left [intermediate_L4_ctr] = left_index;
				intermediate_L4_right[intermediate_L4_ctr] = right_index;
				intermediate_L4_ctr += 1;
			}

			/// NOTE: not really optimal but ok for debugging.
			construct_LR();
		}
	}


	/// runs NN on the full n-k-l intermediate lists L and R
	void compute_finale_list() noexcept {
		nn_algo.L1 = (NNElement *)L;
		nn_algo.L2 = (NNElement *)R;
		// nn_algo.avx2_nn(L_ctr, R_ctr);
		nn_algo.bruteforce_96(L_ctr, R_ctr);
		if (nn_algo.solutions_nr > 0) {

			std::cout << "ok this is crazy\n";
			not_found = false;

			solutions[0] = nn_algo.solutions[0].first;
			solutions[1] = nn_algo.solutions[0].second;
		}

	}

	/// TODO not finished
	void reconstruct() noexcept {
		uint16_t rows_l[p], rows_r[p];

		size_t right_index_L2=0, left_index_L2=0, right_index_L4=0, left_index_L4=0;
		/// first recompute the collision on the first level of the tree
		for (size_t right_index = 0; right_index < baselist_enumeration_size; ++right_index) {
			biject<baselist_enumeration_length, p>(right_index, rows_r);

			// TODO IT
			uint32_t tmpl = 0;
			uint32_t tmpr;

			for (uint32_t j = 0; j < p; ++j) {
				tmpl ^= lHTr[rows_r[j]];
			}
			tmpr = tmpl ^ syndrome;


			const size_t hashmap_offset_L2 = tmpl * HM1_BUCKETSIZE;
			const size_t hashmap_offset_L4 = tmpr * HM1_BUCKETSIZE;

			const size_t left_load_L2 = hm->__internal_load_array[tmpl];
			const size_t left_load_L4 = hm->__internal_load_array[tmpr];

			for (uint64_t j = 0; j < left_load_L2; j++) {
				const listType left_index = hm->__internal_hashmap_array[hashmap_offset_L2 + j];
				biject<baselist_enumeration_length, p>(left_index, rows_l);

				uint64_t tmp[NR_HT_LIMBS] = {0};
				for (uint16_t a = 0; a < NR_HT_LIMBS; ++a) {
					for (uint16_t a2 = 0; a2 < p; a2++) {
						tmp[a] ^= pHT[rows_l[a2]  * NR_HT_LIMBS + a];
						tmp[a] ^= pHTr[rows_r[a2] * NR_HT_LIMBS + a];
					}
				}

				bool equal = true;

				for (uint16_t a = 0; a < NR_HT_LIMBS; ++a) {
					if (tmp[a] != L[solutions[0] * NR_HT_LIMBS + a]) {
						equal = false;
						break;
					}
				}

				if (equal) {
					std::cout << "found left\n";
					right_index_L2 = right_index;
					left_index_L2 = left_index;
					goto found_L;
				}
			}

		found_L:
			for (uint64_t j = 0; j < left_load_L4; j++) {
				const listType left_index = hm->__internal_hashmap_array[hashmap_offset_L4 + j];
				biject<baselist_enumeration_length, p>(left_index, rows_l);

				uint64_t tmp[NR_HT_LIMBS] = {0};
				for (uint16_t a = 0; a < NR_HT_LIMBS; ++a) {
					tmp[a] = ws->rows[0][a];
					for (uint16_t a2 = 0; a2 < p; a2++) {
						tmp[a] ^= pHT[rows_l[a2]  * NR_HT_LIMBS + a];
						tmp[a] ^= pHTr[rows_r[a2] * NR_HT_LIMBS + a];
					}
				}

				bool equal = true;

				for (uint16_t a = 0; a < NR_HT_LIMBS; ++a) {
					if (tmp[a] != R[solutions[1] * NR_HT_LIMBS + a]) {
						equal = false;
						break;
					}
				}

				if (equal) {
					std::cout << "found right\n";
					right_index_L4 = right_index;
					left_index_L4 = left_index;
					goto found_R;
				}
			}
		}

		/// found both solutions
		found_R:

		uint16_t rows_l_L4[p], rows_r_L4[p];

		biject<baselist_enumeration_length, p>(right_index_L2, rows_r);
		biject<baselist_enumeration_length, p>(left_index_L2, rows_l);
		biject<baselist_enumeration_length, p>(right_index_L4, rows_r_L4);
		biject<baselist_enumeration_length, p>(left_index_L4, rows_l_L4);

		mzd_t *eprime = mzd_init(1, n);

		std::cout << loops << "\n";
		std::cout << solutions[0] << " " << solutions[1] << "\n";
		std::cout << right_index_L2 << " " << left_index_L2 << " " << right_index_L4 << " " << left_index_L4 << "\n";
		for (uint16_t i = 0; i < p; ++i) {
			rows_r[i] += baselist_enumeration_length;
			rows_r_L4[i] += baselist_enumeration_length;
			std::cout << rows_r[i] << " ";
			std::cout << rows_l[i] << " ";
			std::cout << rows_r_L4[i] << " ";
			std::cout << rows_l_L4[i] << "\n";

			mzd_write_bit(eprime, 0, n-k-l1 + rows_r[i], 1);
			mzd_write_bit(eprime, 0, n-k-l1 + rows_l[i], 1);
			mzd_write_bit(eprime, 0, n-k-l1 + rows_r_L4[i], 1);
			mzd_write_bit(eprime, 0, n-k-l1 + rows_l_L4[i], 1);
		}

		print_matrix("", eprime);

		for (uint32_t i = 0; i < NR_HT_LIMBS; ++i) {
			eprime->rows[0][i] ^= L[solutions[0] * NR_HT_LIMBS + i];
			eprime->rows[0][i] ^= R[solutions[1] * NR_HT_LIMBS + i];
		}

		print_matrix("", eprime);

		for (uint32_t i = 0; i < n; ++i) {
			mzd_write_bit(e, 0, P->values[i], mzd_read_bit(eprime, 0, i));
		}

		mzd_free(eprime);
    }

	// runs MO
	uint64_t run() noexcept {
		config.print();
		print();

		ASSERT(l2 == 0);
		loops = 0;
		not_found = true;
		
		uint64_t cycles = cpucycles(), gaus_cycles = 0;

		while (not_found && loops < config.loops) {
			gaus_cycles -= cpucycles();
			matrix_echelonize_partial_plusfix_opt
					<n, k, l1>
					(wA, wAT, config.m4ri_k, n - k - l1, md, P);
			gaus_cycles += cpucycles();

			// Extract the sub-matrices. the length of the matrix is only n-c + DOOM_nr but we need to copy everything.
			mzd_submatrix(H, wA, 0, n-k-l1, n-k, n+1);
			mzd_transpose(HT, H);

			//print_matrix("", HT);

			extract_l1HT();
			extract_pHT();

			extract_syndrome();
#ifdef USE_AVX2
			fill_hashmap_avx2();
#else
			fill_hashmap();
#endif
			// intermediate_lists_avx2();

			//print_matrix("", wA, 0, n-k, n, n+1);
			//print_matrix("", ws);
			//print_binary(syndrome);

			// TODO
			if (syndrome == 0)
				continue;

			L_ctr = 0;
			R_ctr = 0;
			intermediate_lists();


			/// flush the remaining elements in the cache
			///construct_LR();
			ASSERT(check_LR());

			// clean up if needed. Internally this function is only execute if `coll_hash_simple_avx()`
			// left some elements over.
			compute_finale_list();

			loops += 1;

			// print information
			if (loops == 1) {
				const size_t load = hm->load();
				const double load_per_bucket = double(load) / HM1_NRBUCKETS;
				printf("nrbuckets: %d\n", HM1_BUCKETSIZE);
				printf("n: %d, k: %d, l: %d, p: %d, HM load/Bucket: %.2f\n",
				               n, k, l, p, load_per_bucket);
			}

			if (loops % 128 == 0) {
				uint64_t cyc     = cpucycles() - cycles;
				double time      = double(cyc) / ghz;
				double gaus_proc = 100 * double(gaus_cycles) / cyc;
				double lps       = double(loops) / time; 

				printf("loops: %" PRIu64 ", sec: %f, gaus: %.2fp, lps %f\n",
						loops, time, gaus_proc, lps);
			}
		}

		uint64_t cyc     = cpucycles() - cycles;
		double time      = double(cyc) / ghz;

		// compute the final solution
		reconstruct();

		printf("loops: %" PRIu64 " time: %f %flps\n", loops, time, double(loops)/time);
		return loops;
	}

	/// print important information about the instantiation of the algorithm which is not
	/// part of the config
	void print() const {
		std::cout
		        << "N: " << N
		        << ", r: " << r
		        << ", dk: " << dk
				<< ", d: " << d
		        << ", bf_thresh: " << bruteforce_threshold
		        << ", epsilon: " << epsilon
		        << "\n";
	}
};

#endif
