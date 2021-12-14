#ifndef SMALLSECRETLWE_POLLARD_H
#define SMALLSECRETLWE_POLLARD_H

#include "helper.h"
#include "glue_m4ri.h"
#include "custom_matrix.h"
#include "combinations.h"
#include "list.h"
#include "sort.h"

#include <type_traits>
#include <algorithm>
#include <omp.h>

/// configuration structure of the pollard=memoryless Lee-Brickell algorithm
///	Usage:
///		static constexpr ConfigPollard config(n, k, w, 1, 8, 1 << 7);
struct ConfigPollard {
public:
	const uint32_t n, k, w;     // instance Parameters
	// this is the final weight on the baselist.
	const uint32_t baselist_p;
	const uint32_t l;
	const uint32_t nr_steps;

	// at what weight on the first n-k-l coordinates do we classify a label as a solution.
	const uint32_t weight_threshhold = w - 4 * baselist_p;

	const int m4ri_k; // opt parameter for faster gaus elimination
					  // calculated on the fly. You normally dont have to touch this.

	const uint32_t nr_threads = 1;

	const bool UseBrent = false;
	const bool DOOM = false;

	constexpr ConfigPollard(const uint32_t n,           // instance parameter
							const uint32_t k,           // instance parameter
							const uint32_t w,           // instance parameter
							const uint32_t baselist_p,  // weight in the baselist
							const uint32_t l,           // l window to match on
							const uint32_t nr_steps     // number of repetitions
							) :
			n(n), k(k), w(w), baselist_p(baselist_p), l(l), nr_steps(nr_steps),
			m4ri_k(matrix_opt_k(n - k, MATRIX_AVX_PADDING(n))) {};

	// prints information about the problem instance.
	void print() const {
		std::cout << "n: " << n << ", k: " << k <<  ", p: " << baselist_p << ", l: " << l << "\n";
	}
};

///	Pollard/Memoryless Lee-Brickell
/// Usage:
///		static constexpr ConfigPollard config(G_n, G_k, G_w, 1, 8, 1 << 7);
///		Pollard<config> pollard(ee, ss, A);
///		pollard.run();
/// \tparam config
template<const ConfigPollard &config>
class Pollard {
public:
	// constants from the problem instance
	constexpr static uint32_t n     = config.n;
	constexpr static uint32_t k     = config.k;
	constexpr static uint32_t kh    = config.k/2;
	constexpr static uint32_t w     = config.w;
	constexpr static uint32_t p     = config.baselist_p;
	constexpr static uint32_t l     = config.l;
	constexpr static uint32_t lprime= (l-1)/2; //const_log(bc(k, p)) + 1;
	constexpr static uint32_t nkl   = n - k - l;
	constexpr static uint32_t c     = 0;

	constexpr static uint32_t threads       = config.nr_threads;            // number of threads
	constexpr static uint32_t lVCLTBits     = sizeof(uint64_t)*8;           // bits of the label
	constexpr static uint32_t loffset       = nkl / lVCLTBits;              //
	constexpr static uint32_t lshift        = nkl - (loffset * lVCLTBits);  //

	// Main datatypes
	using DecodingValue     = Value_T<BinaryContainer<k>>;
	using DecodingLabelOut  = Label_T<BinaryContainer<n-k>>;
	using DecodingLabel     = Label_T<BinaryContainer<l>>;

	using DecodingMatrix    = mzd_t *;
	using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList      = Parallel_List_T<DecodingElement>;
	using ChangeList        = std::vector<std::pair<uint32_t, uint32_t>>;

	typedef typename DecodingList::ValueContainerType ValueContainerType;
	typedef typename DecodingList::ValueContainerLimbType ValueContainerLimbType;
	typedef typename DecodingLabelOut::ContainerType LabelOutContainerType;

	using LabelContainerType     = LogTypeTemplate<l>;
	using LabelContainerLimbType = LogTypeTemplate<l>;
	LabelContainerType iSyndrom;
	LabelOutContainerType wSyndrom;

	// flavor: randomize the each trail of the algorithm
	ValueContainerLimbType r;

	// M4Ri matrix structures
	mzd_t *wH, *wHT, *sT, *H, *HT, *iHT;
	mzp_t *permutation;
	customMatrixData *matrix_data;

	// Instance parameters
	mzd_t *e;
	const mzd_t *s;
	const mzd_t *A;

	// Test/Bench parameter
	uint32_t ext_tid = 0;
	bool not_found = true;

	// helper array. Position [x][y] corresponds to binom(x, y)
	uint64_t **binom;

	/// Constructor
	/// \param e output error
	/// \param s input syndrome
	/// \param A input matrix
	/// \param ext_tid external thread id. Used to init the internal prng
	Pollard(mzd_t *e, const mzd_t *const s, const mzd_t *const A,
			const uint32_t ext_tid = 0)
			: e(e), s(s), A(A), ext_tid(ext_tid) {
		static_assert(n > k, "wrong dimension");
		static_assert(p <= 2);
		static_assert(l < 64);

		// reset the found indicator variable.
		not_found = true;

		// Seed the internal prng.
		srand(ext_tid + time(NULL));
		random_seed(ext_tid + rand() * time(NULL));


#if defined(NUMBER_OUTER_THREADS) && NUMBER_OUTER_THREADS != 1
		// Ok this is ridiculous.
		// Apparently m4ri mzd_init is not thread safe. Cool.
#pragma omp critical
		{
#endif
		// Transpose the input syndrom
		sT = mzd_init(s->ncols, s->nrows);
		mzd_transpose(sT, s);

		wH          = matrix_init(n - k, n + 1);
		wHT         = mzd_init(wH->ncols, wH->nrows);
		mzd_t *tmp  = matrix_concat(nullptr, A, sT);
		mzd_copy(wH, tmp);
		mzd_free(tmp);
		H           = mzd_init(n - k, k);
		HT          = matrix_init(H->ncols, H->nrows);
		iHT         = mzd_init(k, l);

		// init the helper struct for the gaussian elimination and permutation data structs.
		matrix_data = init_matrix_data(wH->ncols);
		permutation = mzp_init(n-c);

		// Check if the working matrices are all allocated
		if ((wH == NULL) || (wHT == NULL) || (sT == NULL) ||
		    (H == NULL) || (HT == NULL) || (permutation == NULL) || (matrix_data == NULL)) {
			std::cout << "ExtTID: " << ext_tid << ", alloc error\n";
		}

		// create a lookup table for binomial coefficients for every (i, j) in [k, p]
		binom = new uint64_t*[k + 1];
		for (int i = 0; i < k + 1; ++i) {
			binom[i] = new uint64_t[p + 1];
			for (int j = 0; j<p + 1; ++j) {
				binom[i][j] = bc(i, j);
			}
		}

#if defined(NUMBER_OUTER_THREADS) && NUMBER_OUTER_THREADS != 1
		}
#endif
	}

	~Pollard() {
#if defined(NUMBER_OUTER_THREADS) && NUMBER_OUTER_THREADS != 1
		// I have no idea why.
#pragma omp critical
		{
#endif
		mzd_free(wH);
		mzd_free(wHT);
		mzd_free(sT);
		mzd_free(H);
		mzd_free(HT);
		mzd_free(iHT);
		mzp_free(permutation);

		free_matrix_data(matrix_data);
#if defined(NUMBER_OUTER_THREADS) && NUMBER_OUTER_THREADS != 1
		}
#endif
	}

	/// helper function printing `val` in binary, `str` is printed out additionally in the end.
	/// \param val	value to print
	/// \param str	helper string to identiy your value in a stream of printings
	void print_binary(const LabelContainerType val, const std::string &str) noexcept {
		LabelContainerType v = val;
		for (int i = 0; i < l; ++i) {
			std::cout << (v&1);
			v >>= 1;
		}
		std::cout << " " << str;
		std::cout << "\n";
	}

	/// extracts the bits [n-k-l, ..., n-k]
	/// \param v3	pointer to the label to extract the l bits from
	/// \return	l bits
	constexpr inline LabelContainerType extract_l_bit(uint64_t *v3) noexcept {
		uint64_t ret;
		if constexpr ((nkl / lVCLTBits) == ((n - k - 1) / lVCLTBits)) {
			// if the two limits are in the same limb
			ret = v3[loffset];
			ret >>= lshift;
		} else {
			// the two limits are in two different limbs
			__uint128_t data = v3[loffset];
			data            += (__uint128_t(v3[loffset+1]) << 64);
			ret = data >> lshift;
		}

		// IMPORTANT AUTO CAST HERE
		return ret;
	}

	// sample a random starting point with weight p
	constexpr static void start_point(ValueContainerType &e1, uint32_t bitpos[p]) noexcept {
		e1.random_with_weight(p);
		e1.get_bits_set(bitpos, p);
	}

	///  perfect hash function, maps l bits to an index of the list containing binom(k, p) elements, which are
	///		in a normal Lee Brickell implementation the labels one would test.
	///	Additionally a `next_bit` flag is needed if one add the syndrome or not.
	///	Works with every weight p
	/// \param bitpos		output: Entries in the non exisitant list, there indices if columns in the matrix to add up.
	/// \param next_bit		output: Flag specifying to add the syndrome or not. The first bit of `label` is used for this.
	/// \param label		input: label to hash
	/// \return something
	constexpr uint64_t hash22(uint32_t bitpos[p], uint16_t *next_bit, const LabelContainerType label) noexcept {
		uint64_t a = (label >> 1) % bc(k, p);
		*next_bit = label & 1;

		uint32_t wn = k;
		uint32_t wk = p;
		uint64_t v = 0;
		int set = 0;
		while (wn != 0) {
			if (set == p)
				break;
			else if (wn + set == p) {
				bitpos[set] = (wn - 1);
				wn -= 1;
				set += 1;
			}
			else if (a < binom[wn - 1][wk])
				wn -= 1;
			else {
				a -= binom[wn - 1][wk];
				bitpos[set] = (wn - 1);

				wn -= 1;
				wk -= 1;
				set += 1;
			}
		}

		return v;
	}

	/// probabilistic version of hash22, meaning its not perfect.
	/// See `hash22` for the meaning of the paramters
	/// works only with weight p = 2
	/// \param bitpos	output
	/// \param next_bit	output
	/// \param label	input
	constexpr inline static void hash2(uint32_t bitpos[p], uint16_t *next_bit, const LabelContainerType label) noexcept {
		// only usable if p = 2
		//	0   l/2   l-1 l
		// [     |      | ]
		constexpr LabelContainerLimbType mask1 = (LabelContainerLimbType(1) << (1+lprime)) - 1;
		constexpr uint64_t mod = k;// bc(k, p);
		bitpos[0] = ((label & mask1) >> 1)  % mod;
		bitpos[1] = (label >> (l-lprime))   % mod;
		if (bitpos[0] == bitpos[1])
			bitpos[1] = (bitpos[1] + 1)     % mod;

		*next_bit = label & 1;
	}

	/// same as hash2 but only for weight p = 1
	/// \param bitpos 	output
	/// \param next_bit output
	/// \param label 	input
	constexpr static inline void hash1(uint32_t bitpos[p], uint16_t *next_bit, const LabelContainerType label) noexcept {
		// only usable if p = 1
		//	0          l-1 l
		// [            | ]
		constexpr LabelContainerLimbType mask1 = ((LabelContainerLimbType(1) << l) - 1);
		bitpos[0] = ((label & mask1) >> 1) % k;
		*next_bit = label & 1;
	}

	/// calculates H times the error vector e on l bits.
	/// \param label 	output label He
	/// \param bitpos 	input the bit positions of the bits set in the corresponding value. E.g. nothing else than the columns to sum up.
	constexpr inline void calcHe(LabelContainerType &label, const uint32_t bitpos[p]) noexcept {
		ASSERT((bitpos[0]) < iHT->nrows);
		if constexpr(p == 2) {
			ASSERT((bitpos[1]) < iHT->nrows);
		}

		for (uint32_t i = 0; i < p; ++i) {
			label ^= iHT->rows[bitpos[i]][0];
		}
	}

	/// function1 only calcs the matrix vector multiplication
	/// \param label	output = He
	/// \param bitpos	bit positions of the error vector
	constexpr inline void f1(LabelContainerType &label, uint32_t bitpos[p]) noexcept {
		label = r;
		calcHe(label, bitpos);
	}

	/// function2 nearly the same as f1, but adds the syndrome additonally
	/// \param label 	output = He + s
	/// \param bitpos 	bit positions of error vector
	/// \return
	constexpr inline void f2(LabelContainerType &label, uint32_t bitpos[p]) noexcept {
		label = iSyndrom;
		label ^= r;

		calcHe(label, bitpos);
	}

	/// calculates one step
	/// \param label
	/// \param bitpos
	/// \return
	constexpr inline void step(LabelContainerType &label, uint32_t bitpos[p]) noexcept {
		uint16_t next;

		// reset the error vector
		for (uint32_t i = 0; i < p; ++i) {
			bitpos[i] = 0;
		}

		// hash the current label
		if constexpr(p==1) {
			hash1(bitpos, &next, label);
		} else {
			hash2(bitpos, &next, label);
		}

		ASSERT(next <= 1);

		// calculate depending on the `next` bit either He or He + s
		if (next == 0) {
			f1(label, bitpos);
		} else {
			f2(label, bitpos);
		}
	}

	/// helper function
	/// \tparam label
	/// \param a
	/// \param b
	/// \param limbs
	template<typename label>
	constexpr void xor_helper(label &a, const word *b, const uint64_t limbs) noexcept {
		unsigned int i = 0;

		for (; i < limbs; ++i) {
			a.data().data()[i] ^= b[i];
		}
	}

	/// generate a new randomize flavor
	constexpr void flavour() noexcept {
		r = fastrandombytes_uint64() & ((1 << (l-1)) -1);
		r <<=1 ;
	}

	/// \param bitpos1
	/// \param bitpos2
	/// \return
	constexpr bool check_collision(const uint32_t bitpos1[p], const uint32_t bitpos2[p]) noexcept {
		LabelOutContainerType label = wSyndrom;

		//
		for (int i = 0; i < p; ++i) {
			xor_helper(label, HT->rows[bitpos1[i]], LabelOutContainerType::limbs());
			xor_helper(label, HT->rows[bitpos2[i]], LabelOutContainerType::limbs());
		}

		const uint32_t weight = label.weight();

		//
		if(weight <= (w-2*p)) {
			not_found = false;

			std::cout << label << " label: weight: " << weight << "\n";
			std::cout << "rows: " << bitpos1[0] << " " << bitpos2[0];
			if constexpr(p == 2) {
				std::cout << " " << bitpos1[1]  << " " <<  bitpos2[1];
			}
			std::cout <<  "\n";
			//mzd_print(wH);

			for (int j = 0; j < n - k; ++j) {
				uint32_t bit = label[j];
				mzd_write_bit(e, 0, permutation->values[j], bit);
			}

			for (int i = 0; i < p; ++i) {
				mzd_write_bit(e, 0, permutation->values[n-k+bitpos1[i]], 1);
				mzd_write_bit(e, 0, permutation->values[n-k+bitpos2[i]], 1);
			}
			mzd_print(e);
			return true;
		}

		return false;
	}

	// execute the program and return the number of loops needed to find the solution
	uint64_t run() noexcept {
		uint64_t loops = 0, tsteps = 0;
		not_found = true;

		while (not_found) {
			matrix_create_random_permutation(wH, wHT, permutation);
			matrix_echelonize_partial_plusfix(wH, config.m4ri_k, n - k, matrix_data, 0, n - k, 0,
			                                  permutation);
			mzd_submatrix(H,  wH, 0, n - k, n - k, n);
			matrix_transpose(HT, H);
			mzd_submatrix(iHT, HT, 0, n - k - l, k, n-k);

			if constexpr(!config.DOOM) {
				wSyndrom.column_from_m4ri(wH, n - c);
				iSyndrom = extract_l_bit(wSyndrom.data().data());
			}
//#pragma omp parallel default(none) shared(cout, H, HT, wSyndrom, not_found, loops, steps, functions) num_threads(threads)
			{
				uint64_t ctr1= 0, ctr2= 0;
				uint64_t steps = 0;
				ValueContainerType e1, e2;
				LabelContainerType l1, l2, sp;
				uint32_t bitsset1[p] = {0}, bitsset2[p] = {0};

				while (not_found && (steps < config.nr_steps)) {
					flavour();
					sp = fastrandombytes_uint64();
					LabelContainerType slow_tmp, fast_tmp;
					l1 = sp; l2 = sp;

					if constexpr(!config.UseBrent) {
						step(l1, bitsset1);
						step(l2, bitsset2);
						step(l2, bitsset2);

						while (l1 != l2) {
							step(l1, bitsset1);
							step(l2, bitsset2);
							step(l2, bitsset2);
						}
						uint64_t ctr = 0;

						l1 = sp;
						while (l1 != l2) {
							ctr += 1;
							slow_tmp = l1;
							fast_tmp = l2;

							step(l1, bitsset1);
							step(l2, bitsset2);
						}
					} else {
						// l1 = slow, l2 = fast
						step(l2, bitsset2);
						uint64_t  lam = 1, power = 1;
						while(l1 != l2) {
							// std::cout << bitsset2[0] << " " << bitsset2[1] << "\n";
							if (lam == power) {
								l1 = l2;
								power <<= 1;
								lam = 0;
							}

							step(l2, bitsset2);
							lam += 1;
						}

						l1 = sp; l2 = sp;
						for (uint64_t i = 0; i < lam; ++i) {
							step(l2, bitsset2);
						}

						if (sp == l2) {
							// restart with a new randomly chosen element.
							std::cout << "restart\n";
							break;
						}

						while (l1 != l2) {
							slow_tmp = l1;
							fast_tmp = l2;

							step(l1, bitsset1);
							step(l2, bitsset2);
						}
					}

					if ((slow_tmp&1) != (fast_tmp&1)) {
//					print_binary(slow_tmp, "label 1");
//					print_binary(fast_tmp, "label 2");
//					print_binary(l1, "coll label 1");
//					print_binary(l2, "coll label 2");
//					print_binary(iSyndrom, " iSyndrom");
//					std::cout << ctr << " counter\n";
//					std::cout << slow_tmp << " label 1\n";
//					std::cout << fast_tmp << " label 2\n";
//					std::cout << l1 << " coll label 1\n";
//					std::cout << l2 << " coll label 2\n";
//					std::cout << iSyndrom << " iSyndrom\n";
//					std::cout << wSyndrom << " wSyndrom\n";
//					std::cout << "rows: " << bitsset1[0] << " " << bitsset1[1] << " " << bitsset2[0] << " "  << bitsset2[1] << "\n";
						//std::cout << "rows: " << bitsset1_tmp[0] << " " << bitsset1_tmp[1] << " " << bitsset2_tmp[0] << " "  << bitsset2_tmp[1] << "\n";
						//std::cout << "rows: " << bitsset1[0] << " " << bitsset2[0] << "\n";
						//ctr1 += 1;
						check_collision(bitsset1, bitsset2);
					} else {
						//ctr2 += 1;
					}

					steps += 1;
				}

				tsteps += steps;
				//std::cout << double(ctr1) / double(ctr1 + ctr2) << "\n" << std::flush;
				//std::cout << "restart\n";
			}
			loops += 1;
		}

		std::cout << "loops: " << loops << " "<< tsteps/loops << "\n";
		return loops;
	}
};

#endif //SMALLSECRETLWE_POLLARD_H
