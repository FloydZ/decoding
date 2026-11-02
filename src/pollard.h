#ifndef DECODING_POLLARD_H
#define DECODING_POLLARD_H

#include "helper.h"
#include "isd.h"
#include "list/list.h"
#include "container/binary_packed_vector.h"

#include <type_traits>
#include <algorithm>
#include <omp.h>

/// configuration structure of the pollard=memory-less Stern algorithm
///	Usage:
///		static constexpr ConfigPollard config(n, k, w, 1, 8, 1 << 7);
struct ConfigPollard : public ConfigISD{
public:
	const uint32_t nr_steps;

	// Instead of using floyd`s cyclic finding algorithm
	// use Brents algorithm
	const bool UseBrent     = true;
	const bool DOOM         = false;    // NOT IMPLEMENTED

	// Instead of using a highly optimised hash function
	// use a perfect isomorphism, which is quite slow.
	const bool isomorhism   = true;

	// print every `X` loops some status update about
	// the program.
	const uint32_t print_loops = 100000;
	const uint64_t loops = 100;

	constexpr ConfigPollard(const uint32_t nr_steps,    // number of repetitions
							const bool bijections=true,
	                        const bool UseBrent=true) :
			nr_steps(nr_steps),
	        UseBrent(UseBrent), isomorhism(bijections) {
	            assert(!DOOM);
	        };

	// prints information about the problem instance.
	void print() const {
		std::cout << "{ \"nr_steps\": " << nr_steps
		          << ", \"bijection\": " << isomorhism
		          << ", \"UseBrent\": " << UseBrent
		          << " }" << std::endl;
	}
};

///	Pollard/Memoryless Lee-Brickell
/// Usage:
///		static constexpr ConfigPollard config(G_n, G_k, G_w, 1, 8, 1 << 7);
///		Pollard<config> pollard(ee, ss, A);
///		pollard.run();
/// \tparam config constexpr initialisation of `ConfigPollard`
template<const ConfigPollard &config>
class Pollard {
public:
	// constants from the problem instance
	constexpr static uint32_t n     = config.n;             // code length
	constexpr static uint32_t k     = config.k;             // code dimension
	constexpr static uint32_t w     = config.w;             // weight
	constexpr static uint32_t p     = config.p;				// base weight enumerated in the collision functions
	constexpr static uint32_t l     = config.l;             // number of bits to match in the collision function
	constexpr static uint32_t lprime= (l-1)/2;              // TODO explain
	constexpr static uint32_t nkl   = n-k-l;                // little helper to simplify some formulars.

	constexpr static uint32_t threads       = config.threads;				// number of threads
	constexpr static uint32_t lVCLTBits     = sizeof(uint64_t)*8;           // bits of the label
	constexpr static uint32_t loffset       = nkl / lVCLTBits;              // limb offset of the `l`part
	constexpr static uint32_t lshift        = nkl - (loffset * lVCLTBits);  // number of bits to shift down the limb
																			// containing the `l` part, s.t. the first bit is on bit zero.

	// Main datatypes
	using DecodingValue     = FqPackedVector<k, 2>;			// Datatype holding `e_2`
	using DecodingLabelOut  = FqPackedVector<n-k, 2>;		// Datatype holding `s`
	using DecodingLabel     = FqPackedVector<l, 2>;			// Datatype holding `s_2`

	using DecodingMatrix    = mzd_t *;                                                  // Datatype of `H`

	typedef typename DecodingValue::ContainerType       ValueContainerType;     // `uint64_t`
	typedef typename DecodingValue::ContainerLimbType   ValueContainerLimbType; // `uint64_t`
	typedef typename DecodingLabelOut::ContainerType    LabelOutContainerType;  // `uint64_t`

	// This is the minimal data type needed to represent `l` bit. So
	// it will be `uint16_t` if l <= 16 or `uint32_t` if l <= 32
	using LabelContainerType     = LogTypeTemplate<l>;

	// Datatype capable of holding bit positions of a `DecodingValue`,
	// so its the min data type capable to hold log2(k) bits.
	using BitPosType             = uint16_t;

	// These two data types hold `s_2`. The first one holds the syndrome from
	// bit zero to bit zero `l`, e.g. the syndrome is shifted down.
	// Whereas `wSyndrome` holds the syndrome extracted from the matrix.
	// In other word:
	//                  iSyndrome = cast_to_minimal_data_type(wSyndrome >> lshift);
	LabelContainerType iSyndrom;    // most likely a `uint32_t` or even smaller
	LabelOutContainerType wSyndrom;

	// flavor: randomize each trail of the algorithm.
	ValueContainerLimbType r;

	// M4Ri matrix structures
	mzd_t   *wH,    // working matrix. Holds the permuted n-k \times n Matrix
	        *wHT,   // wH transposed
	        *sT,    // current syndrome in row form, e.g. transposed.
	        *H,     // n-k \time k+l submatrix the collision functions are working on
	        *HT,    // H transposed
	        *iHT;   // TODO explain

	// current permutation applied to `wH`
	mzp_t *permutation;

	// Some internal helper structure for a faster gaussian elimination.
	customMatrixData *matrix_data;

	// Instance parameters
	mzd_t *e;
	const mzd_t *s;
	const mzd_t *A;

	// internal stuff
	uint32_t ext_tid = 0;
	bool not_found = true;

	// helper array. Position [x][y] corresponds to binom(x, y)
	uint64_t **binom;

	// helping variables, for a little debugging.
	DEBUG_MACRO(uint64_t ctr1=0;)
	DEBUG_MACRO(uint64_t ctr2=0;)

	/// Constructor
	/// \param e output error
	/// \param s input syndrome
	/// \param A input matrix
	/// \param ext_tid external thread id. Used to init the internal prng
	Pollard(mzd_t *e, const mzd_t *const s, const mzd_t *const A,
			const uint32_t ext_tid = 0)
			: e(e), s(s), A(A), ext_tid(ext_tid) {
		// safety measurements.
		static_assert(n > k, "wrong dimension");
		static_assert(p <= 2, "sorry needs to be implemented.");
		static_assert(l < 64);

		// reset the found indicator variable.
		not_found = true;

		// Seed the internal prng.
		srand(ext_tid + time(NULL));
		random_seed(ext_tid + rand() * time(NULL));

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
		permutation = mzp_init(n);

		// Check if the working matrices are all allocated
		if ((wH == NULL) || (wHT == NULL) || (sT == NULL) ||
		    (H == NULL) || (HT == NULL) || (permutation == NULL) || (matrix_data == NULL)) {
			std::cout << "ExtTID: " << ext_tid << ", alloc error\n";
		}

		if constexpr (config.isomorhism) {
			// create a lookup table for binomial coefficients for every (i, j) in [k, p]
			binom = new uint64_t*[k+1];
			for (uint32_t i = 0; i < k+1; ++i) {
				binom[i] = new uint64_t[p+1];
				for (uint32_t j = 0; j < p+1; ++j) {
					binom[i][j] = bc(i, j);
				}
			}
		}
	}

	~Pollard() {
		// clear up the matrix stuff
		mzd_free(wH);
		mzd_free(wHT);
		mzd_free(sT);
		mzd_free(H);
		mzd_free(HT);
		mzd_free(iHT);
		mzp_free(permutation);
		free_matrix_data(matrix_data);

		// clear up the helper data for the isomorhism.
		if constexpr (config.isomorhism) {
			for (uint32_t i = 0; i < k+1; ++i) {
				delete binom[i];
			}
			delete[] binom;
		}
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

	/// extracts the bits [n-k-l, ..., n-k] from a given label.
	/// \param v3 pointer to the label data to extract the l bits from
	/// \return	l bits shifted down to bit position 0
	constexpr inline LabelContainerType extract_l_bit(ValueContainerLimbType *v3) noexcept {
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

	/// sample a random starting point with weight p and writes the positions of the
	/// randomly chosen bits of `e1` into `bitpos`.
	/// \param e1 		output: randomly chosen value with p bits set.
	/// \param bitpos 	output: posisitons of those errors
	constexpr static inline void start_point(ValueContainerType &e1, uint16_t bitpos[p]) noexcept {
		e1.random_with_weight(p);
		e1.get_bits_set(bitpos, p);
	}

	///  perfect hash function, maps l bits to an index of the list containing binom(k, p) elements, which are
	///		in a normal Lee Brickell implementation the labels one would test.
	///	Additionally, a `next_bit` flag is needed if one add the syndrome or not.
	///	Works with every weight p
	/// \param bitpos		output: Entries in the non-existent list, there indices if columns in the matrix to add up.
	///								Or in other words these are the bit positions of `ones` in the corresponding `Value`
	/// \param next_bit		output: Flag specifying to add the syndrome or not. The first bit of `label` is used for this.
	/// \param label		input: label to hash
	constexpr uint32_t hash_perfect(BitPosType bitpos[p], uint16_t *next_bit, const LabelContainerType label) noexcept {
		uint64_t a = (label >> 1) % binom[k][p]; //bc(k, p);
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

	/// probabilistic version of `hash_perfect`, meaning its not perfect.
	/// See `hash_perfect` for the meaning of the parameters
	/// works only with weight p = 2
	/// \param bitpos	output
	/// \param next_bit	output
	/// \param label	input
	constexpr inline static void hash2(BitPosType bitpos[p], uint16_t *next_bit, const LabelContainerType label) noexcept {
		// only usable if p = 2
		//	0   l/2   l-1 l
		// [     |      | ]
		constexpr LabelContainerType mask1 = (LabelContainerType(1) << (1+lprime)) - 1;
		constexpr uint64_t mod = k;// bc(k, p);
		bitpos[0] = ((label & mask1) >> 1)  % mod;
		bitpos[1] = (label >> (l-lprime))   % mod;
		if (bitpos[0] == bitpos[1])
			bitpos[1] = (bitpos[1] + 1)     % mod;

		*next_bit = label & 1;
	}

	/// same as hash2 but only for weight p = 1
	/// See `hash_perfect` for the meaning of the parameters.
	/// \param bitpos 	output
	/// \param next_bit output
	/// \param label 	input
	constexpr static inline void hash1(BitPosType bitpos[p], uint16_t *next_bit, const LabelContainerType label) noexcept {
		// only usable if p = 1
		//	0          l-1 l
		// [            | ]
		constexpr LabelContainerType mask1 = ((LabelContainerType(1) << l) - 1);
		bitpos[0] = ((label & mask1) >> 1) % k;
		*next_bit = label & 1;
	}

	/// calculates H times the error vector e on l bits.
	/// \param label 	output label He
	/// \param bitpos 	input the bit positions of the bits set in the corresponding value. E.g. nothing else than the columns to sum up.
	constexpr inline void calcHe(LabelContainerType &label, const BitPosType bitpos[p]) noexcept {
		// Some debugging safety measurements.
		ASSERT((bitpos[0]) < iHT->nrows);
		if constexpr(p == 2) { ASSERT((bitpos[1]) < iHT->nrows); }

		for (uint32_t i = 0; i < p; ++i) {
			label ^= iHT->rows[bitpos[i]][0];
		}
	}

	/// UNUSED
	/// function1 only calcs the matrix vector multiplication
	/// \param label	output = He
	/// \param bitpos	bit positions of the error vector
	constexpr inline void f1(LabelContainerType &label, BitPosType bitpos[p]) noexcept {
		label = r;
		calcHe(label, bitpos);
	}

	/// UNUSED
	/// function2 nearly the same as f1, but adds the syndrome additionally
	/// \param label 	output = He + s
	/// \param bitpos 	bit positions of error vector
	constexpr inline void f2(LabelContainerType &label, BitPosType bitpos[p]) noexcept {
		label = iSyndrom;
		label ^= r;

		calcHe(label, bitpos);
	}

	/// calculates one step of the collision search. Meaning its executing
	///		x = H(f1(x)) if some random value set
	///		x = H(f2(x) if some random value is not set
	/// first its maps the label into p bit positions and a random element choosing either
	///		f1 or f2.
	/// Next the new label is calculated from this bit positions
	/// \param label	// input/output:
	/// \param bitpos	// input/output:
	constexpr inline void step(LabelContainerType &label, BitPosType bitpos[p]) noexcept {
		uint16_t next;

		// hash the current label
		if constexpr (config.isomorhism) {
			hash_perfect(bitpos, &next, label);
		} else {
			if constexpr (p == 1) {
				hash1(bitpos, &next, label);
			} else if constexpr (p == 2) {
				hash2(bitpos, &next, label);
			} else if constexpr (p == 3) {
				// TODO hash3(bitpos, &next, label);
			} else {
				ASSERT(0 && "wrong p");
			}
		}
		ASSERT(next <= 1);

		// calculate depending on the `next` bit either He or He + s
		// if (next == 0) { f1(label, bitpos);}
		// else { f2(label, bitpos); }

		// The following code is the same as the if/else clause two line above
		// Note that `label` after the multiplication is either zero or `iSyndrome`
		label = next*iSyndrom;
		label ^= r;

		calcHe(label, bitpos);
	}

	/// helper function. Do not care about it, only for internal use.
	/// Simply calculates a = a ^ b, but a and b are of different types.
	/// \tparam label 	type of `a`
	/// \param a		input/output: label
	/// \param b		input:
	/// \param limbs	number of limbs of `b`
	template<typename label>
	constexpr inline void xor_helper(label &a, const word *b, const uint64_t limbs) noexcept {
		unsigned int i = 0;

		for (; i < limbs; ++i) {
			a.data().data()[i] ^= b[i];
		}
	}

	/// generate a new randomize flavor
	constexpr inline void flavour() noexcept {
		r = fastrandombytes_uint64() & ((1 << (l-1)) -1);
		r <<=1 ;
	}

	constexpr void __attribute__ ((noinline))
	internal_logging(const auto slow_tmp, const auto fast_tmp,
	                 const auto &l1, const auto &l2,
	                 const BitPosType bitpos1[p], const BitPosType bitpos2[p]) {

		print_binary(slow_tmp, "label 1");
		print_binary(fast_tmp, "label 2");
		print_binary(l1, "coll label 1");
		print_binary(l2, "coll label 2");
		print_binary(iSyndrom, " iSyndrom");
		std::cout << "rows: 1:" << bitpos1[0] << ", 1:" << bitpos1[1]
		          << ", 2:" << bitpos2[0] << ", 2:" << bitpos2[1] << "\n";

	}

	constexpr void __attribute__ ((noinline))
	recompute_collision(LabelOutContainerType &label,
	                    const BitPosType bitpos1[p],
	                    const BitPosType bitpos2[p]) noexcept {
		// NOTE: In a multithreading setting, this must be atomic.
		if (!not_found)
			return;

		not_found = false;

		std::cout << label << " label: weight: " << label.weight() << "\n";
		std::cout << "rows: bitpos1" << bitpos1[0] << ", bitpos2" << bitpos2[0];
		if constexpr(p == 2) {
			std::cout << ", bitpos1" << bitpos1[1]  << ", bitpos2" <<  bitpos2[1];
		}
		std::cout <<  "\n";
		//mzd_print(wH);

		for (uint32_t j = 0; j < n - k; ++j) {
			uint32_t bit = label[j];
			mzd_write_bit(e, 0, permutation->values[j], bit);
		}

		for (uint32_t i = 0; i < p; ++i) {
			mzd_write_bit(e, 0, permutation->values[n-k+bitpos1[i]], 1);
			mzd_write_bit(e, 0, permutation->values[n-k+bitpos2[i]], 1);
		}

		mzd_print(e);
	}

	/// given to sets of bit positions, this functions
	/// \param bitpos1 as in the rest of the program, this specifies the bit positions of `ones`
	///					in the left collision function
	/// \param bitpos2 same as `bitpos2` but only of the right function
	constexpr inline void check_collision(const BitPosType bitpos1[p], const BitPosType bitpos2[p]) noexcept {
		LabelOutContainerType label = wSyndrom;

		// some up the in `bitpos1`, `bitpos2` specified columns of the parity check matrix.
		for (uint32_t i = 0; i < p; ++i) {
			xor_helper(label, HT->rows[bitpos1[i]], LabelOutContainerType::limbs());
			xor_helper(label, HT->rows[bitpos2[i]], LabelOutContainerType::limbs());
		}

		// calc the weight
		const uint32_t weight = label.weight();

		// this is split into two functions to reduce instruction cache misses
		if (unlikely(weight <= config.weight_threshhold)) {
			recompute_collision(label, bitpos1, bitpos2);
		}
	}

	// execute the program and return the number of loops needed to find the solution
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		// reset some internal
		uint64_t loops = 0, tsteps = 0;
		not_found = true;

		while (not_found && loops < config.loops) {
			// Start gaussian elimination step.
			matrix_create_random_permutation(wH, wHT, permutation);
			matrix_echelonize_partial_plusfix(wH, config.m4ri_k, n - k, matrix_data, 0, n - k, 0,
			                                  permutation);
			mzd_submatrix(H,  wH, 0, n - k, n - k, n);
			matrix_transpose(HT, H);
			mzd_submatrix(iHT, HT, 0, n - k - l, k, n-k);
			// end gaussian elimination step

			// extract the last column from the working matrix, which is syndrome.
			wSyndrom.column_from_m4ri(wH, n);
			// extract the highest l bits from the syndrome.
			iSyndrom = extract_l_bit(wSyndrom.data().data());

			//#pragma omp parallel default(none) shared(cout, H, HT, wSyndrom, not_found, loops, steps, functions) num_threads(threads)
			{
				uint64_t steps = 0;
				ValueContainerType e1, e2;
				LabelContainerType l1, l2, sp;
				BitPosType bitsset1[p] = {0}, bitsset2[p] = {0};

				while (not_found && (steps < config.nr_steps)) {
					// choose a random flavor
					flavour();

					// choose a random starting point
					sp = fastrandombytes_uint64();
					LabelContainerType slow_tmp, fast_tmp;
					l1 = sp; l2 = sp;

					if constexpr(!config.UseBrent) {
						// run floyds circle finding algorithm
						// l1 is the slow kangaroo and l2 the fast.
						do {
							step(l1, bitsset1);
							step(l2, bitsset2);
							step(l2, bitsset2);
						} while (l1 != l2);

						l1 = sp;
						// recompute collision.
						while (l1 != l2) {
							DEBUG_MACRO(ctr1 += 1;)
							slow_tmp = l1;
							fast_tmp = l2;

							step(l1, bitsset1);
							step(l2, bitsset2);
						}
					} else {
						// run brents circle finding algorithm
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

						if (unlikely(sp == l2)) {
							// restart with a new randomly chosen element.
							//DEBUG_MACRO(std::cout << "restart\n";)
							break;
						}

						// recompute the collision
						while (l1 != l2) {
							slow_tmp = l1;
							fast_tmp = l2;

							step(l1, bitsset1);
							step(l2, bitsset2);
						}
					}

					// extract the lowest bit to make sure we have the collision
					// in the two different function.
					if ((slow_tmp&1) != (fast_tmp&1)) {
						DEBUG_MACRO(ctr1 += 1;)
						//DEBUG_MACRO(internal_logging(slow_tmp, fast_tmp, l1, l2, bitsset1, bitsset2);)
						check_collision(bitsset1, bitsset2);
					} else {
						// increment an debug counter to check we have around a 50% chance
						// to have a valid collision.
						DEBUG_MACRO(ctr2 += 1;)
					}

					steps += 1;
				}

				tsteps += steps;    // increment total counter
				// DEBUG_MACRO(std::cout << double(ctr1) / double(ctr1 + ctr2) << "\n" << std::flush;)
			}
			loops += 1;
		}

		std::cout << "loops: " << loops << " "<< tsteps/loops << "\n";
		return loops;
	}
};

#endif //SMALLSECRETLWE_POLLARD_H
