#ifndef DECODING_POLLARD_H
#define DECODING_POLLARD_H

#include <cstdint>

#include "isd.h"
#include "algorithm/bits/popcount.h"
#include "math/bc.h"
#include "random.h"

using namespace cryptanalysislib;

/// configuration structure of the pollard=memory-less Stern algorithm
///	Usage:
///		static constexpr ConfigPollard config{isdConfig, 1u << 8, true, true};
struct ConfigPollard : public ConfigISD {
public:
	// number of collisions searched per permutation
	const uint32_t nr_steps = 1u << 8;

	// Instead of using a highly optimised hash function
	// use a perfect isomorphism, which is quite slow.
	const bool isomorhism = true;

	// Instead of using floyd`s cyclic finding algorithm
	// use Brents algorithm
	const bool UseBrent = true;

	// `cuPCS` (van Oorschot-Wiener): walks end at points with `D` zero low bits,
	// the table has 2^log_table entries, `versions` flavours are tried per permutation
	const uint32_t D = 1;
	const uint32_t log_table = 16;
	const uint32_t versions = 1;

	// returns the expected number of iterations
	[[nodiscard]] constexpr uint64_t compute_loops() const noexcept {
#ifdef EXPECTED_PERMUTATIONS
		return EXPECTED_PERMUTATIONS;
#else
		return bc(n, w)/(bc(n-k-l, w-2*p) * bc(k+l, 2*p));
#endif
	}

	// prints information about the problem instance.
	void print() const noexcept {
		ConfigISD::print();
		std::cout << "{ \"nr_steps\": " << nr_steps
		          << ", \"bijection\": " << isomorhism
		          << ", \"UseBrent\": " << UseBrent
		          << " }" << std::endl;
	}
};

///	Pollard/Memoryless Stern
/// Usage:
///		Pollard<isdConfig, config> pollard{};
///		pollard.from_string(h, s);
///		pollard.run();
/// \tparam config constexpr initialisation of `ConfigPollard`
template<const ConfigISD &isd, const ConfigPollard &config>
class Pollard : public ISDInstance<uint64_t, isd> {
public:
	// constants from the problem instance
	constexpr static uint32_t n     = config.n;             // code length
	constexpr static uint32_t k     = config.k;             // code dimension
	constexpr static uint32_t w     = config.w;             // weight
	constexpr static uint32_t p     = config.p;				// base weight enumerated in the collision functions
	constexpr static uint32_t l     = config.l;             // number of bits to match in the collision function
	constexpr static uint32_t kl    = k+l;                  // number of columns the collision functions choose from
	constexpr static uint32_t lprime= (l-1)/2;              // TODO explain

	static_assert(n > k, "wrong dimension");
	static_assert((p > 0) && (p <= 2), "sorry needs to be implemented.");
	static_assert((l > 1) && (l < 32));

	using ISD			= ISDInstance<uint64_t, isd>;
	using Error 		= typename ISD::Error;
	using Label 		= typename ISD::Label;
	using limb_type 	= typename ISD::limb_type;
	using l_type 		= typename ISD::l_type;
	using ISD::HT,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::expected_loops;
	using ISD::cycles,ISD::lHT;

	// This is the minimal data type needed to represent `l` bit.
	using LabelContainerType     = l_type;

	// Datatype capable of holding bit positions of a `Value`,
	// so its the min data type capable to hold log2(k+l) bits.
	using BitPosType             = uint16_t;

	constexpr static uint32_t NR_HT_LIMBS = (n-k+sizeof(limb_type)*8 - 1u)/((sizeof(limb_type)*8));
	constexpr static LabelContainerType lmask = (LabelContainerType(1) << l) - 1u;

	// flavor: randomize each trail of the algorithm.
	LabelContainerType r = 0;

	// helper array. Position [x][y] corresponds to binom(x, y)
	uint64_t binom[kl+1][p+1];

	// rows of `HT` of the solution
	BitPosType solutions[2*p] = {0};

	// helping variables, for a little debugging.
	DEBUG_MACRO(uint64_t ctr1=0;)
	DEBUG_MACRO(uint64_t ctr2=0;)

	Pollard() noexcept {
		expected_loops = config.compute_loops();

		if constexpr (config.isomorhism) {
			// create a lookup table for binomial coefficients for every (i, j) in [k+l, p]
			for (uint32_t i = 0; i < kl+1; ++i) {
				for (uint32_t j = 0; j < p+1; ++j) {
					binom[i][j] = bc(i, j);
				}
			}
		}
	}

	/// helper function printing `val` in binary, `str` is printed out additionally in the end.
	/// \param val	value to print
	/// \param str	helper string to identiy your value in a stream of printings
	void print_binary(const LabelContainerType val, const std::string &str) const noexcept {
		LabelContainerType v = val;
		for (uint32_t i = 0; i < l; ++i) {
			std::cout << (v&1);
			v >>= 1;
		}
		std::cout << " " << str;
		std::cout << "\n";
	}

	///  perfect hash function, maps l bits to an index of the list containing binom(k+l, p) elements, which are
	///		in a normal Stern implementation the labels one would test.
	///	Additionally, a `next_bit` flag is needed if one add the syndrome or not.
	///	Works with every weight p
	/// \param bitpos		output: Entries in the non-existent list, there indices if columns in the matrix to add up.
	///								Or in other words these are the bit positions of `ones` in the corresponding `Value`
	/// \param next_bit		output: Flag specifying to add the syndrome or not. The first bit of `label` is used for this.
	/// \param label		input: label to hash
	constexpr void hash_perfect(BitPosType bitpos[p], uint16_t *next_bit, const LabelContainerType label) const noexcept {
		uint64_t a = (label >> 1) % binom[kl][p];
		*next_bit = label & 1;

		uint32_t wn = kl;
		uint32_t wk = p;
		uint32_t set = 0;
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
		constexpr uint64_t mod = kl;
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
		bitpos[0] = ((label & lmask) >> 1) % kl;
		*next_bit = label & 1;
	}

	/// calculates H times the error vector e on l bits.
	/// \param label 	output label He
	/// \param bitpos 	input the bit positions of the bits set in the corresponding value. E.g. nothing else than the columns to sum up.
	constexpr inline void calcHe(LabelContainerType &label, const BitPosType bitpos[p]) const noexcept {
		for (uint32_t i = 0; i < p; ++i) {
			assert(bitpos[i] < kl);
			label ^= lHT[bitpos[i]];
		}
	}

	/// UNUSED
	/// function1 only calcs the matrix vector multiplication
	/// \param label	output = He
	/// \param bitpos	bit positions of the error vector
	constexpr inline void f1(LabelContainerType &label, const BitPosType bitpos[p]) const noexcept {
		label = r;
		calcHe(label, bitpos);
	}

	/// UNUSED
	/// function2 nearly the same as f1, but adds the syndrome additionally
	/// \param label 	output = He + s
	/// \param bitpos 	bit positions of error vector
	constexpr inline void f2(LabelContainerType &label, const BitPosType bitpos[p]) const noexcept {
		label = LabelContainerType(syndrome);
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
	constexpr inline void step(LabelContainerType &label, BitPosType bitpos[p]) const noexcept {
		uint16_t next;

		// hash the current label
		if constexpr (config.isomorhism) {
			hash_perfect(bitpos, &next, label);
		} else {
			if constexpr (p == 1) {
				hash1(bitpos, &next, label);
			} else {
				hash2(bitpos, &next, label);
			}
		}
		assert(next <= 1);

		// calculate depending on the `next` bit either He or He + s
		// if (next == 0) { f1(label, bitpos);}
		// else { f2(label, bitpos); }

		// The following code is the same as the if/else clause two line above
		// Note that `label` after the multiplication is either zero or `syndrome`
		label = next*LabelContainerType(syndrome);
		label ^= r;

		calcHe(label, bitpos);
	}

	/// generate a new randomize flavor
	inline void flavour() noexcept {
		r = rng() & ((1u << (l-1)) - 1u);
		r <<= 1;
	}

	void __attribute__ ((noinline))
	internal_logging(const auto slow_tmp, const auto fast_tmp,
	                 const auto &l1, const auto &l2,
	                 const BitPosType bitpos1[p], const BitPosType bitpos2[p]) const noexcept {

		print_binary(slow_tmp, "label 1");
		print_binary(fast_tmp, "label 2");
		print_binary(l1, "coll label 1");
		print_binary(l2, "coll label 2");
		print_binary(syndrome, " syndrome");
		std::cout << "rows: 1:" << bitpos1[0] << ", 2:" << bitpos2[0];
		if constexpr(p == 2) {
			std::cout << ", 1:" << bitpos1[1] << ", 2:" << bitpos2[1];
		}
		std::cout << "\n";
	}

	/// given to sets of bit positions, this functions checks the full weight
	/// \param bitpos1 as in the rest of the program, this specifies the bit positions of `ones`
	///					in the left collision function
	/// \param bitpos2 same as `bitpos2` but only of the right function
	/// \return true if it is the solution
	bool check_collision(const BitPosType bitpos1[p], const BitPosType bitpos2[p]) noexcept {
		uint32_t wt = 0;
		for (uint32_t i = 0; i < NR_HT_LIMBS; i++) {
			// sum up the in `bitpos1`, `bitpos2` specified columns of the parity check matrix.
			limb_type climb = ws.ptr(i);
			for (uint32_t j = 0; j < p; j++) {
				climb ^= HT[bitpos1[j]][i] ^ HT[bitpos2[j]][i];
			}

			// the first `l` coordinates are zero, because of the collision
			assert((i != 0) || ((climb & lmask) == 0));
			wt += cryptanalysislib::popcount::popcount(climb);
			if (wt > w - 2*p) {
				return false;
			}
		}

		not_found = false;
		cycles = cpucycles() - cycles;
		for (uint32_t j = 0; j < p; j++) {
			solutions[j]     = bitpos1[j];
			solutions[p + j] = bitpos2[j];
		}
		return true;
	}

	/// one permutation: searches `config.nr_steps` collisions
	/// \return true if the solution was found
	bool walk() noexcept {
		uint64_t steps = 0;
		LabelContainerType l1, l2, sp;
		LabelContainerType slow_tmp, fast_tmp;
		BitPosType bitsset1[p] = {0}, bitsset2[p] = {0};

		while (steps < config.nr_steps) {
			steps += 1;

			// choose a random flavor
			flavour();

			// choose a random starting point
			sp = rng() & lmask;
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
			} else {
				// run brents circle finding algorithm
				// l1 = slow, l2 = fast
				step(l2, bitsset2);
				uint64_t  lam = 1, power = 1;
				while(l1 != l2) {
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
			}

			// `sp` is on the cycle: no collision, restart with a new randomly chosen element.
			if (unlikely(l1 == l2)) {
				continue;
			}

			// recompute the collision
			while (l1 != l2) {
				DEBUG_MACRO(ctr1 += 1;)
				slow_tmp = l1;
				fast_tmp = l2;

				step(l1, bitsset1);
				step(l2, bitsset2);
			}

			// extract the lowest bit to make sure we have the collision
			// in the two different function.
			if ((slow_tmp&1) != (fast_tmp&1)) {
				//DEBUG_MACRO(internal_logging(slow_tmp, fast_tmp, l1, l2, bitsset1, bitsset2);)
				if (check_collision(bitsset1, bitsset2)) {
					return true;
				}
			} else {
				// increment an debug counter to check we have around a 50% chance
				// to have a valid collision.
				DEBUG_MACRO(ctr2 += 1;)
			}
		}

		return false;
	}

	/// reconstruct the final solution
	void __attribute__ ((noinline))
	reconstruct() noexcept {
		Error tmpe;
		Label tmp;
		// copy in the syndrome
		tmp = ws;
		for (uint32_t i = 0; i < 2*p; ++i) {
			assert(solutions[i] < kl);
			Label::add(tmp.ptr(), tmp.ptr(), HT.row(solutions[i]));
		}

		for (uint32_t i = 0; i < n-k; ++i) {
			const auto bit = tmp.get(i);
			tmpe.set(bit, 0, n-k-1-i);
		}

		// NOTE: flip, `bitsset1` and `bitsset2` may share positions
		for (uint32_t i = 0; i < 2*p; ++i) {
			const uint32_t pos = n-k-l + solutions[i];
			tmpe.set(tmpe.get(0, pos) ^ 1u, 0, pos);
		}

		// apply back permutation.
		for (uint32_t i = 0; i < n; ++i) {
			e.set(tmpe.get(0, i), 0, P.values[i]);
		}
	}

	// execute the program and return the number of loops needed to find the solution
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();

		while (not_found && loops < isd.loops) {
			ISD::step();
			ISD::template extract_lHT<l_type>(lHT);
			if (walk()) {
				break;
			}
		}

		if (!not_found) {
			reconstruct();
		}
		return loops;
	}

	/// important. Dont rename it
	void info() const noexcept {
		std::cout << "{ \"nr_steps\": " << config.nr_steps
		          << ", \"bijection\": " << config.isomorhism
		          << ", \"UseBrent\": " << config.UseBrent
		          << " }" << std::endl;
	}
};

#endif //DECODING_POLLARD_H
