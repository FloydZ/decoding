#ifndef DECODING_SIEVING_H
#define DECODING_SIEVING_H

#include <cstdint>
#include <algorithm>

#include "isd.h"
#include "algorithm/bits/popcount.h"
#include "alloc/alloc.h"
#include "math/bc.h"

using namespace cryptanalysislib;

/// binary version of `FqSieving`
struct ConfigSieving : public ConfigISD {
public:
	// number of sieving steps, each one matches `l/sieving_steps` bits
	const uint32_t sieving_steps = 3;

	// max number of elements per list
	const uint32_t list_size = 1u << 16;

	// returns the expected number of iterations, if all collisions are found
	[[nodiscard]] constexpr uint64_t compute_loops() const noexcept {
#ifdef EXPECTED_PERMUTATIONS
		return EXPECTED_PERMUTATIONS;
#else
		return bc(n, w)/(bc(n-k-l, w-2*p) * bc(k+l, 2*p));
#endif
	}

	void print() const noexcept {
		ConfigISD::print();
		std::cout << "{ \"sieving_steps\": " << sieving_steps
				  << ", \"list_size\": " << list_size
				  << " }" << std::endl;
	}
};

/// Sieving over F2 on the `l` bit window:
/// the base list contains all weight `p` vectors `x` of the `k+l` coordinates,
/// once with label `lHT*x` and once with `lHT*x + s` (`c = 1`). Every sieving
/// step sums pairs which are equal on the next `l/sieving_steps` bits, and keeps
/// the sums of weight `2p`. In the last step the sums with `c = 1` are zero on
/// all `l` bits, hence `H*x = s` on the window and the full weight is checked.
/// Usage:
///		static constexpr ConfigSieving config{isdConfig, 3, 1u << 16};
///		Sieving<isdConfig, config> sieve{};
///		sieve.from_string(h, s);
///		sieve.run();
template<const ConfigISD &isd, const ConfigSieving &config>
class Sieving : public ISDInstance<uint64_t, isd> {
public:
	// copy instance parameters from configuration
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t k = config.k;
	constexpr static uint32_t w = config.w;
	constexpr static uint32_t p = config.p;
	constexpr static uint32_t l = config.l;
	constexpr static uint32_t kl = k + l;
	constexpr static uint32_t sieving_steps = config.sieving_steps;
	constexpr static uint32_t l1 = l / sieving_steps; // bits matched per sieving step

	static_assert(config.q == 2, "use `FqSieving`");
	static_assert(p > 0);
	static_assert(w >= 2*p);
	static_assert((sieving_steps > 0) && (l % sieving_steps == 0));
	static_assert((l > 0) && (l < 32));
	static_assert(kl < (1u << 16));

	using ISD			= ISDInstance<uint64_t, isd>;
	using Error 		= typename ISD::Error;
	using Label 		= typename ISD::Label;
	using limb_type 	= typename ISD::limb_type;
	using l_type 		= typename ISD::l_type;
	using ISD::HT,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::expected_loops;
	using ISD::cycles,ISD::lHT;

	constexpr static uint32_t NR_HT_LIMBS = (n-k+sizeof(limb_type)*8 - 1u)/((sizeof(limb_type)*8));
	constexpr static l_type lmask = (l_type(1) << l) - 1u;
	constexpr static l_type l1mask = (l_type(1) << l1) - 1u;

	// the base list contains every weight `p` vector twice
	constexpr static size_t base_size = 2*bc(kl, p);
	constexpr static size_t list_size = config.list_size;
	static_assert(base_size <= list_size, "increase `list_size`");

	/// `pos` are the sorted positions of the ones in the `k+l` coordinates:
	/// `p` in the base list, `2p` after the first sieving step.
	/// `c` == 1 if the syndrome is added to `label`
	struct Element {
		l_type label;
		uint8_t c;
		uint16_t pos[2*p];
	};

	/// current and next list
	Element *L1, *L2;
	size_t size1 = 0;

	/// rows of `HT` of the solution
	uint16_t solutions[2*p] = {0};

	Sieving() noexcept {
		expected_loops = config.compute_loops();
		constexpr size_t bytes = roundToAligned<64>(sizeof(Element) * list_size);
		L1 = (Element *)cryptanalysislib::aligned_alloc(64, bytes);
		L2 = (Element *)cryptanalysislib::aligned_alloc(64, bytes);
		assert(L1 && L2);
	}

	~Sieving() noexcept {
		cryptanalysislib::aligned_free(L1);
		cryptanalysislib::aligned_free(L2);
	}

	/// enumerates all weight `p` vectors into `L1`
	void init_list() noexcept {
		uint16_t idx[p];
		for (uint32_t i = 0; i < p; i++) {
			idx[i] = i;
		}

		size1 = 0;
		while (true) {
			Element &a = L1[size1++];
			a.label = 0;
			a.c = 0;
			for (uint32_t i = 0; i < p; i++) {
				a.pos[i] = idx[i];
				a.label ^= lHT[idx[i]];
			}

			// the same vector plus the syndrome
			Element &b = L1[size1++];
			b = a;
			b.label ^= l_type(syndrome);
			b.c = 1;

			// next combination
			int32_t i = p - 1;
			while ((i >= 0) && (idx[i] == kl - p + i)) {
				i -= 1;
			}
			if (i < 0) {
				break;
			}

			idx[i] += 1;
			for (uint32_t j = i + 1; j < p; j++) {
				idx[j] = idx[j - 1] + 1;
			}
		}

		assert(size1 == base_size);
	}

	/// checks the full weight of the sum of the rows `pos` of `HT` and the syndrome
	/// \return true if it is the solution
	bool check(const uint16_t pos[2*p]) noexcept {
		uint32_t wt = 0;
		for (uint32_t i = 0; i < NR_HT_LIMBS; i++) {
			limb_type climb = ws.ptr(i);
			for (uint32_t j = 0; j < 2*p; j++) {
				climb ^= HT[pos[j]][i];
			}

			// the first `l` coordinates are zero, because of the sieving
			assert((i != 0) || ((climb & lmask) == 0));
			wt += cryptanalysislib::popcount::popcount(climb);
			if (wt > w - 2*p) {
				return false;
			}
		}

		not_found = false;
		cycles = cpucycles() - cycles;
		for (uint32_t j = 0; j < 2*p; j++) {
			solutions[j] = pos[j];
		}
		return true;
	}

	/// sums all pairs of `L1` which are equal on the bits
	/// [level*l1, (level+1)*l1) and have weight `2p`, into `L2`.
	/// The last level checks the sums instead.
	/// \return true if the solution was found
	template<const uint32_t level>
	bool sieving_step() noexcept {
		constexpr uint32_t shift = level * l1;
		constexpr bool last = level == (sieving_steps - 1u);
		constexpr uint32_t np = (level == 0) ? p : 2*p; // positions per input element

		auto key = [](const Element &a) {
			return (a.label >> shift) & l1mask;
		};
		std::sort(L1, L1 + size1, [&key](const Element &a, const Element &b) {
			return key(a) < key(b);
		});

		size_t size2 = 0;
		size_t s = 0;
		while ((s < size1) && (size2 < list_size)) {
			// bucket [s, t) has the same key
			size_t t = s + 1;
			while ((t < size1) && (key(L1[t]) == key(L1[s]))) {
				t += 1;
			}

			for (size_t i = s; (i < t) && (size2 < list_size); i++) {
				const Element &a = L1[i];
				for (size_t j = i + 1; (j < t) && (size2 < list_size); j++) {
					const Element &b = L1[j];
					// x+s + y+s is the same as x + y
					if ((level == 0) && a.c && b.c) {
						continue;
					}

					// symmetric difference of the sorted positions
					uint16_t pos[2*np];
					uint32_t cnt = 0, ia = 0, ib = 0;
					while ((ia < np) && (ib < np)) {
						if (a.pos[ia] == b.pos[ib]) { ia++; ib++; }
						else if (a.pos[ia] < b.pos[ib]) { pos[cnt++] = a.pos[ia++]; }
						else { pos[cnt++] = b.pos[ib++]; }
					}
					while (ia < np) { pos[cnt++] = a.pos[ia++]; }
					while (ib < np) { pos[cnt++] = b.pos[ib++]; }
					if (cnt != 2*p) {
						continue;
					}

					const uint8_t c = a.c ^ b.c;
					const l_type label = a.label ^ b.label;
					assert(((label >> shift) & l1mask) == 0);

					if constexpr (last) {
						assert((label & lmask) == 0);
						if (c && check(pos)) {
							return true;
						}
					} else {
						Element &d = L2[size2++];
						d.label = label;
						d.c = c;
						for (uint32_t m = 0; m < 2*p; m++) {
							d.pos[m] = pos[m];
						}
					}
				}
			}

			s = t;
		}

		if constexpr (!last) {
			// remove duplicates: {a,b} + {b,c} = {a,c} for every b
			auto less = [](const Element &a, const Element &b) {
				if (a.label != b.label) { return a.label < b.label; }
				if (a.c != b.c) { return a.c < b.c; }
				return std::lexicographical_compare(a.pos, a.pos + 2*p, b.pos, b.pos + 2*p);
			};
			auto equal = [](const Element &a, const Element &b) {
				return (a.label == b.label) && (a.c == b.c) && std::equal(a.pos, a.pos + 2*p, b.pos);
			};
			std::sort(L2, L2 + size2, less);
			size2 = std::unique(L2, L2 + size2, equal) - L2;

			std::swap(L1, L2);
			size1 = size2;
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

		for (uint32_t i = 0; i < 2*p; ++i) {
			assert(n-k-l + solutions[i] < n);
			tmpe.set(1, 0, n-k-l + solutions[i]);
		}

		// apply back permutation.
		for (uint32_t i = 0; i < n; ++i) {
			e.set(tmpe.get(0, i), 0, P.values[i]);
		}
	}

	// executes the algorithm and returns the number of loops/iterations needed
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();

		while (not_found && loops < isd.loops) {
			ISD::step();
			ISD::template extract_lHT<l_type>(lHT);
			init_list();

			constexpr_for<0u, sieving_steps, 1u>([this](auto i) {
				if (this->not_found && (this->size1 > 0)) {
					this->template sieving_step<i>();
				}
			});
		}

		if (!not_found) {
			reconstruct();
		}
		return loops;
	}

	/// important. Dont rename it
	void info() const noexcept {
		std::cout << "{ \"sieving_steps\": " << sieving_steps
		          << ", \"l1\": " << l1
		          << ", \"base_size\": " << base_size
		          << ", \"list_size\": " << list_size
		          << " }" << std::endl;
	}
};

#endif//DECODING_SIEVING_H
