#ifndef DECODING_PCS_H
#define DECODING_PCS_H

#include <cstdint>
#include <cstring>

#include "isd.h"
#include "algorithm/bits/popcount.h"
#include "alloc/alloc.h"
#include "math/bc.h"
#include "random.h"

using namespace cryptanalysislib;

/// memoryless Dumer/Stern via parallel collision search (van Oorschot-Wiener)
struct ConfigPCS : public ConfigISD {
public:
	// a point is distinguished, if its lower `dp_bits` bits are zero
	const uint32_t dp_bits = 4;

	// number of distinguished points per permutation
	const uint32_t nr_dps = 1u << 12;

	// log2 of the number of entries of the distinguished point table
	const uint32_t table_bits = 12;

	// returns the expected number of iterations, if all collisions are found
	[[nodiscard]] constexpr uint64_t compute_loops() const noexcept {
#ifdef EXPECTED_PERMUTATIONS
		return EXPECTED_PERMUTATIONS;
#else
		return bc(n, w)/( bc(n-k-l, w-2*p) * bc((k+l)/2, p) * bc((k+l)/2, p));
#endif
	}

	void print() const noexcept {
		ConfigISD::print();
		std::cout << "{ \"dp_bits\": " << dp_bits
				  << ", \"nr_dps\": " << nr_dps
				  << ", \"table_bits\": " << table_bits
				  << " }" << std::endl;
	}
};

/// Instead of the two base lists of Stern, a random function `f` on `l` bit
/// values is iterated: `sel(x)` chooses the left or the right half of `HT`,
/// `idx(x)` a weight `p` combination of it and `f(x)` is its `l` bit label
/// (plus the syndrome for the right half). A collision `f(a) = f(b)` with
/// `sel(a) != sel(b)` is a Stern collision. 8 walks run in parallel, every
/// walk ends in a distinguished point, colliding walks are located afterwards.
/// NOTE: `p <= 2`, see `biject_simd`
template<const ConfigISD &isd, const ConfigPCS &config>
class PCS : public ISDInstance<uint64_t, isd> {
public:
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t k = config.k;
	constexpr static uint32_t w = config.w;
	constexpr static uint32_t p = config.p;
	constexpr static uint32_t l = config.l;

	static_assert((p > 0) && (p <= 2), "`biject_simd` supports p <= 2");
	static_assert((l > 0) && (l < 32));
	static_assert(config.dp_bits < l);

	using ISD = ISDInstance<uint64_t, isd>;
	using Error 		= typename ISD::Error;
	using Label 		= typename ISD::Label;
	using limb_type 	= typename ISD::limb_type;
	using l_type 		= typename ISD::l_type;
	using ISD::HT,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::expected_loops;
	using ISD::cycles,ISD::lHT;
	static_assert(sizeof(l_type) == sizeof(uint32_t));

	constexpr static uint32_t kl_half = (k+l)/2;
	constexpr static uint64_t size = bc(kl_half, p);
	constexpr static uint32_t size_bits = ceil_log2(size);
	static_assert(size_bits < 32);

	constexpr static uint32_t NR_HT_LIMBS = (n-k+sizeof(limb_type)*8 - 1u)/((sizeof(limb_type)*8));
	constexpr static uint32_t lmask = (1u << l) - 1u;
	constexpr static uint32_t dp_mask = (1u << config.dp_bits) - 1u;
	// walks longer than this are probably in a cycle, they are discarded
	constexpr static uint32_t max_walk = 20u << config.dp_bits;
	constexpr static size_t table_size = 1ull << config.table_bits;

	/// a distinguished point and the walk to it. `len == 0` is an empty entry
	struct DP {
		uint32_t dp, start, len;
	};
	DP *table;

	/// list entries of the current permutation if a solution was found.
	size_t solutions[2*p] = {0};

	/// randomizes `f` per permutation
	uint32_t flavor = 0;

	PCS() noexcept {
		expected_loops = config.compute_loops();
		table = (DP *)cryptanalysislib::aligned_alloc(64, roundToAligned<64>(sizeof(DP) * table_size));
		assert(table);
	}

	~PCS() noexcept {
		cryptanalysislib::aligned_free(table);
	}

	/// the random function, evaluated on 8 points at once
	/// \param x input points, `l` bits
	/// \param sel[out] 0: left half, 1: right half
	/// \param rows[out] the rows of `HT` (the right ones are `+kl_half`)
	/// \return f(x), `l` bits
	inline uint32x8_t f(const uint32x8_t x, uint32x8_t &sel, uint32x8_t rows[p]) const noexcept {
		const uint32x8_t y = (x ^ uint32x8_t::set1(flavor)) & uint32x8_t::set1(lmask);
		sel = (y * uint32x8_t::set1(0x85EBCA6Bu)) >> 31u;

		// NOTE: maps the upper bits of the hash uniformly into [0, size)
		const uint32x8_t h = y * uint32x8_t::set1(0x9E3779B1u);
		const uint32x8_t idx = ((h >> size_bits) * uint32x8_t::set1(size)) >> (32u - size_bits);
		biject_simd<kl_half, p>(idx, rows);

		const uint32x8_t offset = sel * uint32x8_t::set1(kl_half);
		uint32x8_t label = uint32x8_t::set1(uint32_t(syndrome)) & (uint32x8_t::set1(0u) - sel);

		#pragma unroll
		for (uint32_t j = 0; j < p; j++) {
			rows[j] = rows[j] + offset;
			label = label ^ uint32x8_t::template gather<sizeof(l_type)>(lHT, rows[j]);
		}

		return label;
	}

	/// checks the full weight of the collision of the lanes `left` and `right`
	/// \return true if it is the solution
	bool check(const uint32x8_t rows[p], const uint32_t left, const uint32_t right) noexcept {
		uint32_t wt = 0;
		for (uint32_t i = 0; i < NR_HT_LIMBS; i++) {
			limb_type climb = ws.ptr(i);
			for (uint32_t j = 0; j < p; j++) {
				climb ^= HT[rows[j].v32[left]][i] ^ HT[rows[j].v32[right]][i];
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
			solutions[j]     = rows[j].v32[left];
			solutions[p + j] = rows[j].v32[right];
		}
		return true;
	}

	/// the walks from `a` (length `la`) and `b` (length `lb`) end in the same
	/// distinguished point, finds the collision.
	/// NOTE: lane 0 walks `a`, lane 1 `b`
	/// \return true if the solution was found
	bool locate(uint32_t a, uint32_t la, uint32_t b, uint32_t lb) noexcept {
		uint32x8_t sel, rows[p];
		while (la > lb) { a = f(uint32x8_t::set1(a), sel, rows).v32[0]; la--; }
		while (lb > la) { b = f(uint32x8_t::set1(b), sel, rows).v32[0]; lb--; }

		// `b` starts on the walk of `a`
		if (a == b) {
			return false;
		}

		while (true) {
			uint32x8_t x = uint32x8_t::set1(a);
			x.v32[1] = b;
			const uint32x8_t y = f(x, sel, rows);
			if (y.v32[0] == y.v32[1]) {
				// both in the same half, this is no Stern collision
				if (sel.v32[0] == sel.v32[1]) {
					return false;
				}

				const uint32_t left = sel.v32[0] == 0 ? 0 : 1;
				return check(rows, left, 1u - left);
			}

			a = y.v32[0];
			b = y.v32[1];
		}
	}

	/// one permutation: runs the walks until `config.nr_dps` distinguished
	/// points are found
	/// \return true if the solution was found
	bool walk() noexcept {
		memset(table, 0, sizeof(DP) * table_size);
		flavor = uint32_t(rng()) & lmask;

		alignas(32) uint32x8_t x, start, len = uint32x8_t::set1(0u);
		for (uint32_t c = 0; c < 8; c++) {
			start.v32[c] = x.v32[c] = uint32_t(rng()) & lmask;
		}

		const uint32x8_t one  = uint32x8_t::set1(1u);
		const uint32x8_t zero = uint32x8_t::set1(0u);
		const uint32x8_t dpm  = uint32x8_t::set1(dp_mask);
		const uint32x8_t maxw = uint32x8_t::set1(max_walk);
		uint32x8_t sel, rows[p];

		uint32_t dps = 0;
		while (dps < config.nr_dps) {
			x = f(x, sel, rows);
			len = len + one;

			// lanes which reached a distinguished point or are too long
			uint64_t m = uint64_t((x & dpm) == zero) | uint64_t(len > maxw);
			while (m) {
				const uint32_t c = __builtin_ctzll(m);
				m &= m - 1u;

				if (len.v32[c] <= max_walk) {
					dps += 1;
					DP &d = table[(x.v32[c] >> config.dp_bits) & (table_size - 1u)];
					if ((d.len != 0) && (d.dp == x.v32[c]) && (d.start != start.v32[c])) {
						if (locate(d.start, d.len, start.v32[c], len.v32[c])) {
							return true;
						}
					} else {
						d = DP{x.v32[c], start.v32[c], len.v32[c]};
					}
				}

				// restart the lane
				start.v32[c] = x.v32[c] = uint32_t(rng()) & lmask;
				len.v32[c] = 0;
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
			assert(solutions[i] < k+l);
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
		std::cout << "{ \"size\": " << size
		          << ", \"dp_bits\": " << config.dp_bits
		          << ", \"nr_dps\": " << config.nr_dps
		          << ", \"table_size\": " << table_size
		          << ", \"max_walk\": " << max_walk
		          << " }" << std::endl;
	}
};
#endif//DECODING_PCS_H
