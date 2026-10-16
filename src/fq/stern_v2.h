#ifndef DECODING_FQ_STERNV2_H
#define DECODING_FQ_STERNV2_H

#include <algorithm>
#include <array>
#include <vector>

#include "helper.h"
#include "container/fq_vector.h"
#include "matrix/matrix.h"
#include "list/list.h"
#include "algorithm/bits/popcount.h"
#include "mitm.h"
#include "isd.h"
#include "../stern.h"

using namespace cryptanalysislib;


/// Stern over Z_q: both base lists contain all weight `p` vectors with
/// coefficients in [1, q) on their half of the k+l coordinates.
template<const ConfigISD &isd, const ConfigStern &config>
class FqSternV2 : public ISDInstance<uint64_t, isd> {
private:
	constexpr static uint32_t n = isd.n;
	constexpr static uint32_t k = isd.k;
	constexpr static uint32_t w = isd.w;
	constexpr static uint32_t q = isd.q;
	constexpr static uint32_t qbits = ceil_log2(q);

	constexpr static uint32_t l = isd.l;
	constexpr static uint32_t p = isd.p;

	constexpr static size_t final_list_size = 1024;

	static_assert(p > 0);

public:
	using ISD = ISDInstance<uint64_t, isd>;
	using Error 		= ISD::Error;
	using Label 		= ISD::Label;
	using Value 		= ISD::Value;
	using limb_type 	= ISD::limb_type;
	using ISD::HT,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::cycles,ISD::lHT;

	constexpr static uint32_t HM_nrb = ISD::HashMasp_BucketSize(l);
	constexpr static uint32_t HM_bs = config.HM_bucketsize;

	/// minimal datatype to keep all l fq elements
	using l_type 	= MinLogTypeTemplate<qbits*l, 32>;
	using IndexType = TypeTemplate<HM_nrb * HM_bs>;

	constexpr static uint32_t NR_HT_LIMBS = Label::internal_limbs;
	constexpr static uint32_t bits_per_number = Label::bits_per_number;
	static_assert(l*bits_per_number <= sizeof(limb_type)*8u);

	/// base lists: left [0, enum_length), right [enum_offset, k+l)
	constexpr static uint32_t enum_length = (k + l + isd.epsilon) >> 1u;
	constexpr static uint32_t enum_offset = k + l - enum_length;
	constexpr static size_t nr_supports = bc(enum_length, p);
	constexpr static size_t nr_coefficients = compute_gray_size<p, q>();
	constexpr static size_t enum_size = nr_supports * nr_coefficients;
	static_assert(enum_size <= (1ull << 32u));

	using V1 = CollisionType<l_type, uint32_t, 1>;
	constexpr static SimpleHashMapConfig simpleHashMapConfig{
			HM_bs, HM_nrb, config.threads
	};
	using HM = SimpleHashMap<l_type, V1, simpleHashMapConfig, Hash<l_type, 0, l, q>>;
	HM *hm;
	using HM_LoadType = HM::load_type;

	/// change list of the supports
	chase_t<enum_length, p> c{};
	using cle = std::pair<uint16_t, uint16_t>;
	std::vector<cle> cL;

	/// list indices of the collisions on the l window
	size_t cfls = 0;
	alignas(256) std::array<uint32_t, final_list_size> final_list_left;
	alignas(256) std::array<uint32_t, final_list_size> final_list_right;

	Value value_solution_to_recover;
	Label label_solution_to_recover;

	/// base constructor
	FqSternV2() noexcept {
		hm = new HM{};
		assert(hm);

		cL.resize(nr_supports);
		size_t ctr = 0;
		c.enumerate([&, this](const uint16_t p1, const uint16_t p2){
			cL[ctr] = cle{p1, p2};
			ctr += 1;
		});
		assert(ctr == (nr_supports - 1));
	}

	/// free all the memory
	~FqSternV2() {
		delete hm;
	}

	/// calls `f(label, index)` for every element of a base list
	/// \param offset first coordinate of the list
	template<typename F>
	void enumerate(const uint32_t offset, F &&f) noexcept {
		uint16_t rows[p];
		for (uint32_t j = 0; j < p; ++j) { rows[j] = j; }

		for (size_t i = 0; i < nr_supports; ++i) {
			std::sort(rows, rows + p);
			uint32_t coeff[p];
			l_type label = 0;
			for (uint32_t j = 0; j < p; ++j) {
				coeff[j] = 1;
				label = Label::template add_T<l_type>(label, lHT[offset + rows[j]]);
			}

			for (size_t t = 0; t < nr_coefficients; ++t) {
				f(label, i*nr_coefficients + t);

				// next coefficients (odometer)
				for (uint32_t j = 0; j < p; ++j) {
					const l_type h = lHT[offset + rows[j]];
					if (coeff[j] < (q - 1u)) {
						coeff[j] += 1;
						label = Label::template add_T<l_type>(label, h);
						break;
					}

					coeff[j] = 1;
					label = Label::template sub_T<l_type>(label, Label::template scalar_T<l_type>(h, q - 2u));
				}
			}

			// next support: exactly one of the two positions is in it
			if (i + 1 < nr_supports) {
				const uint16_t a = cL[i].first, b = cL[i].second;
				for (uint32_t j = 0; j < p; ++j) {
					if (rows[j] == a) { rows[j] = b; break; }
					if (rows[j] == b) { rows[j] = a; break; }
				}
			}
		}
	}

	/// inverse of the list index of `enumerate`
	/// \param rows[out] sorted positions, relative to the list
	/// \param coeff[out] coefficients
	void decode(const size_t index, uint16_t *rows, uint32_t *coeff) noexcept {
		const size_t support = index / nr_coefficients;
		if (support == 0) {
			for (uint32_t j = 0; j < p; ++j) { rows[j] = j; }
		} else {
			c.biject(support, rows);
		}
		std::sort(rows, rows + p);

		size_t t = index % nr_coefficients;
		for (uint32_t j = 0; j < p; ++j) {
			coeff[j] = 1u + (t % (q - 1u));
			t /= (q - 1u);
		}
	}

	/// checks the collisions on the full syndrome
	bool compute_finale_list() noexcept {
		constexpr limb_type lmask = (l*bits_per_number == sizeof(limb_type)*8u) ?
		        limb_type(-1) : limb_type((limb_type(1) << (l*bits_per_number)) - 1u);
		uint16_t rows[2*p];
		uint32_t coeff[2*p];
		limb_type label[NR_HT_LIMBS];

		for (size_t m = 0; (m < cfls) && not_found; ++m) {
			decode(final_list_left[m], rows, coeff);
			decode(final_list_right[m], rows + p, coeff + p);
			for (uint32_t j = 0; j < p; ++j) { rows[p + j] += enum_offset; }

			// e1 = s - Q*e2
			bool zero_window = true;
			uint32_t wt = 0;
			for (uint32_t i = 0; i < NR_HT_LIMBS; ++i) {
				limb_type climb = ws.ptr(i);
				for (uint32_t j = 0; j < 2*p; ++j) {
					const limb_type t = Label::template scalar_T<limb_type>(HT.limb(rows[j], i), coeff[j]);
					climb = Label::template sub_T<limb_type>(climb, t);
				}

				if ((i == 0) && (climb & lmask)) {
					zero_window = false;
					break;
				}

				label[i] = climb;
				wt += Label::template popcnt_T<limb_type>(climb);
			}

			if (!zero_window) {
				continue;
			}

			Value e2;
			e2.zero();
			for (uint32_t j = 0; j < 2*p; ++j) {
				e2.set((e2.get(rows[j]) + coeff[j]) % q, rows[j]);
			}

			if (isd.compare(wt + e2.popcnt(), w)) {
				not_found = false;
				cycles = cpucycles() - cycles;
				for (uint32_t i = 0; i < NR_HT_LIMBS; ++i) {
					label_solution_to_recover.ptr()[i] = label[i];
				}
				value_solution_to_recover = e2;
			}
		}

		cfls = 0;
		return !not_found;
	}

	/// left list into the hashmap
	void init_list() noexcept {
		hm->clear();
		enumerate(0, [this](const l_type label, const size_t index) {
			const uint32_t index32 = index;
			hm->insert(label, V1::create(label, &index32));
		});
	}

	/// right list against the hashmap: left + right = syndrome on the l window
	void find_collisions() noexcept {
		enumerate(enum_offset, [this](const l_type label, const size_t index) {
			if (!not_found) { return; }

			const l_type data = Label::template sub_T<l_type>(l_type(syndrome), label);
			HM_LoadType load;
			const IndexType pos = hm->find(data, load);
			for (uint64_t j = pos; j < pos + load; j++) {
				final_list_left[cfls]  = hm->ptr(j).index[0];
				final_list_right[cfls] = index;
				cfls += 1;

				if (cfls >= final_list_size) {
					compute_finale_list();
				}
			}
		});

		compute_finale_list();
	}

	///
	void rebuild_solution() {
		Error tmpe;
		tmpe.zero();
		e.zero();

		for (uint32_t i = 0; i < l; ++i) {
			assert(label_solution_to_recover.get(i) == 0);
		}
		assert(isd.compare(label_solution_to_recover.popcnt() + value_solution_to_recover.popcnt(), w));

		for (uint32_t i = 0; i < n-k-l; ++i) {
			tmpe.set(label_solution_to_recover.get(l + i), 0, (n-k-l) - i - 1);
		}

		for (uint32_t i = 0; i < k+l; ++i) {
			tmpe.set(value_solution_to_recover.get(i), 0, (n-k-l)+i);
		}

		// apply back permutation
		for (uint32_t i = 0; i < n; ++i) {
			const auto data = tmpe.get(0, i);
			e.set(data, 0, P.values[i]);
		}
	}


	/// \return
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();

		while (not_found && (loops < config.loops)) {
			ISD::step();
			ISD::template extract_lHT<l_type>(lHT);

			cfls = 0;
			init_list();
			find_collisions();
		}

		rebuild_solution();
		return loops;
	}
};
#endif//DECODING_FQ_STERNV2_H
