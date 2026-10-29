#ifndef DECODING_FQ_LARGE_WEIGHT_H
#define DECODING_FQ_LARGE_WEIGHT_H

#include <cstdint>

#include "helper.h"
#include "random.h"
#include "isd.h"

using namespace cryptanalysislib;

/// Prange for the large weight syndrome decoding problem, see
/// https://decodingchallenge.org/large-weight: find e in F_q^n with
/// H*e^T = s^T and wt(e) >= w.
/// After the gaussian elimination H = [I_{n-k} | Q], hence every choice of the
/// information part e2 in F_q^k determines e1 = s' - Q*e2. The information part
/// is chosen with full weight k, so only wt(e1) >= w - k is left to check.
/// \tparam config
template<const ConfigISD &config>
class FqLargeWeightPrange : public ISDInstance<uint64_t, config> {
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t k = config.k;
	constexpr static uint32_t w = config.w;
	constexpr static uint32_t q = config.q;

	static_assert(config.p == 0);
	static_assert(config.l == 0);
	static_assert(q > 2, "over F_2 the only full weight vector is the all one vector");
	static_assert(w <= n);

	/// number of information parts tried per gaussian elimination, as the
	/// elimination is far more expensive than a single try.
	constexpr static uint32_t tries_per_step = 32;

public:
	using ISD = ISDInstance<uint64_t, config>;
	using Error 		= typename ISD::Error;
	using Syndrome 		= typename ISD::Syndrome;
	using PCMatrixOrg 	= typename ISD::PCMatrixOrg;
	using ISD::A,ISD::wA,ISD::s,ISD::e,ISD::P,ISD::not_found,ISD::loops;

	/// current candidate in the permuted coordinates:
	/// [0, n-k) is e1, [n-k, n) is e2.
	uint32_t candidate[n] = {0};

	/// tries `tries_per_step` random full weight information parts e2
	/// \return true if a solution was found. It is written into `candidate`.
	bool try_information_parts() noexcept {
		for (uint32_t t = 0; t < tries_per_step; ++t) {
			for (uint32_t j = 0; j < k; ++j) {
				candidate[n - k + j] = 1u + rng<uint32_t>(q - 1u);
			}

			uint32_t wt = k;
			for (uint32_t i = 0; i < n - k; ++i) {
				// e1_i = s'_i - sum_j Q_ij * e2_j
				uint32_t v = uint32_t(wA.get(i, n));
				for (uint32_t j = 0; j < k; ++j) {
					const uint32_t m = (uint32_t(wA.get(i, n - k + j)) * candidate[n - k + j]) % q;
					v = (v + q - m) % q;
				}

				candidate[i] = v;
				wt += (v != 0u);
			}

			if (wt >= w) {
				return true;
			}
		}

		return false;
	}

	/// \return needed loops
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();

		while (not_found && (loops < config.loops)) {
			ISD::template step<false, false, false>();
			if (try_information_parts()) {
				not_found = false;
			}
		}

		// apply back permutation
		e.zero();
		for (uint32_t i = 0; i < n; ++i) {
			e.set(typename Error::DataType(candidate[i]), 0, P.values[i]);
		}

		return loops;
	}

	/// NOTE: hides `ISDInstance::correct`, which checks wt(e) <= w.
	/// \return true if H*e^T = s^T and wt(e) >= w
	bool correct() const noexcept {
		uint32_t wt = 0;
		for (uint32_t i = 0; i < n; ++i) {
			wt += (e.get(0, i) != 0);
		}

		Syndrome s2;
		PCMatrixOrg::mul_transposed(s2, A, e);
		return (wt >= w) && Syndrome::is_equal(s, s2);
	}

	/// important. Dont rename it
	void info() const noexcept {}
};
#endif//DECODING_FQ_LARGE_WEIGHT_H
