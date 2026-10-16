#ifndef DECODING_PRANGE_H
#define DECODING_PRANGE_H

#include <cstdint>

#include "matrix/matrix.h"
#include "container/hashmap.h"
#include "algorithm/bits/popcount.h"
#include "math/bc.h"
#include "alloc/alloc.h"
#include "hash/simple.h"
#include "mitm.h"

#include "isd.h"

using namespace cryptanalysislib;

template<const ConfigISD &isd>
class Prange : public ISDInstance<uint64_t, isd> {
public:
	constexpr static uint32_t n = isd.n;
	constexpr static uint32_t k = isd.k;
	constexpr static uint32_t w = isd.w;
	constexpr static uint32_t p = isd.p; // NOTE: from now on p is the baselist p not the full p.
	constexpr static uint32_t l = isd.l;
	constexpr static uint32_t q = isd.q;

	static_assert(l == 0);
	static_assert(p == 0);

	using ISD = ISDInstance<uint64_t, isd>;
	using Label 		= ISD::Label;
	using limb_type 	= ISD::limb_type;
	using ISD::A,ISD::H,ISD::wA,ISD::wAT,ISD::HT,ISD::s,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::ghz;
	using ISD::cycles, ISD::gaus_cycles, ISD::periodic_print;
	using ISD::pHT;

	constexpr static bool packed = true;

	// base datatype of the hashmap
	using l_type   = MinLogTypeTemplate<l, 32>;

	/// reconstruct the final solution
	void __attribute__ ((noinline))
	reconstruct() noexcept {
		e.zero();
		// apply back permutation.
		for (uint32_t i = 0; i < n-k; ++i) {
			e.set(wA.get(i, n), 0, P.values[i]);
		}
	}

	// runs: optimized stern
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();

		while (not_found && loops < isd.loops) {
			ISD::template step<false, false, false>();

			if (ws.popcnt() <= w) {
				ISD::extract_pHT(pHT);

				not_found = false;
				break;
			}
		}

		// compute the final solution
		reconstruct();
		return loops;
	}

	/// important. Dont rename it
	void info() const noexcept {}
};
#endif//DECODING_PRANGE_H
