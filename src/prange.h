#ifndef DECODING_STERN_H
#define DECODING_STERN_H

#include <cstdint>

#include "matrix/binary_matrix.h"
#include "container/hashmap.h"
#include "popcount/popcount.h"
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

	using ISD = ISDInstance<uint64_t, isd>;
	using PCMatrixOrg 	= ISD::PCMatrixOrg;
	using PCMatrixOrg_T = ISD::PCMatrixOrg_T;
	using PCMatrix 		= ISD::PCMatrix;
	using PCMatrix_T 	= ISD::PCMatrix_T;
	using PCSubMatrix 	= ISD::PCSubMatrix;
	using PCSubMatrix_T = ISD::PCSubMatrix_T;
	using Syndrome 		= ISD::Syndrome;
	using Error 		= ISD::Error;
	using Label 		= ISD::Label;
	using limb_type 	= ISD::limb_type;
	using ISD::A,ISD::H,ISD::wA,ISD::wAT,ISD::HT,ISD::s,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::ghz;
	using ISD::cycles, ISD::gaus_cycles, ISD::periodic_print;

	constexpr static bool packed = true;

	// base datatype of the hashmap
	using l_type   = MinLogTypeTemplate<l, 32>;

	constexpr static uint32_t NR_HT_T_LIMBS = (n-k+(sizeof(l_type)*8) - 1u)/(sizeof(l_type)*8);

	/// internal arrays pointing to
	l_type *pHT;


	Prange() noexcept {
		constexpr size_t size_pHT = roundToAligned<1024>(sizeof(l_type) * (k+l) * NR_HT_T_LIMBS);
		pHT =(l_type *)cryptanalysislib::aligned_alloc(1024, size_pHT);
		ASSERT(pHT);
	}

	~Prange() noexcept {
		free(pHT);
	}

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
#endif//DECODING_STERN_H
