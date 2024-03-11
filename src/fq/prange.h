#ifndef DECODING_FQ_PRANGE_H
#define DECODING_FQ_PRANGE_H

#include "combination/chase.h"
#include "helper.h"
#include "container/fq_vector.h"
#include "matrix/fq_matrix.h"
#include "list/list.h"
#include "list/enumeration/fq.h"
#include "thread/thread.h"
#include "isd.h"


///
/// \tparam config
template<const ConfigISD &config>
class FqPrange : public ISDInstance<uint64_t, config> {
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t k = config.k;
	constexpr static uint32_t w = config.w;
	constexpr static uint32_t q = config.q;
	constexpr static uint32_t p = config.p;

	static_assert(config.l == 0);

public:
	using ISD = ISDInstance<uint64_t, config>;
	using Error 		= ISD::Error;
	using Label 		= ISD::Label;
	using Value 		= ISD::Value;
	using Element 		= ISD::Element;
	using limb_type 	= ISD::limb_type;
	using ISD::A,ISD::H,ISD::wA,ISD::wAT,ISD::HT,ISD::s,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::ghz;
	using ISD::cycles, ISD::gaus_cycles, ISD::periodic_print;

	/// needed list stuff
	using List = List_T<Element>;
	using Generator = ListEnumerateMultiFullLength<List, k, q, p>;
	Generator G;

	FqPrange() noexcept : G(HT, 0) {}

	/// rebuild the final solution
	void rebuild_solution() {
		Error tmpe;
		tmpe.zero();
		e.zero();

		/// NOTE: the `element1` the code below is referencing is part of the
		/// `Enumeration class'. Its the field which is actually enumerated by the class.
		/// And we need access to it, to know the label, value tuple.
		if constexpr (p == 0) {
			for (uint32_t i = 0; i < n-k; ++i) {
				const auto data = ws.get(i);
				tmpe.set(data, 0, n-k-1-i);
			}
		} else {
			// ASSERT(G.element1.label.popcnt() == (w-p));
			ASSERT(G.element1.value.popcnt() == p);

			G.element1.label.print();
			G.element1.value.print();

			Label tmp;
			Label::sub(tmp, ws, G.element1.label);
			ws.print();
			tmp.print();
			ASSERT(tmp.popcnt() == (w-p));
			/// set e1 (get the label)
			for (uint32_t i = 0; i < n-k; ++i) {
				const auto data = tmp.get(i);
				tmpe.set(data, 0, n-k-1-i);
			}

			/// set e2 (get the value)
			for (uint32_t i = 0; i < k; ++i) {
				const auto data = G.element1.value.get(i);
				tmpe.set(data, 0, i+n-k);
			}
		}

		// apply back permutation
		for (uint32_t i = 0; i < n; ++i) {
			const auto data = tmpe.get(0, i);
			e.set(data, 0, P.values[i]);
		}
	}

	/// \return needed loops
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();

		while (not_found && (loops < config.loops)) {

			/// check if prange or Lee-Brickell
			if constexpr (p == 0) {
				ISD::template step<false, false, false>();
				const uint32_t weight = ws.popcnt();
				ASSERT(weight <= n-k);
				ASSERT(weight > 0);
				if (unlikely(weight <= w)) {
					not_found = false;
					break;
				}
			} else {
				// TODO can be optimizes. Currently we need to swap because the syndrome is swapped to.
				ISD::template step<true>();
				auto predicate = [&, this](const Label &in){
					Label tmp;
					Label::sub(tmp, ws, in);
					return tmp.popcnt() <= (w - p);
				};

				const uint32_t tid = Thread::get_tid();

				/// start the iterations
				bool found = G.template run
				             <std::nullptr_t, std::nullptr_t>
				             (nullptr, nullptr, 0, tid, nullptr, nullptr, &predicate);

				if (unlikely(found)) {
					not_found = false;
					break;
				}

			}
		}

		rebuild_solution();
		return loops;
	}
};

#endif//DECODING_FQ_PRANGE_H
