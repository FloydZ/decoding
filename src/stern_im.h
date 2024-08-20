#ifndef DECODING_STERN_IM_H
#define DECODING_STERN_IM_H
#include <cstdint>

#include "mitm.h"
#include "isd.h"
#include "stern.h"

using namespace cryptanalysislib;


struct ConfigSternIM : public ConfigISD {
public:
	const uint32_t nr_views = 1;

	[[nodiscard]] consteval uint64_t compute_loops() const noexcept {
#ifdef EXPECTED_PERMUTATIONS
		return EXPECTED_PERMUTATIONS;
#else
		return 1; // TODO
#endif
	}

	void print() const noexcept {
		ConfigISD::print();
		std::cout << "{ \"nr_views\": " << nr_views
				  << " }" << std::endl;
	}
};


template<const ConfigISD &isd, const ConfigStern &configStern, const ConfigSternIM &config>
class SternIM : public Stern<isd, configStern> {
public:
	using ISD = ISDInstance<uint64_t, isd>;
	using STERN = Stern<isd, configStern>;
	using limb_type 	= ISD::limb_type;
	using ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::expected_loops,ISD::ghz,ISD::cycles,ISD::gaus_cycles,ISD::periodic_print;

	using l_type 		= STERN::l_type;
	using STERN::reconstruct,STERN::compute_finale_list,STERN::compute_finale_list_simd;
	using STERN::simd,STERN::lHT,STERN::pHT,STERN::bEnum,STERN::cEnum,STERN::final_list_current_size,STERN::final_list_left,STERN::final_list_right, STERN::final_list_max_size;
	constexpr static uint32_t nr_views = config.nr_views;

	static_assert(nr_views > 0);
	static_assert(nr_views*isd.l < (isd.n - isd.k));

	SternIM() noexcept : Stern<isd, configStern>() {
		expected_loops = config.compute_loops();
	}

	// runs: optimized stern
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();

		// collision function
		auto f = [&, this](const l_type a1, const l_type a2,
		                   uint16_t *index1, uint16_t *index2,
		                   const uint32_t nr_cols) __attribute__((always_inline)) -> bool {
			(void)nr_cols;
		    (void)a1;
		    (void)a2;
			// const l_type a = a1 ^ a2;
			ASSERT(bEnum->check_hashmap2(syndrome, index1, 2u*config.p, isd.l));

		    for (uint32_t i = 0; i < config.p; i++) {
				final_list_left[final_list_current_size][i]  = index1[i];
				final_list_right[final_list_current_size][i] = index2[i];
		    }
			final_list_current_size += 1;

			if (final_list_current_size >= final_list_max_size) {
				compute_finale_list();
			}
			return false;
		};

		while (not_found && loops < isd.loops) {
			const uint32_t tid = 0;
			const bool simd = false;
			ISD::step();
			ISD::extract_pHT(pHT);

			constexpr_for<0, nr_views, 1>([&, this, f](const auto i){
				constexpr auto offset = i*isd.l;
				ISD::template extract_lHT<limb_type, offset>(lHT);
				ISD::template extract_syndrome_limb<offset>();

				bEnum->step(0, tid, simd);
				cEnum->step(syndrome, f, tid, simd);
				compute_finale_list();
			});
		}

		reconstruct();
		return loops;
	}
};
#endif//DECODING_STERN_IM_H
