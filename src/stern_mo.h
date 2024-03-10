#ifndef DECODING_STERNMO
#define DECODING_STERNMO

#include <cstdint>
#include <utility>

#include "isd.h"
#include "stern.h"

struct ConfigSternMO : public ConfigISD {
public:
	const uint32_t nr_views = 0;

	/// Well, ... currently this forces the windowsize `k` = 64.
	const uint32_t r = 3;//(n-k + 63) / 64;
	const uint32_t N = 50; //
	const uint32_t dk = 12;// (w-2*p) + 6;
	const uint32_t nnk = 32;

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
		          << ", \"r\": " << r
				  << ", \"N\": " << N
				  << ", \"dk\": " << dk
				  << ", \"nnk\": " << nnk
				  << " }" << std::endl;
	}
};

template<const ConfigISD &isd, const ConfigSternMO &config>
class SternMO : public ISDInstance<uint64_t, isd>{
public:
	constexpr static uint32_t n = isd.n;
	constexpr static uint32_t k = isd.k;
	constexpr static uint32_t w = isd.w;
	constexpr static uint32_t p = isd.p;
	constexpr static uint32_t l = isd.l;

	constexpr static uint32_t r = config.r;

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
	using ISD::A,ISD::H,ISD::wA,ISD::wAT,ISD::HT,ISD::s,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::ghz,ISD::expected_loops;
	using ISD::cycles, ISD::gaus_cycles, ISD::periodic_print, ISD::packed, ISD::simd;

	// if set to
	constexpr static uint32_t kl_half =  k/2;
	constexpr static uint32_t NR_HT_LIMBS = PCSubMatrix_T::limbs();

	// list shit, todo rename to baselist
	constexpr static size_t list_enumeration_length = kl_half;
	constexpr static size_t list_enumeration_weigth = p;
	constexpr static size_t list_enumeration_size =
			bc(list_enumeration_length, list_enumeration_weigth);
	constexpr static size_t size_MO_L = roundToAligned<1024>(list_enumeration_size*sizeof(Label));
	Label *stern_MO_L1 = nullptr, *stern_MO_L2 = nullptr;


	constexpr static NN_Config nn_config{
	        n-k, config.r, config.N, config.nnk,
	        list_enumeration_size, config.dk, w-(2*p), 0, 1024};
	NN<nn_config> algo {};
	using NNElement = typename NN<nn_config>::Element;

	size_t solutions[2*p] = {0};

	constexpr SternMO() noexcept {
		expected_loops = config.compute_loops();

		stern_MO_L1 = (Label *)cryptanalysislib::aligned_alloc(1024, size_MO_L);
		stern_MO_L2 = (Label *)cryptanalysislib::aligned_alloc(1024, size_MO_L);

		algo.L1 = (NNElement *)stern_MO_L1;
		algo.L2 = (NNElement *)stern_MO_L2;
	}

	/// reconstruct the solution
	void __attribute__ ((noinline))
	reconstruct() {
		// we need to translate the indices from the NN to stern list indices.
		const size_t soll = solutions[0];
		const size_t solr = solutions[1];
		uint16_t rows[p];
		Label tmpl, tmpr;
		bool foundl = false, foundr = false;


		for (size_t i = 0; i < list_enumeration_size; i++) {
			biject<k, p>(i, rows);
			bool correctl = true;
			bool correctr = true;

			tmpl.zero(); tmpr = ws;

			for (uint16_t j = 0; j < p; j++) {
				ASSERT(rows[j] < kl_half);
				Label::add(tmpl.ptr(), tmpl.ptr(), HT[rows[j]]);
				Label::add(tmpr.ptr(), tmpr.ptr(), HT[rows[j] + list_enumeration_length]);
			}

			if (!tmpl.is_equal(((Label*)algo.L1)[soll])) {
				correctl = false;
			}
			if (!tmpr.is_equal(((Label*)algo.L2)[solr])) {
				correctr = false;
			}

			if (correctl) {
				ASSERT(!foundl);
				foundl = true;
				for (uint32_t j = 0; j < p; j++) {
					solutions[j] = rows[j];
				}
			}

			if (correctr) {
				ASSERT(!foundr);
				foundr = true;
				for (uint32_t j = 0; j < p; j++) {
					solutions[p + j] = rows[j] + list_enumeration_length;
				}
			}
		}

		ASSERT(foundl);
		ASSERT(foundr);

		Error tmpe, tmpe2;
		Label tmp; tmp.zero();
		// copy in the syndrome
		tmp = ws;
		for (uint32_t i = 0; i < 2*p; ++i) {
			ASSERT(solutions[i] < k+l);
			Label::add(tmp.ptr(), tmp.ptr(), HT.row(solutions[i]));
		}

		tmp.print();

		// reversing and shit
		for (uint32_t i = 0; i < n-k; ++i) {
			const auto bit = tmp.get(i);
			tmpe.set(bit, 0, n-k-1-i);
		}

		tmpe.print();
		for (uint32_t i = 0; i < 2*p; ++i) {
			ASSERT(n-k-l + solutions[i] < n);
			tmpe.set(1, 0, n-k-l + solutions[i]);
		}

		// apply back permutation.
		for (uint32_t i = 0; i < n; ++i) {
			e.set(tmpe.get(0, i), 0, P.values[i]);
		}
	}


	///
	constexpr void construct_nearest_neighbour_lists() noexcept {
		alignas(32) Label tmpl, tmpr;
		uint16_t rows[p];

		for (size_t i = 0; i < list_enumeration_size; i++) {
			biject<k, p>(i, rows);
			tmpl.zero(); tmpr = ws;

			#pragma unroll
			for (uint16_t j = 0; j < p; j++) {
				ASSERT(rows[j] < kl_half);
				Label::add(tmpl.ptr(), tmpl.ptr(), HT[rows[j]]);
				Label::add(tmpr.ptr(), tmpr.ptr(), HT[rows[j] + list_enumeration_length]);
			}

			stern_MO_L1[i] = tmpl;
			stern_MO_L2[i] = tmpr;
		}
	}

	/// generates the two lists L1 and L2 from HT
	constexpr void apply_nearest_neighbour() noexcept {
		// TODO
		// algo.avx2_nn(enumeration_size, enumeration_size);
		algo.bruteforce(list_enumeration_size, list_enumeration_size);
		if (algo.solutions_nr > 0) {
			solutions[0] = algo.solutions[0].first;
			solutions[1] = algo.solutions[0].second;

			not_found = false;
		}
	}

	/// runs the May-Ozerov Stern versions
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();

		while (not_found && loops < isd.loops) {
			ISD::step();
			construct_nearest_neighbour_lists();
			apply_nearest_neighbour();
		}

		reconstruct();
		return loops;
	}

	/// important. Dont rename it
	constexpr void info() const noexcept {
		std::cout << "{"
				  << " \"size_MO_L\": " << size_MO_L
				  << " }" << std::endl;
	}
};
#endif
