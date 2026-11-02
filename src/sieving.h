#ifndef DECODING_SIEVING_H
#define DECODING_SIEVING_H

#include "bjmm.h"
#include <cstddef>


struct ConfigSieving : public ConfigISD {
public:
	const uint32_t l1 = 4;

	// returns the expected number of iterations, if all collisions are found
	[[nodiscard]] constexpr uint64_t compute_loops() const noexcept {
#ifdef EXPECTED_PERMUTATIONS
		return EXPECTED_PERMUTATIONS;
#else
		return 1; // TODO
#endif
	}

	void print() const noexcept {
		ConfigISD::print();
		std::cout << "{ \"l1\": " << l1
				  // << ", \"table_bits\": " << table_bits
				  << " }" << std::endl;
	}
};

template<const ConfigISD &isd, const ConfigBJMM &config>
class Sieving : public ISDInstance<uint64_t, isd> {
	// copy instance parameters from configuration
	constexpr static uint32_t n = config.n;
	constexpr static uint32_t k = config.k;
	constexpr static uint32_t w = config.w;
	constexpr static uint32_t p = config.p;
	constexpr static uint32_t l = config.l;
	constexpr static uint32_t l1 = config.l1;
	constexpr static uint32_t l2 = config.l - config.l1;
	constexpr static uint32_t nkl = n - config.k - l;
	constexpr static uint32_t c = config.c;

	using DecodingValue         = typename BJMM<config>::DecodingValue;
	using DecodingLabel         = typename BJMM<config>::DecodingLabel;
	using DecodingMatrix        = typename BJMM<config>::DecodingMatrix;
	using DecodingElement       = typename BJMM<config>::DecodingElement;
	using DecodingList          = typename BJMM<config>::DecodingList;
	using ChangeList            = typename BJMM<config>::ChangeList;

	using ArgumentLimbType      = LogTypeTemplate<l1+l2>;

	using IndexType             = typename BJMM<config>::IndexType;
	using LoadType              = typename BJMM<config>::LoadType;

	using ValueContainerType    = typename BJMM<config>::ValueContainerType;
	using LabelContainerType    = typename BJMM<config>::LabelContainerType;
	using ValueContainerLimbType= typename BJMM<config>::ValueContainerLimbType;


	using Extractor         = WindowExtractor<DecodingLabel, ArgumentLimbType>;

	const size_t lsize1 = 100,
	             lsize2 = 100;
	const size_t thread_size_lists1 = lsize1/config.nr_threads,
				 thread_size_lists2 = lsize2/config.nr_threads;

	DecodingList L1{lsize1, config.nr_threads, config.no_values},
			     L2{lsize2, config.nr_threads, config.no_values};
public:

	/// constructor. initializes everything you need to run MO.
	/// 	Execute the algorithm with:
	///			static constexpr ConfigBJMM config(n, k, w, 1, l1, l1, 20, 20, 1, 1, 4, NUMBER_THREADS, false, false, false, 0, false, l2, IMv);
	///  		Sieving<config> s(ee, ss, A);
	///			s.run();
	/// \param e 		output: error vector
	/// \param s		input: syndrome
	/// \param A		input: parity check matrix
	/// \param ext_tid	input: thread id of the external calling thread.
	Sieving(mzd_t *e, const mzd_t *const s, const mzd_t *const A, const uint32_t ext_tid = 0)
			: BJMM<config>(e, s, A, ext_tid) {
	};


	void Init(const uint32_t filterWeight, const uint32_t tid) {

	}

	void ProcessCartesian(const uint32_t filterWeight,
		                  const uint32_t tid,
		                  const bool current_iter) {
		DecodingList *inList  = current_iter==0 ? &L1 : &L2;
		DecodingList *outList = current_iter==1 ? &L1 : &L2;

		// sizes not dynamic
		const size_t endInList    = inList->end_pos(tid);
		const size_t endOutList   = outList->end_pos(tid);
		const size_t startInList  = inList->start_pos(tid);
		      size_t startOutList = inList->start_pos(tid);

		LabelContainerType label;
		for (size_t i = startInList; i < endInList; ++i) {
			for (size_t j = startInList; j < endInList; ++j) {
				if (i==j) {
					continue;
				}

				LabelContainerType::add(label, inList->data_label(i), inList->data_label(i));
				const uint32_t weight = label.weight();

				if (weight <= filterWeight) {
					outList->data_label(startOutList++) = label;
				}

				if (startOutList == endOutList) {
					goto thread_finish;
				}
			}
		}
	thread_finish:
		 ;
	}

	// executes the algorithm and returns the number of loops/iterations needed
	uint64_t __attribute((noinline))
	run() noexcept {
		// count the loops we iterated
		this->loops = 0;

		// we have to reset this value, so its possible to rerun the algo more often
		this->not_found = true;

		// reset the timer
		this->internal_time = (double)clock();

		// start the whole thing
		while (this->not_found && this->loops < config.loops) {
			// Apply the random permutation, Gaus elimination, and extraction of the submatrices.
			matrix_create_random_permutation(this->work_matrix_H, this->work_matrix_H_T, this->permutation);
			matrix_echelonize_partial_plusfix(this->work_matrix_H, config.m4ri_k, n-k-l, this->matrix_data, 0, n-k-l, 0, this->permutation);

			mzd_submatrix(this->H, this->work_matrix_H, config.TrivialAppendRows, n-k-l, n-k, n-c);
			matrix_transpose(this->HT, this->H);

			// just for debugging purpose
			DecodingMatrix HH((mzd_t *) this->H);

			//#pragma omp parallel default(none) shared(std::cout, this->L1, this->L2, this->cL1, this->cL2, HH, this->H, this->HT, this->target, this->not_found, this->loops, finished) num_threads(this->threads)
			{
				const uint32_t tid  = config.nr_threads == 1 ? 0 : omp_get_thread_num();
				this->BJMM_fill_decoding_lists(this->L1, this->L2, this->cL1, this->cL2, this->HT, tid);
			} // end pragma omp parallel

			// finally, update the loops counter
			this->loops += 1;
		} // end while not found loop

		return this->loops;
	}


};

#endif
