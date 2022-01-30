#ifndef SMALLSECRETLWE_BJMM_D3_H
#define SMALLSECRETLWE_BJMM_D3_H

#include "bjmm.h"

/// Usage:
///
///
///
///
/// \tparam config
template<const ConfigBJMM &config>
class BJMMD3 : public BJMM<config> {
	/// Depth 3 approach.
	///																Level
	///                         ┌───┐
	///                             │
	///                         │
	///                         └─▲─┘
	///                           │
	///             ┌─────────────┴────────────┐						0
	///             │                          │
	///           ┌─┴─┐                      ┌─┴─┐
	///           │   │ HM2                      │
	///           │   │                      │
	///           └─▲─┘                      └─▲─┘
	///             │                          │
	///      ┌──────┴─────┐              ┌─────┴──────┐					1
	///      │            │              │            │
	///    ┌─┴─┐        ┌─┴─┐          ┌─┴─┐        ┌─┼─┐
	///    │   │ HM1        │              │ HM1        │
	///    │   │        │              │            │
	///    └─▲─┘        └─▲─┘          └─▲─┘        └─▲─┘
	///      │            │              │            │
	///   ┌──┴──┐      ┌──┴──┐        ┌──┴──┐      ┌──┴──┐				2
	///   │     │      │     │        │     │      │     │
	/// ┌─┴─┐ ┌─┴─┐  ┌─┴─┐ ┌─┴─┐    ┌─┴─┐ ┌─┴─┐  ┌─┴─┐ ┌─┴─┐		base lists
	/// │HM0│ │   │      │     │        │     │      │     │
	/// │   │ │   │  │     │        │     │      │     │
	/// └───┘ └───┘  └───┘ └───┘    └───┘ └───┘  └───┘ └───┘
	///	  L1	L2	  L1	L2		  L1	L2	   L1 	 L2

public:
	// types we are using
	using ArgumentLimbType      = typename BJMM<config>::ArgumentLimbType;
	using IndexType             = typename BJMM<config>::IndexType;
	using LoadType              = typename BJMM<config>::LoadType;
	using LabelContainerType    = typename BJMM<config>::LabelContainerType;
	using DecodingList          = typename BJMM<config>::DecodingList;
	using DecodingLabel         = typename BJMM<config>::DecodingLabel;

	// algorith and instance parameters
	constexpr static uint32_t n     = config.n;                             //
	constexpr static uint32_t k     = config.k - config.TrivialAppendRows;  //
	constexpr static uint32_t w     = config.w;                             //
	constexpr static uint32_t p     = config.baselist_p;                    //
	constexpr static uint32_t l     = config.l;                             //
	constexpr static uint32_t l1    = config.l1;                            //
	constexpr static uint32_t l2    = config.l2;                            //
	constexpr static uint32_t nkl   = n - k - l;                            //
	constexpr static uint32_t c     = config.c;                             // dimension to cutoff, currently unused.
	constexpr static uint32_t threads = config.nr_threads;
	constexpr static uint32_t npos_size = 8;
	constexpr static uint32_t number_buckets3 = l - l2 - l1;

	// all the limits we are working with
	constexpr static uint32_t b10 = BJMM<config>::b10;  //
	constexpr static uint32_t b11 = BJMM<config>::b11;  //
	constexpr static uint32_t b12 = BJMM<config>::b12;  //
	constexpr static uint32_t b20 = BJMM<config>::b20;  //
	constexpr static uint32_t b21 = BJMM<config>::b21;  //
	constexpr static uint32_t b22 = BJMM<config>::b22;  //
	constexpr static uint32_t b30 = l1 + l2;            //
	constexpr static uint32_t b31 = l;                  // TODO generalize
	constexpr static uint32_t b32 = l;                  //

	constexpr static ConfigParallelBucketSort chm3{b30, b31, b32,
	                                               config.size_bucket3, uint64_t(1) << number_buckets3,
												   threads, 4, config.n-config.k-config.l, config.l, 0};

	using HM3Type = ParallelBucketSort<chm3, DecodingList, ArgumentLimbType, IndexType, &BJMM<config>::template Hash<b30, b31>>;
	HM3Type *hm3;

	// Because we are now working at depth 3 we need a second and a third intermediate target.
	ArgumentLimbType iT2, iT3;
	using Extractor = WindowExtractor<DecodingLabel, ArgumentLimbType>;

	bool not_found = true;

	BJMMD3(mzd_t *e, const mzd_t *const s, const mzd_t *const A,
		   const uint32_t ext_tid = 0)
			noexcept : BJMM<config>(e, s, A, ext_tid) {
		// currently, there are some restrictions to the code
		static_assert(config.DOOM == false);
		static_assert(config.LOWWEIGHT == false);
		hm3 = new HM3Type();
	};

	~BJMMD3() noexcept {
		delete hm3;
	}

	// execute the algorithm and returns the number of loops the algorith needed.
	uint64_t run() noexcept {
		// we have to reset this value, so It's possible to rerun the algo more often
		not_found = true;
		while (not_found && this->loops < config.loops) {
			matrix_create_random_permutation(this->work_matrix_H, this->work_matrix_H_T, this->permutation);
			matrix_echelonize_partial_plusfix(this->work_matrix_H, config.m4ri_k, n - k - l, this->matrix_data, 0,
			                                  n - k - l, 0, this->permutation);

			// Extract the sub-matrices. the length of the matrix is only n-c + DOOM_nr but we need to copy everything.
			mzd_submatrix(this->H, this->work_matrix_H, config.TrivialAppendRows, n - k - l, n - k,
			              n - c + this->DOOM_nr);
			matrix_transpose(this->HT, this->H);

			Matrix_T<mzd_t *> HH((mzd_t *) this->H);

			// generate targets
			this->iT1 = fastrandombits<ArgumentLimbType, l>();
			iT2 = fastrandombits<ArgumentLimbType, l>();
			iT3 = fastrandombits<ArgumentLimbType, l>();

			if constexpr(!config.DOOM) {
				this->target.data().column_from_m4ri(this->work_matrix_H, n - c - config.LOWWEIGHT, config.TrivialAppendRows);
				this->iTarget = this->extractor(this->target) ^ this->iT1 ^ iT2 ^ iT3;
			}

			// TODO parallel
			{
				const uint32_t tid = 0;
				IndexType pos1, pos2, pos3;
				LoadType load1 = 0, load2 = 0, load3 = 0;
				ArgumentLimbType data1, data2, data3;

				const IndexType s_tid = this->L2.start_pos(tid);
				const IndexType e_tid = this->L2.end_pos(tid);
				uint64_t *Lptr = (uint64_t *) this->L2.data_label() + (s_tid * this->llimbs_a);
				LabelContainerType label5678, label78, label1234, label12, label34;

				IndexType npos[npos_size] = {s_tid, s_tid, s_tid, s_tid, s_tid, s_tid, s_tid, s_tid};

				// start the internal join by preparing everything
				this->BJMM_fill_decoding_lists(this->L1, this->L2, this->cL1, this->cL2, this->HT, tid);
				OMP_BARRIER;

				this->hm1->reset(tid);
				this->hm2->reset(tid);
				hm3->reset(tid);
				OMP_BARRIER;

				// hash the first list
				this->hm1->hash(this->L1, tid);
				OMP_BARRIER;
				this->hm1->sort(tid);
				OMP_BARRIER;

				// HM3              x───────x
				//                   x     x
				//                    x─┬─x
				//                      │
				//                      │
				//            ┌─────────┴──────┐
				//            │                │
				//            │                │
				// HM2    x───┴───x            │
				//         x     x             │
				//          x───x              │
				//            ▲                │
				//            │                │
				//        ┌───┴───┐            │
				//        │       │            │
				//        ├───────┼────────────┴────┐
				// HM1x───┴──x    │                 │
				//     x    x     │                 │
				//      x┌─x      │                 │
				//       ▲        │                 │
				//     ┌────┐  ┌──┴─┐   ┌────┐   ┌──┴─┐
				//     │L1  │  │ L2 │   │ L1 │   │ L2 │
				//     │    │  │    │   │    │   │    │
				//     └────┘  └────┘   └────┘   └────┘
				// first join between L1 L2
				for (; npos[1] < e_tid; ++npos[1], Lptr += this->llimbs_a) {
					data1 = this->extractor_ptr(Lptr);
					ASSERT((this->hm1->check_label(data1, this->L2, npos[1])));
					data1 ^= this->iT1;

					pos1 = this->hm1->find(data1, load1);
					while (pos1 < load1) {
						npos[0] = this->hm1->__buckets[pos1].second[0];
						data2 = data1 ^ this->hm1->__buckets[pos1].first;
						this->hm2->insert(data2, npos, tid);
						pos1 += 1;
					}
				}

				OMP_BARRIER;
				this->hm2->sort(tid);

				OMP_BARRIER;
				ASSERT(this->hm2->check_sorted() OMP_BARRIER);

				// now the join between L3, L4
				Lptr = (uint64_t *) this->L2.data_label() + (s_tid * this->llimbs_a);
				for (; npos[3] < e_tid; ++npos[3], Lptr += this->llimbs_a) {
					data1 = this->extractor_ptr(Lptr);
					ASSERT((this->hm1->check_label(data1, this->L2, npos[3])));
					data1 ^= this->iT2;

					pos1 = this->hm1->find(data1, load1);
					while (pos1 < load1) {
						npos[2] = this->hm1->__buckets[pos1].second[0];
						data2 = data1 ^ this->hm1->__buckets[pos1].first;
						pos1 += 1;

						pos2 = this->hm2->find(data2, load2);
						while (pos2 < load2) {
							npos[0] = this->hm2->__buckets[pos2].second[0];
							npos[1] = this->hm2->__buckets[pos2].second[1];

							data3 = data2 ^ this->hm2->__buckets[pos2].first;
							hm3->insert(data3, npos, tid);
							pos2 += 1;
						}
					}
				}

				OMP_BARRIER;
				hm3->sort(tid);

				OMP_BARRIER;
				ASSERT(hm3->check_sorted() OMP_BARRIER);

				// now you can clear the second hashmap hm2 and refill it with solutions for the join between L5 and L
				this->hm2->reset(tid);
				npos[1] = s_tid;
				OMP_BARRIER;

				// join between L5 and L6
				Lptr = (uint64_t *) this->L2.data_label() + (s_tid * this->llimbs_a);
				for (; npos[1] < e_tid; ++npos[1], Lptr += this->llimbs_a) {
					data1 = this->extractor_ptr(Lptr);
					ASSERT((this->hm1->check_label(data1, this->L2, npos[1])));
					data1 ^= iT3;

					// for all elements
					pos1 = this->hm1->find(data1, load1);
					while (pos1 < load1) {
						npos[0] = this->hm1->__buckets[pos1].second[0];
						data2 = data1 ^ this->hm1->__buckets[pos1].first;
						this->hm2->insert(data2, npos, tid);
						pos1 += 1;
					}
				}

				OMP_BARRIER;
				this->hm2->sort(tid);

				OMP_BARRIER;
				ASSERT(this->hm2->check_sorted() OMP_BARRIER);

				// now finally we are at the last join. The join between L7, L8
				Lptr = (uint64_t *) this->L2.data_label() + (s_tid * this->llimbs_a);
				for (; npos[7] < e_tid; ++npos[7], Lptr += this->llimbs_a) {
					data1 = this->extractor_ptr(Lptr);
					ASSERT(this->hm1->check_label(data1, this->L2, npos[7]));
					data1 ^= this->iTarget;

					pos1 = this->hm1->find(data1, load1);

					// we actually found a collision between L7 and L8
					while (pos1 < load1) {
						// now copy the collision index into the correct position
						npos[6] = this->hm1->__buckets[pos1].second[0];
						data2 = data1 ^ this->hm1->__buckets[pos1].first;
						pos1 += 1;

						// this will maybe find us a collision between L56 and L78
						pos2 = this->hm2->find(data2, load2);

						if (pos2 != IndexType(-1)) {
							// these little ifs are just optimisations to reduce the amount of random memory accesses at once.
							LabelContainerType::add(label78, this->L1.data_label(npos[6]).data(), this->L2.data_label(npos[7]).data());
							LabelContainerType::add(label78, label78, this->target.data());
						}

						// for every colliding element between L56 and L78
						while (pos2 < load2) {
							// copy the collision index into the npos array
							npos[4] = this->hm2->__buckets[pos2].second[0]; // L5
							npos[5] = this->hm2->__buckets[pos2].second[1]; // L6
							pos2 += 1;

							data3 = data2 ^ this->hm2->__buckets[pos2].first;

							// maybe we find a collision between L1234 and L5678
							pos3 = hm3->find(data3, load3);

							if (pos3 != IndexType(-1)) {
								LabelContainerType::add(label5678, this->L1.data_label(npos[4]).data(), this->L2.data_label(npos[5]).data());
								LabelContainerType::add(label5678, label5678, label78);
							}

							// do we actually find a 8 sum?
							while (pos3 < load3) {
								npos[0] = hm3->__buckets[pos3].second[0];
								npos[1] = hm3->__buckets[pos3].second[1];
								npos[2] = hm3->__buckets[pos3].second[2];
								npos[3] = hm3->__buckets[pos3].second[3];
								pos3 += 1;

								LabelContainerType::add(label12, this->L2.data_label(npos[1]).data(), this->L1.data_label(npos[0]).data());
								LabelContainerType::add(label34, this->L2.data_label(npos[3]).data(), this->L1.data_label(npos[2]).data());
								LabelContainerType::add(label1234, label12, label34);
								uint32_t weight = LabelContainerType::add_weight(label1234, label1234, label5678);

								std::cout << label12 << " label12 " << npos[0] << " " << npos[1] << "\n";
								std::cout << label34 << " label34 " << npos[2] << " " << npos[3] << "\n";
								std::cout << label5678 << " label5678 "  << npos[4] << " " << npos[5] << "\n";
								std::cout << label1234 << " label12345678 " << npos[6] << " " << npos[7] << "\n";
								std::cout << this->target << " target\n";

								for (int i = n - config.k - config.l; i < n - config.k; ++i) {
									if (label1234[i] != 0) {
										std::cout << this->target << " target\n";
										std::cout << label1234    << " error label12345678\n";
										ASSERT(false);
									}
								}

								if (unlikely(weight <= config.weight_threshhold)) {
									this->check_final_list(label1234, npos, weight, 0);
								}
							} // while lvl3
						} // while lvl2
					} // while lvl1
				} // while element in L8
			} // while parallel

			if (unlikely(this->loops == 0)) {
				this->info();
			}

			// print periodic information
			if (unlikely((this->loops % config.print_loops) == 0)) {
				this->periodic_info();
			}

			// check if another thread found the solution already.
			OUTER_MULTITHREADED_WRITE(
			if ((unlikely(loops % config.exit_loops) == 0)) {
				if (finished.load()) {
					return loops;
				}
			})

			// and last but least update the loop counter
			this->loops += 1;

		} // while not found

		return this->loops;
	}

	constexpr static std::array<uint64_t, 4> ListSizes() noexcept {
		uint64_t S1 = (BJMM<config>::lsize2 * BJMM<config>::lsize2) >> l1;
		uint64_t S2 = (S1 * S1) >> l2;
		uint64_t S3 = (S2 * S2) >> (l-l2-l1);

		std::array<uint64_t, 4> ret{BJMM<config>::lsize2, S1, S2, S3};
		return ret;
	}
};


#endif //SMALLSECRETLWE_BJMM_D3_H
