#ifndef SMALLSECRETLWE_BJMM_HYBRIDTREE_H
#define SMALLSECRETLWE_BJMM_HYBRIDTREE_H

#include "bjmm.h"

template<const ConfigBJMM &config>
class BJMMHybridTree : public BJMM<config> {
	//	                              Final  Check
	//	                                 ▲
	//	                                 │
	//	               ┌─────────────────┴────────────────┐
	//	               │                                  │
	//	               │                                  │
	//	xxxxxxxxxxxxxxx│xxxxxxxxxxxxxxx                   │
	//	 x                           x                    │
	//	  x                         x                     │
	//	   x                       x                      │
	//	    x        HM2          x                       │
	//	     x                   x                        │
	//	      x                 x                         │
	//	       x               x                          │
	//	        xxxxxxxxxxxxxxx                           │
	//	          │         ▲                             │
	//	          │         │                             │
	//	          │         │ l2                          │
	//	          └─────────│                             │
	//	        ┌───────────┴─┐                    ┌──────┴──────┐
	//	        │             │ L12                │             │ L34
	//	        │             │ Saved on full len  │             │
	//	        │             │                    │             │
	//	        │             │                    │             │
	//	        │             │                    │             │
	//	        │             │                    │             │
	//	        │             │                    │             │
	//	        │             │                    │             │
	//	        │             │                    │             │
	//	        │             │                    │             │
	//	        │             │                    │             │
	//	        └──────▲──────┘                    └──────▲──────┘
	//	               │ l1                               │ l1
	//	       ┌───────┴───────┐                 ┌────────┴───────┐
	//	       │               │                 │                │
	//	┌──────┼──────┐  ┌─────┴───────┐  ┌──────┴──────┐  ┌──────┴──────┐
	//	│Value │Label │  │             │  │             │  │             │
	//	│ e    │ He   │  │             │  │             │  │             │
	//	│      │      │  │             │  │             │  │             │
	//	│      │      │  │             │  │             │  │             │
	//	│      │      │  │             │  │             │  │             │
	//	│      │      │  │             │  │             │  │             │
	//	│      │      │  │             │  │             │  │             │
	//	│      │      │  │             │  │             │  │             │
	//	│      │      │  │             │  │             │  │             │
	//	│      │      │  │             │  │             │  │             │
	//	│      │      │  │             │  │             │  │             │
	//	└──────┴──────┘  └─────────────┘  └─────────────┘  └─────────────┘
	//	      L1               L2               L3               L4
	//
	using DecodingValue     = typename BJMM<config>::DecodingValue;
	using DecodingLabel     = typename BJMM<config>::DecodingLabel;
	using DecodingMatrix    = typename BJMM<config>::DecodingMatrix;
	using DecodingElement   = typename BJMM<config>::DecodingElement;
	using DecodingList      = Parallel_List_FullElement_T<DecodingElement>;
	using ChangeList        = typename BJMM<config>::ChangeList;

	using ArgumentLimbType      = LogTypeTemplate<config.l1 + 1*config.l2>;

	using IndexType             = typename BJMM<config>::IndexType;
	using HMLoadType            = typename BJMM<config>::LoadType;
	using LoadType              = typename DecodingList::LoadType;

	using ValueContainerType    = typename BJMM<config>::ValueContainerType;
	using LabelContainerType    = typename BJMM<config>::LabelContainerType;
	using ValueContainerLimbType= typename BJMM<config>::ValueContainerLimbType;
	using LabelContainerLimbType= typename BJMM<config>::LabelContainerLimbType;
	typedef typename DecodingList::ElementType ElementType;
	using Extractor = WindowExtractor<DecodingLabel, ArgumentLimbType>;

	// TODO explain
	constexpr static uint32_t l2 = (config.l - config.l1)/config.IM_nr_views;

	// List sizes
	constexpr static double iFactor = 2.;
	constexpr static size_t liLsize = size_t(iFactor*(BJMM<config>::lsize1 * BJMM<config>::lsize1 >> BJMM<config>::l1));
	constexpr static size_t riLsize = size_t(iFactor*(BJMM<config>::lsize2 * BJMM<config>::lsize2 >> BJMM<config>::l1));

	// Intermediate target
	DecodingLabel target, iT;

	// Local variables replacing the lists from the base class.
	DecodingList iLl{liLsize, this->threads, liLsize/this->threads},
			     iLr{riLsize, this->threads, riLsize/this->threads};

	// TODO optimise we only need the values in these two lists
	// We need backup lists to copy in the sorted value list
	DecodingList bwL1{this->lsize1, this->threads, this->thread_size_lists1},
				 bwL2{this->lsize2, this->threads, this->thread_size_lists2},
				 wL1{this->lsize1, this->threads, this->thread_size_lists1},
				 wL2{this->lsize2, this->threads, this->thread_size_lists2};

	constexpr static uint32_t b10 = 0;
	constexpr static uint32_t b11 = 0 + config.number_bucket1;
	constexpr static uint32_t b12 = 0 + config.l2;

	//
	uint64_t loops = 0;

	// this is maybe a little bit confusing. The thing is: to increase the capability of the hashmap the data in a l2
	// window is shifted down to the bit position 0. This allows us to use l2 window size of up to 64.
	static ArgumentLimbType Hash(uint64_t a) {
		return 0;
	}
	constexpr static ConfigParallelBucketSort chm{b10, b11, b12, config.size_bucket1, uint64_t(1) << config.number_bucket1, config.nr_threads, 1, 0, 0, 0, config.IM_nr_views};
	using HMType = ParallelBucketSort<chm, DecodingList, ArgumentLimbType, IndexType, &Hash>;
	HMType *hm;

public:
	BJMMHybridTree(mzd_t *e, const mzd_t *const s, const mzd_t *const A, const uint32_t ext_tid = 0)
			noexcept : BJMM<config>(e, s, A, ext_tid, false) {
		hm = new ParallelBucketSort<chm, DecodingList, ArgumentLimbType, IndexType, &Hash>();
		this->template BJMM_prepare_generate_base_mitm2_with_chase2(bwL1, bwL2, this->cL1, this->cL2);
	};

	void __attribute__ ((noinline))
	check_final_list(const LabelContainerType &a, const ValueContainerType &b, const uint32_t tid) noexcept {
		//#pragma omp critical
		if (this->not_found) {
			//#pragma omp atomic write
			this->not_found = false;

			for (uint32_t j = 0; j < config.n ; ++j) {
				uint32_t bit;
				if (j < (config.n - config.k - config.l)) {
					bit = a.data(j);
				} else {
					bit = b.data(j - (config.n - config.k - config.l));
				}

				mzd_write_bit(this->e, 0, this->permutation->values[j], bit);
			}

			std::cout << "loops:" << loops << ", tid: " << tid << "\n";
			std::cout << a << " label\n";
			std::cout << b << " value\n";
			mzd_print(this->e);
		}
	}

	// weg von den hashmaps zurück zu den bäumen.
	uint64_t run() noexcept {
		// count the loops we iterated
		loops = 0;

		// we have to reset this value, so its possible to rerun the algo more often
		this->not_found = true;
		while (this->not_found && loops < config.loops) {
			matrix_create_random_permutation(this->work_matrix_H, this->work_matrix_H_T, this->permutation);
			matrix_echelonize_partial_plusfix(this->work_matrix_H, config.m4ri_k, n - k - this->l, this->matrix_data, 0, n - k - this->l, 0, this->permutation);

			mzd_submatrix(this->H, this->work_matrix_H, 0, this->n - this->k - this->l, this->n - this->k, this->n - this->c + this->DOOM_nr);
			matrix_transpose(this->HT, this->H);

			this->target.data().column_from_m4ri(this->work_matrix_H, this->n-this->c);
			iT.random();

			Matrix_T<mzd_t *> HH((mzd_t *) this->H);
			// TODO enable
			//#pragma omp parallel default(none) shared(cout, wL1, wL2, this->cL1, this->cL2, HH, this->H, this->HT, iT, this->target, this->not_found, loops) num_threads(this->threads)
			{
				const uint32_t tid  = 0;//omp_get_thread_num();
				// const uint64_t iLThreadSize = (this->thread_size_lists2*this->thread_size_lists2) >> this->l1;
				const size_t s_tid1 = tid * this->thread_size_lists1;
				const size_t s_tid2 = tid * this->thread_size_lists2;
				const size_t e_tid1 = ((tid == this->threads - 1) ? wL1.size() : s_tid1 + this->thread_size_lists1);
				const size_t e_tid2 = ((tid == this->threads - 1) ? wL2.size() : s_tid2 + this->thread_size_lists2);
				//std::cout << "s_tid1: " << s_tid1 << ", s_tid2: " << s_tid2 << ", e_tid1: " << e_tid1 << ", e_tid2: " << e_tid2 << "\n";

				// load of the left and right intermediate list.
				LoadType loadl = 0, loadr = 0;
				HMLoadType HMload = 0;
				IndexType HMpos;
				LabelContainerType label;

				ElementType tmpe;
				LabelContainerType tmpl;

				// copy in the lists
				wL1 = bwL1;
				wL2 = bwL2;

				this->template BJMM_fill_decoding_lists<DecodingList>(wL1, wL2,
						this->cL1, this->cL2, this->HT, tid);
				ASSERT(this->template check_list<DecodingList>(wL1, HH, tid));
				ASSERT(this->template check_list<DecodingList>(wL2, HH, tid));

				// TODO think of something smarter to add the target/iT into the list
				auto add_target = [](DecodingList &L, const DecodingLabel &target) {
					for (size_t i = 0; i < L.size(); i++){
						DecodingLabel::add(L.data_label(i), L.data_label(i), target, n-k-config.l, n-k);
					}
				};

				// Lvl 1 sort
				auto sortf1 = [=](const DecodingElement &e) -> ArgumentLimbType {
					return Extractor::template extract<n-k-config.l1, n-k>(e.get_label_container_ptr());
				};

				// TODO put this in the fill decoding lists
				add_target(wL2, iT);
				OMP_BARRIER
				wL1.sort(tid, sortf1);
				OMP_BARRIER
				wL2.sort(tid, sortf1);
				OMP_BARRIER

				iLl.reset(tid);
				iLr.reset(tid);
				OMP_BARRIER

				//wL1.print(0, wL1.size());
				//wL2.print(0, wL2.size());

				// start the join between L1 and L2
				// constexpr uint32_t k_lower1 = n-k-config.l, k_upper1 = n-k-config.l+config.l1;
				constexpr uint32_t k_lower1 = n-k-config.l1, k_upper1 = n-k;
				auto lvl1_join = [&](DecodingList &out, DecodingList &L1, DecodingList &L2, const DecodingLabel &t, LoadType &load) {
					size_t i = s_tid1, j = s_tid2;

					// run the list merge routine from the HGJ paper
					while (i < e_tid1 && j < e_tid2) {
						if (L2[j].template is_greater<k_lower1, k_upper1>(L1[i])) {
								i++;
						} else if (L1[i].template is_greater<k_lower1, k_upper1>(L2[j])) {
							j++;
						} else {
							// std::cout<<L1[i]<<L2[j];
							size_t i_max, j_max;
							// if elements are equal find max index in each list, such that they remain equal
							for (i_max = i + 1; i_max < e_tid1 && L1[i].template is_equal<k_lower1, k_upper1>(L1[i_max]); i_max++) {}
							for (j_max = j + 1; j_max < e_tid2 && L2[j].template is_equal<k_lower1, k_upper1>(L2[j_max]); j_max++) {}

							size_t jprev = j;

							for (; i < i_max; ++i) {
								for (j = jprev; j < j_max; ++j) {
									out.add_and_append(L1[i], L2[j], load, tid);
									// std::cout << out[load-1] << "\n" << L1[i] << "\n" <<  L2[j] << "\n" << std::flush;

									// TODO this is awekward. Somehoe we find colums in the matrix which are the same
									if constexpr(p == 1) {
										if (unlikely(out[load - 1].is_zero())) {
											load -= 1;
											// assert(false);
										}
									}
								}
							}
						}
					}
				};

				// first do the join between L1 L2
				lvl1_join(iLl, wL1, wL2, iT, loadl);
				OMP_BARRIER
				ASSERT(this->template check_list<DecodingList>(iLl, HH, tid, 0, n-k-config.l, loadl) OMP_BARRIER);

				// second join
				add_target(wL2, target);
				OMP_BARRIER
				wL2.sort(tid, sortf1);
				OMP_BARRIER
				lvl1_join(iLr, wL1, wL2, target, loadr);
				OMP_BARRIER

				ASSERT(this->template check_list<DecodingList>(iLr, HH, tid, 0, n-k-config.l, loadr) OMP_BARRIER);

				// And do the final merge.
				for (uint32_t m = 0; m < config.IM_nr_views; ++m) {
					constexpr uint64_t basewin = n-k-config.l;
					const uint64_t k_lower2 = basewin + m*l2 , k_upper2 = basewin + (m+1)*l2;

					// now the fun part begins. First rest the internals of the hashmap
					hm->reset(tid);
					OMP_BARRIER

					// NOTE this hashing only works, because we remembered the number of elements each thread needs to hash.
					// Otherwise, we would have zero elements in the hashmap.
					hm->hash(iLl, loadl, tid, [k_lower2, k_upper2](const DecodingLabel&e) -> ArgumentLimbType {
						return Extractor::extract(e, k_lower2, k_upper2);
					});

					OMP_BARRIER

					// now sort the hashmap. Note this is only needed if we do not hash on the full length
					hm->sort(tid);
					OMP_BARRIER

					ASSERT(hm->check_sorted());
					//hm->print();

					OMP_BARRIER
					uint64_t i = 0, j = s_tid2;
					for (; j < s_tid2+loadr; j++) {
						ArgumentLimbType data = Extractor::extract(iLr[j].get_label(), k_lower2, k_upper2);
						ASSERT(hm->check_label(data, iLr, j, k_lower2, k_upper2));

						HMpos = hm->find(data, HMload);
						while (HMpos < HMload) {
							i = hm->__buckets[HMpos].second[0];
							HMpos += 1;

							// TODO these additions can be optimized
							LabelContainerType::add(label, iLl[i].get_label().data(), iLr[j].get_label().data());
							uint32_t weight = LabelContainerType::add_weight(label, label, target.data(), 0, n-k-config.l);

							if (unlikely(weight <= config.weight_threshhold)) {
								ValueContainerType value; value.zero();
								ValueContainerType::add(value, iLl[i].get_value().data(), iLr[j].get_value().data());
								// std::cout << iLl[i].get_value().data() << "\n" << iLr[j].get_value() << "\n";
								check_final_list(label, value, tid);
							} // found the solution
						} // while lvl2 matches
					} // for every element in the intermediate list
				} // for every hashmap
			} // end pragma omp parallel

			// print loop information
			if (unlikely((loops % config.print_loops) == 0)) {
				periodic_info();
			}

			//  update the global loop counter
			OUTER_MULTITHREADED_WRITE(
			if ((unlikely(loops % config.exit_loops) == 0)) {
				if (finished.load()) {
					return loops;
				}
			})

			loops += 1;
		} // end not found loop


		return loops;
	}

	void __attribute__ ((noinline)) periodic_info() noexcept {
		std::cout << "BJMMF: tid:" << omp_get_thread_num() << ", loops: " << loops;
		std::cout << ", log(inner_loops): " << this->LogLoops(n) << ", inner_loops: " << this->Loops(n, 0) << "\n";
		std::cout << "iFactor: " << iFactor << ", liLsize: " << liLsize << ", riLsize: " << riLsize << "\n";
		hm->print();
	}
};

#endif //SMALLSECRETLWE_BJMM_HYBRIDTREE_H
