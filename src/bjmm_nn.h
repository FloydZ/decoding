#ifndef SMALLSECRETLWE_BJMM_NN_H
#define SMALLSECRETLWE_BJMM_NN_H

#include "bjmm.h"

template<const ConfigBJMM &config>
class BJMMNN : public BJMM<config> {
public:
	//         L1            L2           L3           L4
	//
	//      ┌──┬──────┐  ┌──┬──────┐  ┌─────────┐  ┌─────────┐
	//      │  │      │  │  │      │  │         │  │         │
	//      │  │      │  │  │      │  │         │  │         │
	//      │  │      │  │  │      │  │         │  │         │
	//      │  │      │  │  │      │  │         │  │         │
	//      │  │      │  │  │      │  │         │  │         │
	//      └──┴──────┘  └──┴──┬───┘  └────┬────┘  └────┬────┘
	//           ▼             │           │            │
	//        l1 l2  l2        │           │     l1     │
	//      ┌──┬──┬───┐        │           └─────┬──────┘
	// HM1  │$$│$$│$$$│        │                 │
	//      └──┴─┬┴───┘        │                 │
	//           │             │                 │
	//           │       l1    │                 │
	//           └──────┬──────┘                 │
	//                  │                        │
	//            ┌─────┴──────┐                 │
	//         l1 │l2  l2   l1 │ l2 l2           │
	//       ┌──┬─▼─┬──┐  ┌──┬─▼─┬──┐            │
	//  HM2  │00│$$$│00│  │00│000│$$│            │
	//       └──┴─┬─┴──┘  └──┴─┬─┴──┘            │
	//            │            │                 │
	//            │            └────────┬────────┤
	//            │                     │        │
	//            └─────────┬───────────┼────────┘
	//                      │           │
	//                      ▼           ▼
	//                   Direct      Direct
	//                   Check       Check
	constexpr static uint32_t IM_nr_views   = config.IM_nr_views;

	using ArgumentLimbType      = LogTypeTemplate<config.l1 + IM_nr_views*config.l2>;
	using IndexType             = typename BJMM<config>::IndexType;
	using LoadType              = typename BJMM<config>::LoadType;
	using DecodingList          = typename BJMM<config>::DecodingList;

	using ValueContainerType    = typename BJMM<config>::ValueContainerType;
	using LabelContainerType    = typename BJMM<config>::LabelContainerType;
	using ValueContainerLimbType= typename BJMM<config>::ValueContainerLimbType;
	using LabelContainerLimbType= typename BJMM<config>::LabelContainerLimbType;

	constexpr static uint32_t n             = config.n;
	constexpr static uint32_t k             = config.k - config.TrivialAppendRows;
	constexpr static uint32_t w             = config.w;
	constexpr static uint32_t p             = config.baselist_p;
	constexpr static uint32_t l             = config.l;
	constexpr static uint32_t l1            = config.l1;
	constexpr static uint32_t l2            = config.l2;
	constexpr static uint32_t c             = config.c;
	constexpr static uint32_t threads       = config.nr_threads;
	constexpr static uint32_t npos_size     = 4;

	// Indyk Motwani Stuff
	constexpr static ArgumentLimbType IM_InserMask = (ArgumentLimbType(1) << (l1+l2))-1;

	// Indyk Motwani adapted masks and offset to extract the l1+ IM_nr_views*l2 number of bits of each label.
	constexpr static uint32_t nkl           = n - k - l1 - (IM_nr_views*l2);
	constexpr static uint32_t lVCLTBits     = sizeof(ValueContainerLimbType)*8;
	constexpr static uint32_t loffset       = nkl / lVCLTBits;
	constexpr static uint32_t lshift        = nkl - (loffset * lVCLTBits);

	// masks and limbs of the weight calculations of the final label
	constexpr static uint32_t lupper               = LabelContainerType::round_down_to_limb(nkl - 1);
	constexpr static ValueContainerLimbType lumask = LabelContainerType::lower_mask2(nkl);

	constexpr static uint32_t b10 = 0;
	constexpr static uint32_t b11 = 0 + config.number_bucket1;
	constexpr static uint32_t b12 = 0 + l1;
	constexpr static uint32_t b20 = 0 + l1;
	constexpr static uint32_t b21 = 0 + l1 + config.number_bucket2;
	constexpr static uint32_t b22 = 0 + l1 + l2;

	constexpr static ConfigParallelBucketSort chm1{b10, b11, b12, config.size_bucket1, uint64_t(1) << config.number_bucket1, threads, 1, n - k - l1 - IM_nr_views*l2, l1 + IM_nr_views*l2, 0, IM_nr_views};
	constexpr static ConfigParallelBucketSort chm2{b20, b21, b22, config.size_bucket2, uint64_t(1) << config.number_bucket2, threads, 2, n - k - l1 - IM_nr_views*l2, l /*TODO why not l2?*/, 0, 0};


	// TODO use the more generic approach oneday
	template<const uint32_t l, const uint32_t h>
	static ArgumentLimbType Hash(uint64_t a) {
		return 0;
	}
	using HM1Type  = ParallelBucketSort<chm1,  DecodingList, ArgumentLimbType, IndexType, &Hash<b10, b11>>;
	using HM2Type = ParallelBucketSort<chm2, DecodingList, ArgumentLimbType, IndexType, &Hash<b20, b21>>;
	HM1Type *hm11;
	HM2Type *hm22[IM_nr_views];

	// TODO virtual function in base class and then let each subclass derive it?
	void print() {
		std::cout << "BJMMNN:\n";
		std::cout << "\tArgumentLimbType: " << sizeof(ArgumentLimbType)*8 << "B\n";
		std::cout << "\tIndexType: " << sizeof(IndexType)*8 << "B\n";
		std::cout << "\tLoadType: " << sizeof(LoadType)*8 << "B\n";
		std::cout << "\tDecodingList: " << sizeof(DecodingList)*8 << "B\n";
		std::cout << "\tValueContainerType: " << sizeof(ValueContainerType)*8 << "B\n";
		std::cout << "\tLabelContainerType:"  << sizeof(LabelContainerType)*8 << "B\n";
		std::cout << "\tValueContainerLimbType: " << sizeof(ValueContainerLimbType)*8 << "B\n";
		std::cout << "\tLabelContainerLimbType: " << sizeof(LabelContainerLimbType)*8 << "B\n";

		std::cout << "\tb10: " << b10 << "\n";
		std::cout << "\tb11: " << b11 << "\n";
		std::cout << "\tb12: " << b12 << "\n";
		std::cout << "\tb20: " << b20 << "\n";
		std::cout << "\tb21: " << b21 << "\n";
		std::cout << "\tb22: " << b22 << "\n";

		std::cout << "\tl1: " << l1 << "\n";
		std::cout << "\tl2: " << l2 << "\n";

		std::cout << "\tn-k: " << n-k << "\n";

		std::cout << "\tnkl: " << nkl << "\n";
		std::cout << "\tlVCLTBits: " << lVCLTBits << "\n";
		std::cout << "\tloffset: " << loffset << "\n";
		std::cout << "\tlshift: " << lshift << "\n";

	}

	BJMMNN(mzd_t *e, const mzd_t *const s, const mzd_t *const A, const uint32_t ext_tid = 0)
			: BJMM<config>(e, s, A, ext_tid, false) {
		ASSERT(config.nr_threads == 1);
		ASSERT(ext_tid == 0);
		static_assert(l == l1);
		static_assert(IM_nr_views > 0);
		static_assert(IM_nr_views * l2 <= 64);
		//static_assert((n-k) % 32 != 0); // BUG

		print();

		hm11 = new HM1Type();
		for (unsigned int i = 0; i < IM_nr_views; ++i) {
			hm22[i] = new HM2Type();
		}
	};

	~BJMMNN() {
		delete hm11;
		for (unsigned int i = 0; i < IM_nr_views; ++i) {
			delete hm22[i];
		}
	}

	uint64_t run() {
		if constexpr(config.c == 0) {
			return BJMMNNF();
		} else {
			return BJMMNNF_Outer();
		}
	}

	/// checks if somewhere in the given label is a zero window.
	/// \param l 	the label
	/// \param b0 	start value
	/// \param b1 	end value
	/// \return		true if [b0, b1] is not zero, else false.
	bool check_for_zero_window(const LabelContainerType &l, const uint32_t b0, const uint32_t b1) {
		ASSERT(b0 < b1);

		const uint32_t pos1 = b0 / lVCLTBits;
		const uint32_t pos2 = b1 / lVCLTBits;
		if (pos1 == pos2) {
			// they are in the same limb
			const ValueContainerLimbType mask = (~((ValueContainerLimbType(1) << (b0%lVCLTBits))-1)) &
			                                    ((ValueContainerLimbType(1) << (b1%lVCLTBits))-1);

			return l.ptr()[pos1] & mask;
		} else {
			const ValueContainerLimbType mask1 = ~((ValueContainerLimbType(1) << (b0%lVCLTBits))-1);
			const ValueContainerLimbType mask2 =   (ValueContainerLimbType(1) << (b1%lVCLTBits))-1;
			return (l.ptr()[pos1] & mask1) ^ (l.ptr()[pos2] & mask2);
		}
	}

	/// simplified version. In contrast to the original function this function extracts additional `IM_nr_views`
	/// 	different l2 views for the indyk motwani NN search.
	/// \param v3 label
	/// \param v1 intermediate target/target
	/// \return l1+nr_views*l2 bits
	inline ArgumentLimbType add_bjmm(ValueContainerLimbType *v3, ValueContainerLimbType const *v1) {
		constexpr static ValueContainerLimbType mask = nkl % lVCLTBits == 0 ? ValueContainerLimbType(-1) : ~((ValueContainerLimbType(1) << (nkl % lVCLTBits)) - 1);
		ArgumentLimbType ret;
		if constexpr ((nkl / lVCLTBits) == ((n - k - 1) / lVCLTBits)) {
			v3[loffset] ^= (v1[loffset] & mask);
			ret = v3[loffset] >> lshift; // IMPORTANT AUTO CAST HERE
		} else {
			v3[loffset]   ^= (v1[loffset] & mask);
			v3[loffset+1] ^=  v1[loffset+1];
//			// NOTE this must be generalised to 2*ValueContainerLimbType, if you want to be able to use l > 64
			__uint128_t data = v3[loffset];
			data            += (__uint128_t(v3[loffset+1]) << 64);

//			__uint128_t data = ((__uint128_t *)v3)[loffset];
			ret = data >> lshift;  // IMPORTANT AUTO CAST HERE
		}

		// Explanation see function below
		constexpr ArgumentLimbType mask2 = (ArgumentLimbType(1) << (l1 + (IM_nr_views*l2))) - 1;
		return ((ret >> (IM_nr_views*l2)) ^ (ret << l1)) & mask2;
	}

	// only extract
	inline ArgumentLimbType add_bjmm_oe(ValueContainerLimbType *v3) {
		ArgumentLimbType ret;
		if constexpr ((nkl / lVCLTBits) == ((n - k - 1) / lVCLTBits)) {
			// if the two limits are in the same limb
			ret = v3[loffset] >> lshift; // IMPORTANT AUTO CAST HERE
		} else {
			// the two limits are in two different limbs
			__uint128_t data = v3[loffset];
			data            += (__uint128_t(v3[loffset+1]) << 64);
			ret = data >> lshift; // IMPORTANT AUTO CAST HERE
		}

		// This is how the l windows currently lay in mem.
		// 0            l1+2*l2        64
		// [ l2 | l2 | l1 | 000000000 ]
		// this is how we want it
		// [ l1 | l2 | l2 | 000000000 ]
		// so, rotate the l windows.
		constexpr ArgumentLimbType mask = (ArgumentLimbType(1) << (l1 + (IM_nr_views*l2))) - 1;
		return ((ret >> (IM_nr_views*l2)) ^ (ret << l1)) & mask;
	}


	//
	constexpr static inline ArgumentLimbType IM_Filter(const ArgumentLimbType in, const uint16_t i) {
		// otherwise, makes the -1 no sense.
		ASSERT(i >= 1);
		constexpr ArgumentLimbType IM_l1_FilterMask = (ArgumentLimbType(1) << l1) - 1;
		constexpr ArgumentLimbType IM_l2_FilterMask = ((ArgumentLimbType(1) << (l2+l1)) - 1) ^ IM_l1_FilterMask;
		const ArgumentLimbType tmp1 = in >> (i*l2);
		const ArgumentLimbType tmp2 = tmp1 & IM_l2_FilterMask;
		return tmp2;
	}

	uint64_t BJMMNNF_Outer () {
		uint64_t loops = 0;
		uint64_t outer_loops = 0;
		while (this->not_found) {
			matrix_create_random_permutation(this->outer_matrix_H, this->outer_matrix_HT, this->c_permutation);
			if constexpr (config.DOOM) {
				mzd_submatrix(this->work_matrix_H, this->outer_matrix_H, 0, 0, n-k, n-c);
				mzd_append(this->work_matrix_H, this->DOOM_S, 0, n-c);
			} else {
				mzd_submatrix(this->work_matrix_H, this->outer_matrix_H, 0, 0, n-k, n-c);

				// write the syndrom
				for (int i = 0; i < n - k; ++i) {
					mzd_write_bit(this->work_matrix_H, i, n - c, mzd_read_bit(this->outer_matrix_H, i, n));
				}
			}


			// reset permutation
			for (int i = 0; i < n - c; ++i) {
				this->permutation->values[i] = i;
			}

			loops += BJMMNNF();
			outer_loops += 1;
		}

		std::cout << "Outer Loops: " << outer_loops << "\n";
		return loops;
	}

	uint64_t BJMMNNF() {
		uint64_t loops = 0;
		while (this->not_found && loops < config.loops MULTITHREADED_WRITE(&& !finished.load())) {
			restart:
			if constexpr(c != 0) {
				if (loops >= config.c_inner_loops)
					return loops;
			}

			matrix_create_random_permutation(this->work_matrix_H, this->work_matrix_H_T, this->permutation);
			matrix_echelonize_partial_plusfix(this->work_matrix_H, config.m4ri_k, n-k-l, this->matrix_data, 0, n-k-l, 0, this->permutation);

			// extract the sub-matrices.
			mzd_submatrix(this->H, this->work_matrix_H, config.TrivialAppendRows, n - k - l, n - k, n - c + this->DOOM_nr);
			matrix_transpose(this->HT, this->H);

			Matrix_T<mzd_t *> HH((mzd_t *) this->H);
			if constexpr(!config.DOOM) {
				this->target.column_from_m4ri(this->work_matrix_H, n - c);

				for (uint32_t i = 0; i < IM_nr_views; i++) {
					if (check_for_zero_window(this->target, l1 + i*l2, l1 + (i+1)*l2) == 0) {
						//std::cout << this->target << " target\n";
						goto restart;
					}
				}
			}


			// choose random intermediate target
			this->iT1.random();

			// TODO Parallelize
			{
				const uint32_t tid = 0; // NOTE: Inner multithreading not supported.
				const uint64_t b_tid =this->lsize2 / threads;
				const uint64_t s_tid = tid * b_tid;
				const uint64_t e_tid = ((tid == threads - 1) ? this->L2.size() : s_tid + b_tid);
				IndexType npos[npos_size] = {IndexType(s_tid)};
				IndexType pos1, pos2;
				LoadType load1 = 0, load2 = 0;
				ArgumentLimbType data, data1, data2;
				LabelContainerType label, label2, label3;

				uint64_t *Lptr = (uint64_t *) this->L2.data_label() + (s_tid * this->llimbs_a);

				for (uint32_t j = 0; j < npos_size; ++j) { npos[j] = s_tid; }

				this->BJMM_fill_decoding_lists(this->L1, this->L2, this->cL1, this->cL2, this->HT, tid);
				ASSERT(this->check_list(this->L1, HH, tid));
				ASSERT(this->check_list(this->L2, HH, tid));

				hm11->reset(tid);
				for(unsigned int i = 0; i < IM_nr_views; i++) {
					hm22[i]->reset(tid);
				}
				OMP_BARRIER
				hm11->hash(this->L1, tid);
				OMP_BARRIER
				hm11->sort(tid);
				OMP_BARRIER
				ASSERT(hm11->check_sorted());
				OMP_BARRIER

				for (; npos[1] < e_tid; ++npos[1], Lptr += this->llimbs_a) {
					if constexpr(config.DOOM) {
						data = this->add_bjmm_oe(Lptr);
					} else {
						data = this->add_bjmm(Lptr, this->iT1.ptr());
					}
					pos1 = hm11->find(data, load1);

					while (pos1 < load1) {
						npos[0] = hm11->__buckets[pos1].second[0];
						data1 = data ^ hm11->__buckets[pos1].first;

						ArgumentLimbType aaa = data1 & IM_InserMask;
						hm22[0]->insert(aaa, npos, tid);
						for(uint32_t IM_i = 1; IM_i < IM_nr_views; IM_i++) {
							ArgumentLimbType bbb = IM_Filter(data1, IM_i);
							hm22[IM_i]->insert(bbb, npos, tid);
						}

						pos1 += 1;
					}
				}

				// now sort the second hashmaps
				OMP_BARRIER
				for(uint32_t i = 0; i < IM_nr_views; i++) {
					hm22[i]->sort(tid);
					ASSERT(hm22[i]->check_sorted());
				}

				// reset the list
				OMP_BARRIER
				if constexpr(config.DOOM) {
					Lptr = (uint64_t *) this->DOOM_S_View->rows[0] + (s_tid * this->llimbs_a);
				} else {
					Lptr = (uint64_t *) this->L2.data_label() + (s_tid * this->llimbs_a);
				}

				const uint64_t upper_limit = config.DOOM ? n - k : e_tid;
				for (; (npos[3] < upper_limit); ++npos[3], Lptr += this->BaseList4Inc) {
					if constexpr(config.DOOM) {
						data = this->add_bjmm_oe(Lptr);
					} else {
						data = this->add_bjmm(Lptr, this->target.ptr());
					}
					if constexpr(!config.DOOM) {
						label.zero();
						LabelContainerType::add(label, label, this->L2.data_label(npos[3]).data());
						LabelContainerType::add(label, label, this->target, 0, nkl);
					}

					pos1 = hm11->find(data, load1);

					// if a solution exists, we know that every element in this bucket given is a solution
					while (pos1 < load1) {
						npos[2] = hm11->__buckets[pos1].second[0];
						data1 = data ^ hm11->__buckets[pos1].first;
						ASSERT((data1 & ((uint64_t(1) << l1)-1)) == 0);

						pos1 += 1;

						for(unsigned int IM_i = 0; IM_i < IM_nr_views ; IM_i++) {
							//[ l1 | l2  | l2 | 000000]
							if (IM_i == 0) {
								data2 = data1 & IM_InserMask;
							} else {
								// if we are at the second or higher hashmap we have to extract the values differently.
								// Specially we need to rotate the l2 and l1 part.
								data2 = IM_Filter(data1, IM_i);
							}

							pos2 = hm22[IM_i]->find(data2, load2);
							if (pos2 != IndexType(-1)) {
								if constexpr(!config.DOOM) {
									LabelContainerType::add(label2, label, this->L1.data_label(npos[2]).data());
								} else {
									LabelContainerType::add(label2.data().data(), this->L1.data_label(npos[2]).data().data().data(), Lptr);
								}
							}

							while (pos2 < load2) {
								npos[0] = hm22[IM_i]->__buckets[pos2].second[0];
								npos[1] = hm22[IM_i]->__buckets[pos2].second[1];
								ASSERT((data2 ^ hm22[IM_i]->__buckets[pos2].first) == 0);

								pos2 += 1;

								LabelContainerType::add(label3, this->L2.data_label(npos[1]).data(), this->L1.data_label(npos[0]).data());

								uint32_t weight;
								if constexpr(!config.DOOM) {
									weight = LabelContainerType::template add_only_upper_weight_partly_withoutasm<lupper, lumask>(label3, label3, label2);
//									std::cout << label3 << " label3\n";
//									std::cout << this->target << " this->target\n";
//									std::cout << this->L1.data_label(npos[0]).data() << " npos[0]:" << npos[0] << "\n";
//									std::cout << this->L2.data_label(npos[1]).data() << " npos[1]:" << npos[1] << "\n";
//									std::cout << this->L1.data_label(npos[2]).data() << " npos[2]:" << npos[2] << "\n";
//									std::cout << this->L2.data_label(npos[3]).data() << " npos[3]:" << npos[3] << "\n";
//									std::cout << label2 << " label2\n";
//									std::cout << label3 << " label3\n";

								} else {
									weight = LabelContainerType::add_weight(label3.data().data(), label3.data().data(),label2.data().data());
								}

								if (likely(weight > config.weight_threshhold)) {
									continue;
								}

								this->check_final_list(label3, npos, weight, IM_i);
							}
						}
					}
				}

				// print final loop information.
				print_info(loops);

				// if another thread found the solutionquir
				OUTER_MULTITHREADED_WRITE(
				if (finished.load()) {
						return loops;
				})
			}

			loops += 1;
		}


		hm11->print();
		for (uint32_t i = 0; i < IM_nr_views; ++i) {
			hm22[i]->print();
		}

		double offset_loops = (double(loops) / double(this->Loops())) * double(100);
		std::cout << "loops/expected: " << loops << "/" << this->Loops() << " " << offset_loops << "%\n" << std::flush;
		return loops;
	}


	/// prints current loops information like: Hasmap usage,..
	/// \param loops
	void print_info(uint64_t loops) {
#ifndef NO_LOGGING
		if ((loops % config.print_loops) == 0) {
			std::cout << "BJMMF: tid:" << omp_get_thread_num() << ", loops: " << loops << "\n";
			std::cout << "log(inner_loops): " << this->LogLoops(n, c) << ", inner_loops: " << this->Loops(n, c) ;
			if constexpr(c != 0) {
				std::cout << ", log(outer_loops): " << this->LogOuterLoops(n, c) << ", outer_loops: " << this->OuterLoops(n, c) ;
			}
			std::cout << "\n";
#ifndef CHALLENGE
			hm11->print();
			for(unsigned int i = 0; i < IM_nr_views; i++){
				hm22[i]->print();
			}
			std::cout << "\n";
#endif
		}
#endif
	}
};


#endif //SMALLSECRETLWE_BJMM_NN_H
