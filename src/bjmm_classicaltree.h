#ifndef SMALLSECRETLWE_BJMM_CLASSICALTREE_H
#define SMALLSECRETLWE_BJMM_CLASSICALTREE_H

// internal includes
#include "bjmm.h"
#include "list_fill.h"
#include "tree.h"

/// Usage:
/// 	static constexpr ConfigBJMM config(n, k, w, 1, l1, l1, 80, 20, l1, 1, w-4, NUMBER_THREADS, false, false, false, 0, false, l2, IMv);
///		BJMMClassicalTree<config> bjmm(ee, ss, A);
///		bjmm.MMT2_NormalTree();
/// \tparam config
template<const ConfigBJMM &config>
class BJMMClassicalTree : public BJMM<config> {
	// More or less the same as the class `BJMM` but replaces the internal hashmap as the main datastructure
	// with the classical tree approach. Meaning we do the good old list merging (see MMT paper) without
	// hashmaps

	using DecodingValue     = typename BJMM<config>::DecodingValue;
	using DecodingLabel     = Label_T<BinaryContainer<n - k>>; // TODO be carefully because of the appending of the parity row.
	using DecodingMatrix    = typename BJMM<config>::DecodingMatrix ;
	using DecodingElement   = typename BJMM<config>::DecodingElement;
	using DecodingList      = Parallel_List_FullElement_T<DecodingElement>;

	typedef typename DecodingList::ValueContainerType ValueContainerType;
	typedef typename DecodingList::LabelContainerType LabelContainerType;
	typedef typename DecodingList::ValueContainerLimbType ValueContainerLimbType;
	typedef typename DecodingList::LabelContainerLimbType LabelContainerLimbType;
	typedef typename DecodingList::ElementType ElementType;

	using ArgumentLimbType      = LogTypeTemplate<config.l>;
	using Extractor = WindowExtractor<DecodingLabel, ArgumentLimbType>;

	// intermediate target
	LabelContainerType iT;

	// TODO how to make sure that the list in the baseclass are not allocated?
	// Local variables replacing the lists from the base class.
	DecodingList iL{(this->lsize2 * this->lsize2) >>(this->l1*2), this->threads,(this->thread_size_lists2*this->thread_size_lists2) >> (this->l1*2)},
			bwL1{this->lsize1, this->threads, this->thread_size_lists1},
			bwL2{this->lsize2, this->threads, this->thread_size_lists2},
			wL1{this->lsize1, this->threads, this->thread_size_lists1},
			wL2{this->lsize2, this->threads, this->thread_size_lists2};

public:
	BJMMClassicalTree(mzd_t *e, const mzd_t *const s, const mzd_t *const A,
					  const uint32_t ext_tid = 0)
			: BJMM<config>(e, s, A, ext_tid, false) {
		this->template BJMM_prepare_generate_base_mitm2_with_chase2(bwL1, bwL2, this->cL1, this->cL2);

	};

private:

	/// this function is only called if we found the solution. Its reconstruction the correct solution vector given
	/// only the two elements from the intermediate lists
	/// \param a element of the left intermediate list
	/// \param b element if the right indermediate list
	/// \param tid	thread id which found the element. actually useless.
	void __attribute__ ((noinline))
	check_final_list(const DecodingElement &a, const DecodingElement &b, const uint32_t tid) {
		//#pragma omp critical
		if (this->not_found) {
			//#pragma omp atomic write
			this->not_found = false;

			DecodingElement rebuild;
			DecodingElement::add(rebuild, a, b);

			std::cout << a;
			std::cout << b;
			std::cout << rebuild << "rebuild\n";
			std::cout << this->target << "target\n";

			for (int j = 0; j < config.n ; ++j) {
				uint32_t bit;
				if (j < (config.n - config.k - config.l)) {
					bit = a.get_label(j);
				} else {
					bit = rebuild.get_value(j - (config.n - config.k - config.l));
				}

				mzd_write_bit(this->e, 0, this->permutation->values[j], bit);
			}

			mzd_print(this->e);
		}
	}

public:
	// executes the algorithm and returns the number of loops needed to find the solution
	uint64_t run() {
		// count the loops we iterated
		uint64_t loops = 0;

		// we have to reset this value, so its possible to rerun the algo more often
		this->not_found = true;
		while (this->not_found && loops < config.loops) {
			loops += 1;
			matrix_create_random_permutation(this->work_matrix_H, this->work_matrix_H_T, this->permutation);
			matrix_echelonize_partial_plusfix(this->work_matrix_H, config.m4ri_k, n - k - this->l, this->matrix_data, 0, n - k - this->l, 0, this->permutation);

			mzd_submatrix(this->H, this->work_matrix_H, 0, this->n - this->k - this->l, this->n - this->k, this->n - this->c + this->DOOM_nr);
			matrix_transpose(this->HT, this->H);

			Matrix_T<mzd_t *> HH((mzd_t *) this->H);
			if constexpr(!config.DOOM) {
				this->target.column_from_m4ri(this->work_matrix_H, this->n-this->c-config.LOWWEIGHT);
			}

			iT.random();
//#pragma omp parallel default(none) shared(cout, wL1, wL2, this->cL1, this->cL2, HH, this->H, this->HT, iT, this->target, this->not_found, loops) num_threads(this->threads)
			{
				const uint32_t tid          = omp_get_thread_num();
				const uint64_t iLThreadSize = (this->thread_size_lists2*this->thread_size_lists2) >> this->l1;
				const uint64_t s_tid1 = tid * this->thread_size_lists1;
				const uint64_t s_tid2 = tid * this->thread_size_lists2;
				const uint64_t e_tid1 = ((tid == this->threads - 1) ? wL1.size() : s_tid1 + this->thread_size_lists1);
				const uint64_t e_tid2 = ((tid == this->threads - 1) ? wL2.size() : s_tid2 + this->thread_size_lists2);
				//std::cout << "s_tid1: " << s_tid1 << ", s_tid2: " << s_tid2 << ", e_tid1: " << e_tid1 << ", e_tid2: " << e_tid2 << "\n";

				// instead of letting the list track its load, we do it.
				uint64_t load1 = 0;

				ElementType tmpe;
				LabelContainerType tmpl;

				// uff named them the other waz around
				constexpr uint32_t  k_lower_lvl2 = config.n-config.k-config.l,
									k_upper_lvl2 = k_lower_lvl2 + config.l1,
									k_lower_lvl1 = k_upper_lvl2,
									k_upper_lvl1 = config.n - config.k;

				// TODO think of something smarter to add the target/iT into the list
				auto add_target = [](DecodingList &L, const LabelContainerType &target) {
					for (size_t i = 0; i < L.size(); i++){
						LabelContainerType::add(L.data_label(i).data(), L.data_label(i).data(), target, n-k-l, n-k);
					}
				};

				auto sortf1 = [=](const DecodingElement &e) -> ArgumentLimbType {
					return Extractor::template extract<n-k-config.l1, n-k>(e.get_label_container_ptr());
				};

				wL1 = bwL1;
				wL2 = bwL2;

				this->template BJMM_fill_decoding_lists<DecodingList>(wL1, wL2, this->cL1, this->cL2, this->HT, tid);
				OMP_BARRIER
				ASSERT(this->check_list(wL1, HH, tid));
				ASSERT(this->check_list(wL2, HH, tid));
				OMP_BARRIER

				add_target(wL2, iT);
				OMP_BARRIER

				wL1.sort(tid, sortf1);
				OMP_BARRIER
				wL2.sort(tid, sortf1);
				OMP_BARRIER
				size_t i= s_tid1,j = s_tid2;

				// start the tree
				while ((i < e_tid1) && (j < e_tid2) && (load1 < iLThreadSize)) {
					if (wL2[j].is_greater(wL1[i], k_lower_lvl2, k_upper_lvl2))
						i++;
					else if (wL1[i].is_greater(wL2[j], k_lower_lvl2, k_upper_lvl2))
						j++;
					else {
						uint64_t i_max, j_max;
						// if elements are equal find max index in each list, such that they remain equal
						for (i_max = i + 1; i_max < e_tid1 && wL1[i].is_equal(wL1[i_max], k_lower_lvl2, k_upper_lvl2); i_max++) {}
						for (j_max = j + 1; j_max < e_tid2 && wL2[j].is_equal(wL2[j_max], k_lower_lvl2, k_upper_lvl2); j_max++) {}

						int jprev = j;

						for (; i < i_max; ++i) {
							for (j = jprev; j < j_max; ++j) {
								iL.add_and_append(wL1[i], wL2[j], load1, tid);

								// TODO this is awekward. Somehoe we find colums in the matrix which are the same
								if constexpr(p == 1) {
									if (unlikely(iL[load1 - 1].is_zero())) {
										load1 -= 1;
									}
								}

							}
						}
					}
				}

				//std::cout << "load1: " << load1 << "\n";
				ASSERT(this->check_list(iL, HH, tid, 0, config.n-config.k-config.l, load1));
				OMP_BARRIER

				iL.sort(tid, sortf1, load1);
				OMP_BARRIER
				i = s_tid1; j = s_tid2;

				add_target(wL2, this->target);
				wL2.sort_level(k_lower_lvl2, k_upper_lvl2, tid);

				while (i < e_tid1 && j < e_tid2) {
					if (wL2[j].is_greater(wL1[i], k_lower_lvl2, k_upper_lvl2))
						i++;
					else if (wL1[i].is_greater(wL2[j], k_lower_lvl2, k_upper_lvl2))
						j++;
					else {
						size_t i_max, j_max;
						for (i_max = i + 1; i_max < e_tid1 && wL1[i].is_equal(wL1[i_max], k_lower_lvl2, k_upper_lvl2); i_max++) {}
						for (j_max = j + 1; j_max < e_tid2 && wL2[j].is_equal(wL2[j_max], k_lower_lvl2, k_upper_lvl2); j_max++) {}

						size_t jprev = j;

						// we have found equal elements. But this time we dont have to save the result.
						// Rather we stream join everything up to the final solution.
						for (; i < i_max; ++i) {
							for (j = jprev; j < j_max; ++j) {
								ElementType::add(tmpe, wL1[i], wL2[j]);
								std::pair<size_t, size_t> boundaries = iL.search_boundaries(tmpe, k_lower_lvl1, k_upper_lvl1, tid);
								// std::cout << "boundaries: " << boundaries.first << " " << boundaries.second << "\n";

								// finished or nothing found?
								if (boundaries.first == boundaries.second) {
									continue;
								}

								//std::cout << tmpe;
								//iL.print(boundaries.first, boundaries.second);

								for (size_t t = boundaries.first; t < boundaries.second; ++t) {
									uint32_t weight = LabelContainerType::add_weight(tmpl, tmpe.get_label_container(), iL[t].get_label_container());

									//std::cout << tmpl << "\n";

									if (weight <= config.weight_threshhold) {
										check_final_list(tmpe, iL[t], tid);
									}
								}
							}
						}
					}
				}
			}
		} // end pragma imp parallel


		return loops;
	}
};

#endif //SMALLSECRETLWE_BJMM_CLASSICALTREE_H
