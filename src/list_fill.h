#ifndef SMALLSECRETLWE_LIST_FILL_H
#define SMALLSECRETLWE_LIST_FILL_H

// internal includes
#include "helper.h"
#include "matrix.h"

// external includes
#include "m4ri.h"

template<class Label>
void xor_helper(Label &a, const word *b, const uint64_t limbs) {
	ASSERT(a.data().limbs() == limbs);
	unsigned int i = 0;

	for (; i < limbs; ++i) {
		a.data().data()[i] ^= b[i];
	}
};

/// IMPORTANT: function assumes that H is already transposed.
///		Assumes that the two input Lists are generated via a ChaseSequence/Lexicogrpahical Sequence or any other
///		sequence where the following element is differs in only two bit positions from the previous one.
/// 	This changelist of bit positions __MUST__ be described int `v`. Also it will clear all Labels within the two
///		input lists.
///		One must pass the diff list of the elements, because we are not calculating the Labels via a matrix-vector-multiplication
///		but rather a 'cleverly' chosen XOR chain.
///		The two output lists will be generated as follows:
///			- Copy the first Value from the two input lists into the two output lists.
///			- Recalculate the Label of this Value via a two XOR with the correctly chosen row in `H_T`
///			- Repeat for every Element within the input lists.
///		The two input Lists can be generated via
///				generate_decoding_lists<k + l>(l1, l2, changelist, base_list_p);
/// USAGE:
/// 	fill_decoding_lists<n-k-l>(tree[0], tree[1], l1, l2, changelist, working_s_T, H_prime_T);
///		see `decoding.h` foor a big complete example.
/// \param l1		Output List1
/// \param l2		Output List2
/// \param in_l1	const Input List1 of the following form:
/// 	    <-                 n' = k+l                    ->
/// 	     ___________p/2_______n'/2______________________
/// 	L1: |___________e_1_________|___________0___________|
/// \param in_l2	const Input List2 of the following form:
/// 	     _____________________n'/2__________p/2_________
/// 	L2: |__________0____________|___________e_2_________|
/// \param v		const ChangeList of the input lists
/// \param H_T		const ISD Submatrix. __MUST__ be transposed.
template<const uint64_t n_k_l, const uint64_t baselist_p, class List>
void fill_decoding_lists(List &l1, List &l2, const List &in_l1, const List &in_l2,
                         const std::vector<std::pair<uint64_t, uint64_t>> &v,
                         const mzd_t * syn, const mzd_t *H_T) {
	ASSERT( l1.size() == l2.size() && l1.size() == v.size() && l1.size() > 0);
	typedef typename List::LabelType Label;

	Matrix_T<mzd_t *> H{(mzd_t *)H_T};

	const uint32_t limbs = H_T->width;
	const uint32_t len = l1[0].get_value_size()/2;

	// Set the first value in the two output lists.
	l1[0].get_value() = in_l1[0].get_value();
	l2[0].get_value() = in_l2[0].get_value();

	// The first element we need to do a fill matrix vector multiplication.
	l1[0].get_label().zero();
	l2[0].get_label().zero();
	for (int i = 0; i < baselist_p; ++i) {
		xor_helper<Label>(l1[0].get_label(), H_T->rows[i], limbs);
		xor_helper<Label>(l2[0].get_label(), H_T->rows[i + len], limbs);
	}

	for (int i = 1; i < l1.size(); ++i) {
		// because H_ is already transposed we can xor up the changes on the rows
		auto pos = v[i-1];
		ASSERT(pos.first+len < H.get_rows() && pos.second+len < H.get_rows());

		// update the list element label via the change list.
		l1[i].get_label() = l1[i-1].get_label();
		l2[i].get_label() = l2[i-1].get_label();

		xor_helper<Label>(l1[i].get_label(), H_T->rows[pos.first], limbs);
		xor_helper<Label>(l1[i].get_label(), H_T->rows[pos.second], limbs);
		xor_helper<Label>(l2[i].get_label(), H_T->rows[pos.first + len], limbs);
		xor_helper<Label>(l2[i].get_label(), H_T->rows[pos.second + len], limbs);

		l1[i].get_value() = in_l1[i].get_value();
		l2[i].get_value() = in_l2[i].get_value();
	}
}


/// IMPORTANT
///		- works for inplace lists
///		- H_T __MUST__ be already transposed.
///		The two baselists __MUST__ be generated by
///	USAGE:
///		mceliece_d2_fill_decoding_lists<k, l, l1, w1, l2, w2, epsilon>(List1, List2, changelist1, changelist2, working_s_T, H_prime_T);
///		see `mceliece.h` for a big complete example.
///
/// \tparam k			McEliece Dimension
/// \tparam l			Dumer Window
/// \tparam l1				l1 + l2 = l
/// \tparam w1			weight on the l1 window (on l1/2 coordinates)
/// \tparam l2			zero window.
/// \tparam w2			weight on the k window	(on k/2 coordinates.)
/// \tparam epsilon		also named `e`. Additional overlap of the
/// \tparam T
/// \param List1		must be prepared as follows:
/// 			[	w1   |	0	|	0		|	w2	|	0  ]
///				<l1/2+e ><l1/2-e><    l2    >< k/2+e>< k/2 >
/// \param List2
/// 			[	0   |	w1	|	0		|	0	|	w2 ]
///				<l1/2-e><l1/2+e><    l2    >< k/2-e	>< k/2 >
/// \param v1
/// \param v2
/// \param syn			can be nullptr. Is only used for debugging.
/// \param H_T			McEliece current working submatrix of the input Matrix H.
///						should have the dimension (k+l) x (n-k). So __MUST__ be transposed.
///	\param start_offset	specifies from which point the lists `List1` and `List2` should be filled.
template<const uint64_t n, const uint64_t k, const uint64_t l, const uint64_t l1, const uint64_t w1, const uint64_t l2, const uint64_t w2, const uint64_t epsilon, class List>
void mceliece_d2_fill_decoding_lists(List &List1, List &List2,
                                     const std::vector<std::pair<uint64_t, uint64_t>> &v11, const std::vector<std::pair<uint64_t, uint64_t>> &v12,
                                     const std::vector<std::pair<uint64_t, uint64_t>> &v21, const std::vector<std::pair<uint64_t, uint64_t>> &v22,
                                     const mzd_t *H_T, const uint64_t start_offset1=0, const uint64_t start_offset2=0) {
	ASSERT(List1.size() > start_offset1 && List2.size() > start_offset2);

	// Polymorphism rules.
	typedef typename List::LabelType Label;
	Matrix_T<mzd_t *> H{(mzd_t *)H_T};

	constexpr uint64_t nkl          = n-k-l;
	const uint32_t limbs            = H_T->width;
	constexpr uint64_t first_half   = l1/2;
	constexpr uint64_t second_half  = k/2;

	// prepare the first element
	List1[start_offset1].get_label().zero();
	List2[start_offset2].get_label().zero();
	for (int i = nkl; i < nkl+w1; ++i) {
		List1[start_offset1].get_label().data().write_bit(i, 1);
		// this special case is needed for the dumer implementation when the baselists are instanciated over the full length
		if constexpr (first_half != 0)
			List2[start_offset2].get_label().data().write_bit(i+first_half, 1);
		else
			xor_helper<Label>(List2[start_offset2].get_label(), H_T->rows[i-nkl+(l+k)/2], limbs);
	}

	for (int i = 0; i < w2; ++i) {
		xor_helper<Label>(List1[start_offset1].get_label(), H_T->rows[l+i], limbs);
		xor_helper<Label>(List2[start_offset2].get_label(), H_T->rows[l-epsilon+second_half+i], limbs);
	}

	//ASSERT(v11.size() <= v21.size());
	const uint64_t size1  = MAX(v11.size(),  v21.size());
	const uint64_t size2  = MAX(v12.size(),  v22.size());

	// make sure that out list is at least bigger than the difflists.
	ASSERT(List1.get_load() >= size1*v12.size() && List2.get_load() >= size1* v22.size());

	uint64_t counter1 = 1 + start_offset1;
	uint64_t counter2 = 1 + start_offset2;

	for(uint64_t i = 0; i < size1+1; i++) {
		// because H_T is already transposed we can xor up the changes on the rows
		for(uint64_t j = 0; j < size2; j++) {
			if (j < v12.size() && i < v11.size()+1) {
				List1[counter1].get_label() = List1[counter1-1].get_label();

				auto pos = v12[j];
				// std::cout << pos.first << " " << pos.second << "\n";
				ASSERT(pos.first < H.get_rows() && pos.second < H.get_rows());
				// we do not need to add `l1` and `win`, because the pos counter already have them internally added
				xor_helper<Label>(List1[counter1].get_label(), H_T->rows[pos.first], limbs);
				xor_helper<Label>(List1[counter1].get_label(), H_T->rows[pos.second], limbs);
				ASSERT(counter1 <= List1.size());

				counter1 += 1;
			}

			if (j <  v22.size()){
				List2[counter2].get_label() = List2[counter2-1].get_label();

				auto pos = v22[j];
				ASSERT(pos.first < H.get_rows() && pos.second < H.get_rows());

				xor_helper<Label>(List2[counter2].get_label(), H_T->rows[pos.first], limbs);
				xor_helper<Label>(List2[counter2].get_label(), H_T->rows[pos.second], limbs);
				ASSERT(counter2 <= List2.size());

				counter2 += 1;
			}
		}

		// are we in the last round? If so exit. Note that the outer loop is limited by `size1+1`. This is because we
		// need to tun the inner loop one last time.
		if (i == size1)
			break;

		// reset
		if (i < v11.size()) {
			List1[counter1].get_label() = List1[counter1 - v12.size() - 1].get_label();

			auto pos = v11[i];
			ASSERT(pos.first < H.get_rows() && pos.second < H.get_rows());
			xor_helper<Label>(List1[counter1].get_label(), H_T->rows[pos.first ], limbs);
			xor_helper<Label>(List1[counter1].get_label(), H_T->rows[pos.second], limbs);

			counter1 += 1;
		}

		if (i < v21.size()) {
			List2[counter2].get_label() = List2[counter2 - v22.size() - 1].get_label();

			auto pos = v21[i];
			ASSERT(pos.first < H.get_rows() && pos.second < H.get_rows());
			xor_helper<Label>(List2[counter2].get_label(), H_T->rows[pos.first ], limbs);
			xor_helper<Label>(List2[counter2].get_label(), H_T->rows[pos.second], limbs);

			counter2 += 1;
		}
	}
}

template<const uint64_t n, const uint64_t k, const uint64_t l, const uint64_t l1, const uint64_t w1, const uint64_t l2, const uint64_t w2, const uint64_t epsilon, class List>
void mceliece_d2_fill_decoding_lists(List &List1,
                                     const std::vector<std::pair<uint64_t, uint64_t>> &v1, const std::vector<std::pair<uint64_t, uint64_t>> &v2,
                                     const mzd_t *H_T, const uint64_t start_offset1=0) {
	ASSERT( List1.size() > start_offset1 );
	typedef typename List::LabelType Label;

	Matrix_T<mzd_t *> H{(mzd_t *)H_T};

	const uint64_t nkl           = n-k-l;
	const uint32_t limbs         = H_T->width;
	//const uint64_t first_half    = l1/2;
	//const uint64_t second_half   = k/2;

	// prepare the first element
	List1[0].get_label().zero();
	for (int i = nkl; i < nkl+w1; ++i) {
		List1[0].get_label().data().write_bit(i, 1);
	}

	for (int i = 0; i < w2; ++i) {
		xor_helper<Label>(List1[0].get_label(), H_T->rows[l+i], limbs);
	}

	// make sure that out list is at least bigger than the difflists.
	ASSERT(List1.get_load() >= v1.size()*v2.size());

	uint64_t counter1 = 1;

	for(uint64_t i = 0; i < v1.size()+1; i++) {
		// because H_T is already transposed we can xor up the changes on the rows
		for(const auto &pos2 : v2) {
			List1[counter1].get_label() = List1[counter1-1].get_label();

			ASSERT(pos2.first < H.get_rows() && pos2.second < H.get_rows());
			// we do not need to add `l1` and `win`, because the pos counter already have them internally added
			xor_helper<Label>(List1[counter1].get_label(), H_T->rows[pos2.first], limbs);
			xor_helper<Label>(List1[counter1].get_label(), H_T->rows[pos2.second], limbs);
			ASSERT(counter1 < List1.size());

			counter1 += 1;
		}

		// are we in the last round? If so exit. Note that the outer loop is limited by `size1+1`. This is because we
		// need to tun the inner loop one last time.
		if (i ==  v1.size())
			break;

		// reset
		List1[counter1].get_label() = List1[counter1 - v2.size() - 1].get_label();

		auto pos = v1[i];
		ASSERT(pos.first < H.get_rows() && pos.second < H.get_rows());
		xor_helper<Label>(List1[counter1].get_label(), H_T->rows[pos.first ], limbs);
		xor_helper<Label>(List1[counter1].get_label(), H_T->rows[pos.second], limbs);

		counter1 += 1;
	}
}


struct ConfigFillDecodingLists {
public:
	const uint32_t n, k, l, l1, l2, w1, w2, epsilon;
	constexpr ConfigFillDecodingLists(uint32_t n, uint32_t k, uint32_t l,
	                                  uint32_t l1, uint32_t w1, uint32_t l2, uint32_t w2, uint32_t epsilon) :
			n(n), k(k), l(l), l1(l1), l2(l2), w1(w1), w2(w2), epsilon(epsilon)
	{}
};

/// TODO explain
/// \tparam n
/// \tparam k
/// \tparam l
/// \tparam l1
/// \tparam w1
/// \tparam l2
/// \tparam w2
/// \tparam epsilon
/// \tparam List
/// \param List1
/// \param List2
/// \param v11
/// \param v12
/// \param v21
/// \param v22
/// \param H_T
template<const uint64_t n, const uint64_t k, const uint64_t l, const uint64_t l1, const uint64_t w1, const uint64_t l2, const uint64_t w2, const uint64_t epsilon, class List>
void mceliece_d2_fill_decoding_lists(List &List1, List &List2,
                                     const std::vector<std::vector<std::pair<uint64_t, uint64_t>>> &v11, const std::vector<std::vector<std::pair<uint64_t, uint64_t>>> &v12,
                                     const std::vector<std::vector<std::pair<uint64_t, uint64_t>>> &v21, const std::vector<std::vector<std::pair<uint64_t, uint64_t>>> &v22,
                                     const mzd_t *H_T) {
	ASSERT(v11.size() == v21.size() && v12.size() == v22.size());
	assert(w2 < 4);

	// special case we have the zero element
	uint64_t start_offset1 = 1;
	uint64_t start_offset2 = 1;

	if (w2 > 0) {
		mceliece_d2_fill_decoding_lists<n, k, l, l1, 0, l2, 1, epsilon, List>(List1, List2, v11[0], v12[1], v21[0], v22[1],
										H_T, start_offset1, start_offset2);
		start_offset1 += (v11[0].size() + 1) * (v12[1].size() + 1);
		start_offset2 += (v21[0].size() + 1) * (v22[1].size() + 1);
	}

	if (w2 > 1) {
		mceliece_d2_fill_decoding_lists<n, k, l, l1, 0, l2, 2, epsilon, List>(List1, List2, v11[0], v12[2], v21[0], v22[2],
		                                                                      H_T, start_offset1, start_offset2);
		start_offset1 += (v11[0].size() + 1) * (v12[2].size() + 1);
		start_offset2 += (v21[0].size() + 1) * (v22[2].size() + 1);
	}

	if (w2 > 2) {
		mceliece_d2_fill_decoding_lists<n, k, l, l1, 0, l2, 3, epsilon, List>(List1, List2, v11[0], v12[3], v21[0], v22[3],
		                                                                      H_T, start_offset1, start_offset2);
		start_offset1 += (v11[0].size() + 1) * (v12[3].size() + 1);
		start_offset2 += (v21[0].size() + 1) * (v22[3].size() + 1);
	}

	// ADD HERE MORE CASES
	if (w1 > 0) {
		mceliece_d2_fill_decoding_lists<n, k, l, l1, 1, l2, 0, epsilon, List>(List1, List2, v11[1], v12[0], v21[1], v22[0],
		                                                                      H_T, start_offset1, start_offset2);
		start_offset1 += (v11[1].size() + 1) * (v12[0].size() + 1);
		start_offset2 += (v21[1].size() + 1) * (v22[0].size() + 1);

		if (w2 > 0) {
			mceliece_d2_fill_decoding_lists<n, k, l, l1, 1, l2, 1, epsilon, List>(List1, List2, v11[1], v12[1], v21[1], v22[1],
			                                                                      H_T, start_offset1, start_offset2);
			start_offset1 += (v11[1].size() + 1) * (v12[1].size() + 1);
			start_offset2 += (v21[1].size() + 1) * (v22[1].size() + 1);
		}

		if (w2 > 1) {
			mceliece_d2_fill_decoding_lists<n, k, l, l1, 1, l2, 2, epsilon, List>(List1, List2, v11[1], v12[2], v21[1], v22[2],
			                                                                      H_T, start_offset1, start_offset2);
			start_offset1 += (v11[1].size() + 1) * (v12[2].size() + 1);
			start_offset2 += (v21[1].size() + 1) * (v22[2].size() + 1);
		}

		if (w2 > 2) {
			mceliece_d2_fill_decoding_lists<n, k, l, l1, 1, l2, 3, epsilon, List>(List1, List2, v11[1], v12[3], v21[1], v22[3],
			                                                                      H_T, start_offset1, start_offset2);
			start_offset1 += (v11[1].size() + 1) * (v12[3].size() + 1);
			start_offset2 += (v21[1].size() + 1) * (v22[3].size() + 1);
		}
	}
}

#endif //SMALLSECRETLWE_LIST_FILL_H
