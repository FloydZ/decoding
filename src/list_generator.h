#ifndef SMALLSECRETLWE_LIST_GENERATOR_H
#define SMALLSECRETLWE_LIST_GENERATOR_H

#include "combinations.h"

///
/// \param v
void print_diff_list(std::vector<std::pair<uint64_t, uint64_t>> const &v){
	for (int i = 0; i < v.size(); ++i) {
		std::cout << v[i]. first << " " << v[i].second << "\n";
	}
	std::cout << "\n" << std::flush;
}

/// simple helper function which applies the bitflips. The parameter offset can be used to choose between `left`, `right`
/// forms of the baslists.
template<class Element>
void apply_diff(Element &e, const std::pair<uint64_t, uint64_t> &pos, const uint64_t offset=0){
	if (pos.first == pos.second)
		return;

	e.get_value().data().flip_bit(pos.first + offset);
	e.get_value().data().flip_bit(pos.second + offset);
}


///
/// \tparam List
/// \tparam MatrixType
/// \param L
/// \param H
/// \return
template<class List, class MatrixType=mzd_t *>
bool check_correctness(std::vector<List> &L, const Matrix_T<MatrixType> &H) {
	typedef typename List::LabelType TType;

	for (auto &L1: L) {
		ASSERT(L1.get_load() > 0);
		for (int i = 0; i < L1.get_load(); ++i) {
			TType tmp = L1[i].get_label();
			L1[i].recalculate_label(H);

			if (tmp.is_equal(L1[i].get_label(), 0, TType::size()) == false){
				std::cout << tmp;
				std::cout << L1[i].get_label();
				std::cout << L1[i].get_value() << "\n" << std::flush;
				return false;
			}

			// allow the first element to be zero
			if((tmp.data().is_zero() == true) && (i > 0)) {
				std::cout << tmp;
				std::cout << L1[i];
				return false;
			}
		}
	}

	return true;
}

/// IMPORTANT L ist not const. It will be changed.
template<class List, class MatrixType=mzd_t *>
bool check_correctness(List &L1, const Matrix_T<MatrixType> &H) {
	std::vector<List> L {L1};
	return check_correctness<List, MatrixType>(L, H);
}


// generate a list of the chase sequence
// with the following properties
//     <-                      n                     ->
//      ___________p/2________n/2______________________
// L1: |___________e_1_________|___________0___________|
//
//      _____________________start_________p/2_________
// L2: |__________0____________|___________e_2_________|   Iterate over all
//
// the indices are working as follows:
//            v[i]                  // which means you get the i+1 element of both lists by applying the i diff vector.
//  l[i]    ----->      l[i+1]
// The function only works if n and p are even.
/// \tparam DecodingList
/// \tparam n_
/// \param l1
/// \param l2
/// \param v
/// \param p
template<class DecodingList, const uint64_t n_>
void generate_decoding_lists(DecodingList &l1, DecodingList &l2, std::vector<std::pair<uint64_t, uint64_t>> &v,
                             const uint32_t p) {
	//constexpr uint64_t n_ = DecodingList::ElementType::get_value_size();
	ASSERT(n_/2 > 0 && p > 0);

	using ValueContainerLimbType = typename DecodingList::ValueContainerType::ContainerLimbType;
	using DecodingElement = typename DecodingList::Element;
	using VCLT = ValueContainerLimbType;

	DecodingElement e;  e.zero();
	DecodingElement e_old; e_old.zero();
	const uint64_t limbs = e.get_value().data().limbs();
	uint64_t rt, counter = 0;
	uint16_t pos1 = 0, pos2 = 0;
	uint64_t half_bit_pos    = n_/2;

	// first resize every list for faster saving
	const uint64_t size = bc(n_/2, p);
	l1.resize(size);
	l2.resize(size);
	v.resize(size);

	l1.set_load(size);
	l2.set_load(size);

	// init the combinations generator
	Combinations_Chase_Binary<VCLT> c{n_/2, p};
	c.left_init(e.get_value().data().data().data());
	c.left_step(e.get_value().data().data().data(), true);
	e_old = e;

	// do the first step
	rt = c.left_step(e.get_value().data().data().data());

	// now loop
	while(rt != 0) {
		// calc the diff bit positions
		Combinations_Chase_Binary<VCLT>::diff(e.get_value().data().data().data(),
		                                e_old.get_value().data().data().data(),
		                                limbs, &pos1, &pos2);
		auto b = std::pair<uint64_t, uint64_t> (pos1, pos2);
		v[counter] = b;

		// first copy the easy part
		l1[counter] = e_old;
		BinaryContainer<n_>::shift_right(l2[counter].get_value().data(), e_old.get_value().data(), half_bit_pos);

		// next step
		e_old = e;
		rt = c.left_step(e.get_value().data().data().data());
		counter += 1;
	}
	// dont forget the last one.
	l1[counter] = e_old;
	BinaryContainer<n_>::shift_right(l2[counter].get_value().data(), e_old.get_value().data(), half_bit_pos);
}


// generates all elements with weight distribution
// [    w1   |   w2   |   0   ]
// <   l1    ><   l2  ><n-l1-l2>
// the last part can be empty.
// the binary flag indicates if the succumb data container are binary or k_ary.
// IMPORTANT: this function only prepares the values of each element. It does not calculate the label.
void prepare_generate_base_mitm(DecodingList &L,
								std::vector<std::pair<uint64_t, uint64_t>> &diff_list,
                                const uint64_t l1, const uint64_t l2,
                                const uint64_t w1, const uint64_t w2,
                                const bool binary=true) {
	ASSERT(w1 < l1 && w2 < l2 && 0 < w1 && 0 < w2);
	using Element               = DecodingList::ElementType;
	using ValueContainerLimbType= DecodingList::ValueContainerType::ContainerLimbType;
	using VCLT = ValueContainerLimbType;

	uint16_t pos1, pos2;
	Element e{}, e2{}; e.zero(); e2.zero();
	const unsigned int value_length = e.value_size();
	ASSERT(l1+l2 <= value_length);

	// resize the data.
	const uint64_t lsize = bc(l1, w1)*bc(l2, w2);
	diff_list.resize(lsize);

	if(binary) {
		const uint32_t limbs = 1+ (value_length/64);

		Combinations_Chase_Binary<VCLT> ccb_l1{l1, w1, 0};
		ccb_l1.left_init(e.get_value().data().data().data());
		ccb_l1.left_step(e.get_value().data().data().data(), true);

		uint64_t rt_l1 = 1, counter = 0;
		while (rt_l1 != 0) {
			e2 = e;

			Combinations_Chase_Binary<VCLT> ccb_l2{l1+l2, w2, l1};
			ccb_l2.left_init(e.get_value().data().data().data());
			ccb_l2.left_step(e.get_value().data().data().data(), true);

			do {
				Combinations_Chase_Binary<VCLT>::diff(e.get_value().data().data().data(),
				                                e2.get_value().data().data().data(),
				                                limbs, &pos1, &pos2);
				diff_list[counter++] = std::pair<uint64_t, uint64_t> (pos1, pos2);

				L.append(e);
				e2 = e;
			}while (ccb_l2.left_step(e.get_value().data().data().data()) != 0);
			rt_l1 = ccb_l1.left_step(e.get_value().data().data().data());
		}
	}else {
		std::cout << "not implemented\n";
		exit(1);
		return;
	}
}

///	generates the mitm lists of the form
///	if epsilon = 0
/// 	if side = 0
/// 		[	w1   |	0	|	0		|	w2	 |	0   ]
///			<l1/2   ><	-   >< window   >< l2/2+0><  -  >
///		side = 1
/// 		[	0   |	w1	|	0		|	0	 |	w2  ]
///			<l1/2   ><	-   >< window   >< l2/2-0><  -   >
///	if epsilon != 0
///		if side = 0
/// 		[	w1   |	0	|	0		|	w2	 |	0   ]
///			<l1/2   ><	-   >< window  >< l2/2+e ><  -  >
///		if side = 1
/// 		[	0   |	w1	|	0		|	0	 |	w2  ]
///			<l1/2   ><	-   >< window  >< l2/2-e ><  -  >
///
///	This means that the parameter `epsilon` can be understand as a overlapping factor on the l2 part of the lists.
///	can be called like this:
///			#define G_l                     16
///			#define G_l1                    12
///			#define G_l2                    (G_l-G_l1)
///			#define G_w1                    1u
///			#define G_w2                    1u
///			#define win                    	10u
///			DecodingList List1{0}, List2{0};
///			std::vector<std::pair<uint64_t, uint64_t>> diff_list1, diff_list2;
///			prepare_generate_base_mitm2(List1, diff_list1, diff_list2, G_l1, G_k, win, G_w1, G_w2, false, 0, true);
/// \tparam T			List type
/// \param L			output: List
/// \param diff_list1	output: difflist. The Difference list works as follows
///								 dl[0]		    dl[1]	   		    d[i-1] 		  dl[i]	               dl[last-1]
///							L[0] -------> L[1] -------> ... L[i-1] -------> L[i] -------> ... L[last-1] -------> L[last]
/// \param diff_list2	output: difflist.
/// \param l1			input: length of the e_d part == the aditional identiy amtrix size
/// \param l2			input: length of the e_1 part == the actual ISD part to solve.
/// \param win			input: length of the possible/hopefully zero part
/// \param w1			input: weight on the l1 part
/// \param w2			input: weight on the l2 part
/// \param side			input: if set to true the right part baselists is generated. IMPORTANT: Due to the fact, i do not want to return seperat diff lists for each side of the baselist, it can happen that the difflist is not correct for the right side of the baselist, because l2 can be odd. This leeds to different sized left and right parts. This elements are ignored.
///						input: if set to false, the left part is generated
/// \param epsilon		input:
/// \param binary		input: if set to true : only binary baselists will be generated.
///						input: if set to false: kAry baselists will be generated.
/// \param resize 		input: if set to true : every listinput will first be reset.
///						input: if set to false: No input/Output list will be resetted
/// \param set_w2 		input: if set to true : the output list `diff_list2` will be written to
///						input: if set to false: no data will be written to `diff_list2`
/// \param set_w1 		input: if set to true : the output list `diff_list1` will be written to
///						input: if set to false: no data will be written to `diff_list1`
template<class T>
void prepare_generate_base_mitm2(T &L,
								 std::vector<std::pair<uint64_t, uint64_t>> &diff_list1,
                                 std::vector<std::pair<uint64_t, uint64_t>> &diff_list2,
                                 const uint64_t l1, const uint64_t l2, const uint64_t win,
                                 const uint64_t w1, const uint64_t w2,
                                 const bool side,
                                 const uint64_t epsilon=0,
                                 const bool binary=true, const bool resize=true,
                                 const bool set_w2=true, const bool set_w1=true) {
	ASSERT(l1+l2+win <= T::ElementType::value_size());
	if (l1 != 0) ASSERT(w1 < l1/2);
	if (l2 != 0) ASSERT( w2 < (l2/2 - epsilon) && epsilon < l2/2 );
	typedef typename T::ElementType Element;
	using ValueContainerLimbType = typename T::ValueContainerType::ContainerLimbType;
	using VCLT = ValueContainerLimbType;

	uint16_t pos1, pos2;
	Element e{}, e2{}; e.zero(); e2.zero();
	const unsigned int value_length = e.value_size();

	uint64_t size1 = !side ? bc(l1/2, w1) : bc(l1 - (l1/2), w1);  // allow 1 additional element, if l1 or l2 an uneven
	uint64_t size2 = !side ? bc(l2/2+epsilon, w2) : bc(l2 - (l2/2) + epsilon, w2);
	size1 -= 1; size2 -= 1;
	const uint64_t lsize = (size1+1) * (size2+1);
	if (resize) {
		L.set_load(0);
		L.resize(lsize);
	}

	if (set_w1)
		diff_list1.resize(size1);
	if (set_w2)
		diff_list2.resize(size2);

	bool already_set = false;

	if(binary) {
		// puh complicated. This if/else sequence determine the binary borders of the values as described in the schematic
		// description in the function header.
		const int64_t e_start2 =  side ? -epsilon : 0;
		const int64_t e_len2   = !side ?  epsilon : 0;

		const uint64_t start1 = !side ? 0 : (l1 / 2) ;
		const uint64_t start2 = !side ? win+l1 : (win + l1 + e_start2 + uint64_t(floor(l2 / 2)));
		const uint64_t len1   = !side ? l1/2 : l1;
		const uint64_t len2   = !side ? l1+win+l2/2 + e_len2 : l1+l2+win;

		const uint32_t limbs = 1+ (value_length/64);

		Combinations_Chase_Binary<VCLT> ccb_l1{len1, w1, start1};
		if (w1 != 0) {  // Allow the case w1 = 0.
			ccb_l1.left_init(e.get_value().data().data().data());
			ccb_l1.left_step(e.get_value().data().data().data(), true);
		}

		uint64_t rt_l1 = 1, counter1 = 0, counter2 = 0;
		while (rt_l1 != 0) {
			e2 = e;

			if (w2 != 0) {
				Combinations_Chase_Binary<VCLT> ccb_l2{len2, w2, start2};
				ccb_l2.left_init(e.get_value().data().data().data());
				ccb_l2.left_step(e.get_value().data().data().data(), true);

				// we have to do the first step outside of the loop. Because otherwise the first diff list element
				// is not correctly initialised. This is due to the fact, that only one bit difference exists between e and e2.
				// Also we need to add this element to the output list.
				e2 = e;
				ccb_l2.left_step(e.get_value().data().data().data());
				L.append(e2);

				do {
					if ((!already_set) && set_w2) {
						ASSERT(counter2 < size2);
						Combinations_Chase_Binary<VCLT>::diff(e.get_value().data().data().data(),
						                                e2.get_value().data().data().data(),
						                                limbs, &pos1, &pos2);
						diff_list2[counter2++] = std::pair<uint64_t, uint64_t>(pos1, pos2);
					}

					L.append(e);
					e2 = e;
				} while (ccb_l2.left_step(e.get_value().data().data().data()) != 0);
			}
			// this is needed to get the correct dist index
			if (w1 != 0) {
				ASSERT(counter1 < size1+1);
				if (w2 == 0){
					L.append(e);
				}

				// We have to do a step. No matter what.
				rt_l1 = ccb_l1.left_step(e.get_value().data().data().data());
				// But we only save the diff list if we have to.
				if (set_w1) {
					if (counter1 >= size1)
						continue;

					Combinations_Chase_Binary<VCLT>::diff(e.get_value().data().data().data(),
					                                e2.get_value().data().data().data(),
					                                limbs, &pos1, &pos2);
					diff_list1[counter1++] = std::pair<uint64_t, uint64_t>(pos1, pos2);
				}
			} else {
				// we
				rt_l1 = 0;
			}

			// This flag indicates that we have already filled the diff list in the inner loop and we do not have to recalculate them.
			already_set = true;
		}

	} else {
		std::cout << "not implemented\n";
		return;
	}

	//std::cout << L.get_load() << " " << L.get_size() << " \n";
	if(resize){
		ASSERT(L.get_load() == L.size());
	}
}

/// extends the functions `prepare_generate_base_mitm2` by allowing every weight between 0<=x<=w1 and 0<=y<=w2.
/// This obviously leeds to the fact that the diff lists are now lists of diff lists.
/// Call it with:
///			DecodingList List1{0}, List2{0};
///			std::vector<std::vector<std::pair<uint64_t, uint64_t>>> diff_list1, diff_list2;
///			prepare_generate_base_mitm2_extended(List1, diff_list1, diff_list2, G_l1, G_k, G_l2, 1, 2, false, 0, true);
///			EXPECT_EQ(true, check_correctness(List1, diff_list1, diff_list2));
/// \tparam T
/// \param L
/// \param diff_list1
/// \param diff_list2
/// \param l1
/// \param l2
/// \param win
/// \param w1
/// \param w2
/// \param side
/// \param epsilon
/// \param binary
template<class T>
void prepare_generate_base_mitm2_extended(T &L,
                                 std::vector<std::vector<std::pair<uint64_t, uint64_t>>> &diff_list1,
                                 std::vector<std::vector<std::pair<uint64_t, uint64_t>>> &diff_list2,
                                 const uint64_t l1, const uint64_t l2, const uint64_t win,
                                 const uint64_t w1, const uint64_t w2,
                                 const bool side,
                                 const uint64_t epsilon=0,
                                 const bool binary=true) {
	ASSERT(w1 < l1/2 && w2 < (l2/2 - epsilon) && epsilon < l2/2 && l1+l2+win <= T::ElementType::value_size());
	typedef typename T::ElementType Element;

	// resize the output list.
	L.set_load(0);
	L.resize(0);

	diff_list1.resize(w1+1);
	diff_list2.resize(w2+1);

	bool w2_not_set = true;
	for (uint64_t w1_i = 0; w1_i <= w1; ++w1_i) {
		for (int w2_i = 0; w2_i <= w2; ++w2_i) {

			// catch the case where everything is empty.
			if ((w1_i == 0) && (w2_i == 0)){
				Element e;
				e.zero();

				L.append(e);
			}

			prepare_generate_base_mitm2(L, diff_list1[w1_i], diff_list2[w2_i], l1, l2, win, w1_i, w2_i, side, epsilon, binary, false, w2_not_set);
		}

		// after one inner loop the output list `diff_list2` is completed.
		w2_not_set = false;
	}
}

#endif //SMALLSECRETLWE_LIST_GENERATOR_H
