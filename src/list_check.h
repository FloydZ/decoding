#ifndef SMALLSECRETLWE_LIST_CHECK_H
#define SMALLSECRETLWE_LIST_CHECK_H


struct ConfigCheckResultList {
public:
	const uint32_t n, k, w;     // Instance parameters
	const uint32_t c, l, l1;    // optimize parameters
	constexpr ConfigCheckResultList(uint32_t n, uint32_t k, uint32_t w, uint32_t c, uint32_t l, uint32_t l1) :
			n(n), k(k), w(w), c(c), l(l), l1(l1)
	{}

};

/// \param L					input list to check. Labels will be partly recalculated. So this cannot be const.
/// \param weight_threshold 	weight treshhold to check on
/// \prama s					original syndrom
/// \param P					permutation applied to H
/// \param A					unpermutated, unchanged input decoding/mceliece challenge matrix
/// \return						a pointer to a M4RI matrix containing e. i.e. a 1 x n matrix.
/// special implementation of the `check_resultlist` function, with the possibility to redefine the l2 window to match on-
///	this is needed for our indyk motwani approach.
template<const ConfigCheckResultList &config, class List>
mzd_t *check_resultlist(List &L, const uint64_t weight_threshold, const mzd_t *s, const mzp_t *P, const mzd_t *A) {
	// extract parameters
	constexpr uint32_t n = config.n;
	constexpr uint32_t k = config.k;
	constexpr uint32_t w = config.w;
	constexpr uint32_t c = config.c;
	constexpr uint32_t l = config.l;
	constexpr uint32_t l1 = config.l1;
	ASSERT(n-k-l < n);

	typedef typename List::ValueType ValueType;
	typedef typename List::LabelType LabelType;
	typedef typename List::ElementType ElementType;
	typedef typename ElementType::LabelContainerType LabelContainerType;
	using T = LabelContainerType;

	uint64_t error_counter = 0;

	MADIVE((void *)L.data(), L.get_load() * ElementType::bytes(), POSIX_MADV_WILLNEED | POSIX_MADV_SEQUENTIAL);

	// precompute masks
	constexpr uint32_t lower1 = T::round_down_to_limb(0);
	constexpr uint32_t upper1 = T::round_down_to_limb(n-k-l-1);
	constexpr uint64_t l_mask1 = T::higher_mask(0);
	constexpr uint64_t u_mask1 = T::lower_mask2(n-k-l);

	constexpr uint32_t lower2 = T::round_down_to_limb(n - k - l + l1);
	constexpr uint32_t upper2 = T::round_down_to_limb(n-k-1);
	constexpr uint64_t l_mask2 = T::higher_mask(n-k-l+l1);
	constexpr uint64_t u_mask2 = T::lower_mask2(n-k);

	for (uint64_t i = 0; i < L.get_load(); ++i) {
		//auto weight_test = L[i].get_label().data().weight(0, n - k - l)+L[i].get_label().data().weight(n - k - l + l1,n-k);
		auto weight_test = L[i].get_label().data().template weight<lower1, upper1, l_mask1, u_mask1>()
		                   + L[i].get_label().data().template weight<lower2, upper2, l_mask2, u_mask2>();

		//if weight on first n-k-l coordinates of the label (e1) <= weight_threshold = (decoding case:) w-p
		if (weight_test < weight_threshold) {
			mzd_t *e = mzd_init(1, n-c);
			mzd_t *e_prime = mzd_init(1, n-c);
			mzd_t *e_T = mzd_init(n-c, 1);

			mzd_t *ss = mzd_init(1, n - k);
			mzd_t *ss_T = mzd_init(n - k, 1);

			for (int j =0;  j<n-k-l; ++j) {
				mzd_write_bit(e, 0, j, L[i].get_label()[j]);
			}
			for (int j = n-k-l; j < n-c/*+l1*/; ++j) {
				mzd_write_bit(e, 0, j, L[i].get_value()[j - (n - k - l)]);
			}
			for (int j = n-k-l+l1; j < n-c-k; ++j) {
				mzd_write_bit(e, 0, j, L[i].get_label()[j]);
			}
			for(rci_t j=0; j < n-c; ++j) {
				mzd_write_bit(e_prime, 0, P->values[j], mzd_read_bit(e, 0, j));
			}

			auto wt = hamming_weight(e_prime);
			if (wt > w) {
				mzd_free(e);
				mzd_free(e_prime);
				mzd_free(e_T);
				mzd_free(ss);
				mzd_free(ss_T);
				continue;
			}

			mzd_transpose(e_T, e_prime);
			mzd_mul_naive(ss_T, A, e_T);
			mzd_transpose(ss, ss_T);

			if (mzd_cmp(ss, s) == 0) {
				mzd_free(ss);
				mzd_free(ss_T);
				mzd_free(e_T);
				mzd_free(e);
				return e_prime;
			} else {
				assert(0);
			}

			// count the number of erroneously found elements, which have the correct weight distribution, but do not lead to the correct syndrom.
			error_counter += 1;

			mzd_free(e);
			mzd_free(e_prime);
			mzd_free(e_T);
			mzd_free(ss);
			mzd_free(ss_T);
		}
	}

	if (error_counter)
		std::cout << "weight error " << error_counter << "\n";
	return nullptr;
}

#endif //SMALLSECRETLWE_LIST_CHECK_H
