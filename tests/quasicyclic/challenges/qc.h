#ifndef DECODING_TESTS_QUASICYCLIC_QC_H
#define DECODING_TESTS_QUASICYCLIC_QC_H

#include <array>
#include <cstdint>

/// Expands a quasi-cyclic challenge of decodingchallenge.org into the format
/// of the other challenges: the string of H^T = [M ; I], where H = [I | M^T].
/// Row i of the circulant M is the vector h cyclically shifted to the right
/// by i, i.e. M[i][j] = h[(j - i) mod (n/2)] (see `Doc/format_of_challenges.pdf`).
/// \tparam n code length
/// \tparam k code dimension (= n/2)
/// \param h first column of M^T, string of length n/2
/// \return zero terminated string of length n*(n-k)
template<const uint64_t n, const uint64_t k>
constexpr std::array<char, n*(n-k) + 1> qc_to_string(const char *h) noexcept {
	constexpr uint64_t r = n - k;
	static_assert(k == r, "quasi-cyclic challenges have rate 1/2");

	std::array<char, n*r + 1> ret{};
	for (uint64_t i = 0; i < k; ++i) {
		for (uint64_t j = 0; j < r; ++j) {
			ret[i*r + j] = h[(j + r - i) % r];
		}
	}

	for (uint64_t i = 0; i < r; ++i) {
		for (uint64_t j = 0; j < r; ++j) {
			ret[(k + i)*r + j] = (i == j) ? '1' : '0';
		}
	}

	ret[n*r] = '\0';
	return ret;
}
#endif//DECODING_TESTS_QUASICYCLIC_QC_H
