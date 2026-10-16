#ifndef DECODING_TESTS_LOWWEIGHT_CODEWORD_H
#define DECODING_TESTS_LOWWEIGHT_CODEWORD_H

#include <array>
#include <cstdint>

/// Reduces a syndrome decoding challenge (H, s, w) to a low weight codeword
/// challenge: every solution e of H*e^T = s^T with wt(e) <= w gives the
/// non-zero codeword (e, 1) of weight <= w+1 of the code with the parity-check
/// matrix H' = [H | s^T].
/// \tparam n code length of the syndrome decoding challenge
/// \tparam k code dimension of the syndrome decoding challenge
/// \param h string of H^T (n rows of n-k symbols)
/// \param s syndrome (n-k symbols)
/// \return zero terminated string of H'^T: (n+1) rows of n-k symbols
template<const uint64_t n, const uint64_t k>
constexpr std::array<char, (n+1)*(n-k) + 1> append_syndrome(const char *h, const char *s) noexcept {
	constexpr uint64_t r = n - k;

	std::array<char, (n+1)*r + 1> ret{};
	for (uint64_t i = 0; i < n*r; ++i) {
		ret[i] = h[i];
	}

	for (uint64_t i = 0; i < r; ++i) {
		ret[n*r + i] = s[i];
	}

	ret[(n+1)*r] = '\0';
	return ret;
}

/// \tparam r length of the syndrome
/// \return zero terminated string of the zero syndrome
template<const uint64_t r>
constexpr std::array<char, r + 1> zero_syndrome() noexcept {
	std::array<char, r + 1> ret{};
	for (uint64_t i = 0; i < r; ++i) {
		ret[i] = '0';
	}

	ret[r] = '\0';
	return ret;
}
#endif//DECODING_TESTS_LOWWEIGHT_CODEWORD_H
