#ifndef DECODING_TESTS_TERNARY_LARGE_WEIGHT_H
#define DECODING_TESTS_TERNARY_LARGE_WEIGHT_H

#include <array>
#include <cstdint>

/// Expands a challenge of decodingchallenge.org, which only stores the rows of
/// M (where H = [I | M^T]), into the format of the other challenges: the
/// string of H^T = [M ; I].
/// \tparam n code length
/// \tparam k code dimension
/// \param m the k rows of M, each n-k symbols, as one string
/// \return zero terminated string of length n*(n-k)
template<const uint64_t n, const uint64_t k>
constexpr std::array<char, n*(n-k) + 1> append_identity(const char *m) noexcept {
	constexpr uint64_t r = n - k;

	std::array<char, n*r + 1> ret{};
	for (uint64_t i = 0; i < k*r; ++i) {
		ret[i] = m[i];
	}

	for (uint64_t i = 0; i < r; ++i) {
		for (uint64_t j = 0; j < r; ++j) {
			ret[(k + i)*r + j] = (i == j) ? '1' : '0';
		}
	}

	ret[n*r] = '\0';
	return ret;
}
#endif//DECODING_TESTS_TERNARY_LARGE_WEIGHT_H
