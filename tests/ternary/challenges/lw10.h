#ifndef DECODING_TESTS_TERNARY_LW10_H
#define DECODING_TESTS_TERNARY_LW10_H
#include <cstdint>

#include "large_weight.h"

// decodingchallenge.org: Challenges/LargeWeight/LargeWeight_10_0
constexpr uint64_t n = 10;
constexpr uint64_t k = 3;
constexpr uint64_t q = 3;
constexpr uint64_t w = 9;
constexpr const char *m = "1101211"
                          "1112020"
                          "1002122";
constexpr const char *s = "2010202";

constexpr auto h_matrix = append_identity<n, k>(m);
constexpr const char *h = h_matrix.data();
#endif//DECODING_TESTS_TERNARY_LW10_H
