#ifndef DECODING_TESTS_TERNARY_LW20_H
#define DECODING_TESTS_TERNARY_LW20_H
#include <cstdint>

#include "large_weight.h"

// decodingchallenge.org: Challenges/LargeWeight/LargeWeight_20_0
constexpr uint64_t n = 20;
constexpr uint64_t k = 7;
constexpr uint64_t q = 3;
constexpr uint64_t w = 19;
constexpr const char *m = "1101211111202"
                          "0100212220102"
                          "0211201112202"
                          "1121020021222"
                          "0211021200200"
                          "0210012101212"
                          "0212022211021";
constexpr const char *s = "1201000022110";

constexpr auto h_matrix = append_identity<n, k>(m);
constexpr const char *h = h_matrix.data();
#endif//DECODING_TESTS_TERNARY_LW20_H
