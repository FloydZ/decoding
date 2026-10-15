#ifndef DECODING_TESTS_TERNARY_LW30_H
#define DECODING_TESTS_TERNARY_LW30_H
#include <cstdint>

#include "large_weight.h"

// decodingchallenge.org: Challenges/LargeWeight/LargeWeight_30_0
constexpr uint64_t n = 30;
constexpr uint64_t k = 11;
constexpr uint64_t q = 3;
constexpr uint64_t w = 29;
constexpr const char *m = "1101211111202010021"
                          "2220102021120111220"
                          "2112102002122202110"
                          "2120020002100121012"
                          "1202120222110211201"
                          "0000221100200002221"
                          "2212002212111222101"
                          "2012210002102010110"
                          "0020022222000202201"
                          "0100200020102020212"
                          "0100021110021020210";
constexpr const char *s = "1121202200200121021";

constexpr auto h_matrix = append_identity<n, k>(m);
constexpr const char *h = h_matrix.data();
#endif//DECODING_TESTS_TERNARY_LW30_H
