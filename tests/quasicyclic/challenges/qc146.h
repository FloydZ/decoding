#ifndef DECODING_TESTS_QUASICYCLIC_QC146_H
#define DECODING_TESTS_QUASICYCLIC_QC146_H
#include <cstdint>

#include "qc.h"

// decodingchallenge.org: Challenges/QC/Provider0/QC_12
constexpr uint64_t n = 146;
constexpr uint64_t k = 73;
constexpr uint64_t w = 12;
constexpr const char *h_vector = "1100110000000010010101100001110001001001101001000111110010100000001110110";
constexpr const char *s = "0011000011001100001100110010001001101111011000011111000111000010101101100";

constexpr auto h_matrix = qc_to_string<n, k>(h_vector);
constexpr const char *h = h_matrix.data();
#endif//DECODING_TESTS_QUASICYCLIC_QC146_H
