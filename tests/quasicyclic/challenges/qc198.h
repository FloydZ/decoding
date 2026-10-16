#ifndef DECODING_TESTS_QUASICYCLIC_QC198_H
#define DECODING_TESTS_QUASICYCLIC_QC198_H
#include <cstdint>

#include "qc.h"

// decodingchallenge.org: Challenges/QC/Provider0/QC_14
constexpr uint64_t n = 198;
constexpr uint64_t k = 99;
constexpr uint64_t w = 14;
constexpr const char *h_vector = "001011101011111111110010111000110011010000001010000011010011101010101001110101111110001000000110010";
constexpr const char *s = "001110000010110101001000110111101101001111010110101011111110010000001110001111001101011011101000101";

constexpr auto h_matrix = qc_to_string<n, k>(h_vector);
constexpr const char *h = h_matrix.data();
#endif//DECODING_TESTS_QUASICYCLIC_QC198_H
