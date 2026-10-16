#ifndef DECODING_TESTS_QUASICYCLIC_QC102_H
#define DECODING_TESTS_QUASICYCLIC_QC102_H
#include <cstdint>

#include "qc.h"

// decodingchallenge.org: Challenges/QC/Provider0/QC_10
constexpr uint64_t n = 102;
constexpr uint64_t k = 51;
constexpr uint64_t w = 10;
constexpr const char *h_vector = "001000110010011100101010101000001010111101110010101";
constexpr const char *s = "110101101011010001000101001011100010101010111000010";

constexpr auto h_matrix = qc_to_string<n, k>(h_vector);
constexpr const char *h = h_matrix.data();
#endif//DECODING_TESTS_QUASICYCLIC_QC102_H
