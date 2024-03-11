#ifndef SMALLSECRETLWE_DECODING_FILE_H
#define SMALLSECRETLWE_DECODING_FILE_H
#include <cstdint>
constexpr uint64_t n = 10;
constexpr uint64_t k = 3;
constexpr uint64_t seed = 0;
constexpr uint64_t w = 9;
constexpr const char *s = "1020101";
constexpr const char *h = "1000000010000000100000001000000010000000100000001110121111120201002122";
// load with `mzd_t *A = mzd_from_str(k, n, h);`
#endif //SMALLSECRETLWE_DECODING_FILE_H