#ifndef SMALLSECRETLWE_DECODING_FILE_H
#define SMALLSECRETLWE_DECODING_FILE_H
#include <cstdint>
constexpr uint64_t n = 10;
constexpr uint64_t k = 5;
constexpr uint64_t seed = 0;
constexpr uint64_t w = 4;
constexpr const char *s = "01110";
constexpr const char *h = "11011111100100101001101111000001000001000001000001";
// load with `mzd_t *A = mzd_from_str(k, n, h);`
#endif //SMALLSECRETLWE_DECODING_FILE_H