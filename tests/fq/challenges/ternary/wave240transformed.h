#ifndef SMALLSECRETLWE_DECODING_FILE_H
#define SMALLSECRETLWE_DECODING_FILE_H
#include <cstdint>
constexpr uint64_t n = 240;
constexpr uint64_t k = 88;
constexpr uint64_t seed = 0;
constexpr uint64_t w = 227;
constexpr const char *s = "21102220202112101220020010122122002100100102202010220102110010100010022122101102101000220210000020102101220112102222002000100112200200120210200111112000";
constexpr const char *h = "100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111012111112020100212220102021120111220211210200212220211021200200021001210121202120222110211201000022110020000222122120022121112221012012210002102010110002002222200020220101002000201020202120100021110021020210112120220020012102120012122121121022012012021001121212220211212002021000211222101221220010120121220011110102202001211000202220002201120011200110102222100100002121120222212212112001210001111201200100211120101200102201111100211110102110112200221200002001001221112220101012010100202120011102002000111200210212212121112202200122110201201212210020000010110021121220202220210122111220020201221101111110020012100110122022011021110010002202102221011002212212022210112202001110112000110222200022221210012100100102011210112101001201100010200002220201120200010112100011111221020022101000120001112002010200112120211210012122011012110111000102220211011001120112201010212100102122012102102112001200112221102111202000112102000111202220201221211112200020112102211212101122112011021020011102020222101002012212002100112110112210120210001111112001022010020010222002120221111022222221211121011000010120001202120101202011211102001002100122211211112001200112101111000000101201210001001111021220211221221100022002100122022210012102210222221020021210121120002211220010020001220020011220210202211110022021212120102120111000122011222201120101220220101200000021001210011011021221101122222102011002022222001000002220201202212022011221020101120102102111110121210222120112011100120122221202211100210222221001202020111211112100020101021221220021102002202022221111101211222110210020222111021220012020212210220201010210221220001012111112101010101112221202112022110112000211000202210101102112020112000012222022022222101200201011221121012002202201002120120212012020210001212111212001002001200120211022010101122121211001101010001001222011222110021102200110100221001102200002202101221000211001001112012002021200001200002200210112212001202100122101101101021120112120221010211202002101211021210011102120110211220010221112012102111201000210012110222201222000000020112212120110122122012221011112111020002222202220222201222010000121102112212100220200202122012211020101201022110210120121221110021012010011212011001010020202200012221220121102000111001121001210211001021211120011210221211111210110101020110021112122021211012200111010201010210002100001200022002100221020001110101021021211212022010201020220102210122020210211012101110221010111012021101011100111212022001222000101201120110010122021022222102221220210012122000000122022021221211200121211121100000112201200100122120202022111220022100121102002022101102110112210210001222102111120012200002100211101112202220122001021101022101202020102010220022010222110121200110122002112100201002102211012210021101021221000210211221020220120111201121120110200211000100010121020111111200221000010021200021200202202021220210010121012210002000202112111211200202000202101220222102012111222100120211111011211221121221220010120202222211111122020112000201000202121022002010112110210200022112010100120222122000120222020111222020101221112112010021221211202102202002112202111021220002000111012002122200010021022211120212020202002111101202020021002001021012001011121122001111212120002121000001220202121122112122212102021112002010112020220021010111110202212212200212201212210021222021121122021101110010221102122010012022220110202120202101112011120000202101212111222210001012221121110202020202221220200002102120222011002000011102210002100201221202102222122000022201001202102011210222122001111221012100102202021011220020111120112212022220201212210101002102212200000021102020010002020222221212121222011022001022000011021211122221202011222000100110221220021122010012212221100101220202110202112000012100112021200102002220112212210020222201220000222020220001012121210200101112110120002200201101122202122221010122102110101012102122112222000101210111210211020011220201112201102001211021111120021000221210122010212011202221000011100122112011000012012021002212000201001120122021000102101210122221012021201121122122111000112011221200012200122120011111002001020222221210010212212002111222100011211222011120212121012112222112111220101111020101021001000200200222121011112001002012121110211100221012120101000101100020020120200220002022000110012122020012120022022002112122221102200211211210000122202000121201100021210202102110202220110112102101012101021201210210220112200102220112201002010102200220122122111001101022222221202022111002111222201020201200010112211221110121112112201222011221200120120012121212122221202021011000021201112011112220012112220102000100011022002120210110110100220121210021112001111211010120100111111001021010022212100210112200111110022120020100221200120111202000020120102200211021211202110020022122120121110010100001101001000221211221221102220102220110001222222221112101102220210002000011200200201122221101202202012010002201111121111211211211120021111222001002001220202101010002022022020020211001101220202212100102020021121012120010001101120101211012011002110220022212212011110101001112011021010102001212201102212000000122001111211220111022102221210211002210012101122010200110022020012122200022201102010020200120212220012021212021021101110120102022101012110022120202000112020220001110102121202120201122001120200000111210022110102101011022122102220021102202000101002222021101002021221100210120211110120110100002102120110220212012021200212210001221011022200101111001212001111211010120120001012012212020121210002200220111201022010201002120022202212221112000211221021122201002012111101212012112100122211202010100002210210200211222210011210021000022001020011121222122122000200000022022211211022222122002121212201220112212012212211102112021120021202022200012102201201211021211110222101200111112220202111100202011110121011022111221010120121211010010020111220102011020102212010000112221011211100220122021102022020022102021211011001212121010020021200020200002011011120020111022211010121002020201222220201111101110101011011121010002121222001001120011220102201211211101001022102102010111120120120101012120002101222200122211212200221211101111220102011022102121011000100102200001211001210111222221022001212112200112021212000011011222122002200122100002101222010100201022001100021120202121120120202201120001102100120011010000022011220100201021011021110120202111110022210021221011010121010120222020012111202101120100122202220222022002211000002001211002210001221000001211120202221021220011212122112200122012102111112100110001201011012110211200202110110020011112101122200002200211021102010200002011122011220002021101021220110111002022020222211102201210222001011110022001222201222000022111221121100212020001102121211220202120221102021201201201111002212120222200122221022020120102122110002101100112202020110112120201020111102202220100100100001112002121202011102221021102200121011200022101012221210002101221200221120212100112121202101121221120210111120211012202022120020222222121100211110200012001120010012000101000202200111020022002011011020012122202222022120001001112120011000120212222100000010222201110010112222200201111001111001100011220002202012210210122012010121122220122011110011122122200010000112002220100111111112012021012211002021120021120021222200110000022011012120120101022122021122210100112002020101012212102020020101002021111211222220022222011111000221110220220210201222012021100211021210111111121001100101102102222221100021001000022121100221001220011210220221100112111110210112110111201112210022121211221012121201212010212210220122022020210201110122022022221111222201000120011022100100220101212020210001011111102212202111020212212112020000102202212121202222221002101212220001121201102222212211020111021001012122002020222010021111001212222022221101210222210210201201101122201121011222111100100010000202202010200100112021000020022020001122012122201102210120002211102020100100010200211121012202111120200221102120100012210112000102200001220211200111020220201200212110002020110002112000022000101210112001222022021122202010212012210210100110221122002010220112110221120020100001120000220211221022221211200111010100122010202110002021200221121220210110121210211211110201020220112212122102110202112000022202121200112110122211210220201200202010111102110100011210211202210021100102121012000012011010220211111020020020220221200210212120222100000121111010011020111012112021111202012220212012200021000110211011210200211112202021120012022121210110000211000112200202202110111012121002211221100020120010221000002011111211012102122121102020022100020211111100121110000120201212202212000211200122211211211202021010002011201212022102201212120001212000221220020200200200200000112101201210210001120001110121022000101221222002001001200221011102120212211211210012002001002202021111212221002020120100211210010001002020020001210100222220000211122022110002010020100111101222111102210110021222202202200211020110021111121120200000121211010122122122121220201022222211022011112010210022210121211222101000222221011201121101101122121012102001120101000112202120101001001111121220000110222120211011110100111112020200102110002111221221020210011012221001221010211021212111112100000211021202002122102100110010102010201110220112201212011221110120110000011010001112011111122121102202110020011101000022120122110012200210000002201010212210022222011002221221012222002012021122121101121220210022212011112221220002101202002001110220020200102222100012020210121000112220101120102020221000200220110010202012111022112110020122011000121222010102020201110200212202020100022101012201101101020221112000222200110010112120010110100001220012110000012000111022222122221202110102011012100202220121100102020001111200012000102011120221111111102122120001102221110002101122102102121221220222010201222021101121112111022201011021201000112012001102012010021222110220210122110100012002101201000000021020221202010022210021021200022212222212012021220100111221102021100222012112210020111122101002101110120121110200010101101202221012011221222220000211112210210012202010221220222110222120111112221110011100202022210220222100211200221120011102110211221221121200000012212202100200100111002212010122121112200112021211122101022110022101020120020211102210000210221012220201211212112020112010112110220012110022010112122112010102111100102200110012121021102211221122222120100201221202201221000021100222221011011121121201121001121221121212011210201010100122120220221201022102211020021012210020120202221201201100111101020020000221121002010200202122201220221121102021011220010001201202010200011122111120010210110112011101220220200020110212220012000222021001020002200002001222222212202121101220100010110210122221101212201220120211212121112200100111111000021022211022210022202020200122221022012211102102210020201210102022201120011112112211212121020120112112120001200001112210102002112201021101101010102210012212120000100020021201001202021100120012002112100200200122220001000010102221221221200012111201212100001001101021121012210022112001120122111111121100222221201012222102122000010000022201221102120120102110010222021212100112121211111202200112221022012220221101212220200111211200100022210011010000011101102201122200122110122102110221210021220002120211010222102100102120202010022120202211100201020122101010210000122201001202010000201102022210221121212202001102020012211102011101000201001222010020100000022020112121222201112111000011021122220110011002021121100201012222021002111212121010012100122201122001011102020200202002120002010121221022010101111020101100102001120022100100011211102222220211112101210101110120210002000111112201111221201100220012220010220012021221010010110011000210100112001202122012012202101122021100000101210201120201120200110011021202200022100111211222221101012022001012121011012011222000021112112001121001010222212011112100010022220010012220000220111210000102220212202102201112222212210202202200121100101220120201210210202201202120201110110100122010201110001201220220121120020102101120100100120011121200211221001001220100202101002120122202200002122222011221201211120101100210202020110020000122201100010201112212221121211222022010011012121012120212010122022012102221202210100011100022121101202212210102222221021112122221011011021100002210221212112100012112201022020212011211011222022111102010001200220120210002211202210010011121122200210222202221202000021112010222101100101120012002021022000111011120000010112022020222100220202002002210101010202221020222110201210020211101211212022110210011210012022201021001010202211110202011111111110001120111102200220100101021110200120022210000100010010111211022200102012021121201001001112002002222010001221212012021212200012110211000120120201002010001122002101111222200122010022210220202101022110102100002021210011011221022221120002101220010200111220121220221002102110100120022220210201022222212212100100021012022222222021010220211212221001120002122021111112120122120000212121111011120102000110200121121012202220001002021211121120011200011210002222220011101221100021122022112121112201012121200020101222120110021200221121212111002010200121210210210121011110110112120202001100111122010022221202111012022101000000110201122100120202021200111211100100112112212010101022002011222112222221121020221001102211220201001110212102122221222202012021112110100000010012002012222022120202021201200122112100020200110020100010002201211211010220022002210012200020122112000002102101022201100021002112122021121011211012121100100010012102102120000212200001120101002221222010021100022102111111102010211212122012202021110012022200";
// load with `mzd_t *A = mzd_from_str(k, n, h)`
#endif //SMALLSECRETLWE_DECODING_FILE_H