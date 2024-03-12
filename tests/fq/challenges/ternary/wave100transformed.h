#ifndef SMALLSECRETLWE_DECODING_FILE_H
#define SMALLSECRETLWE_DECODING_FILE_H
#include <cstdint>
constexpr uint64_t n = 100;
constexpr uint64_t k = 36;
constexpr uint64_t seed = 0;
constexpr uint64_t w = 94;
constexpr const char *s = "1000011011121001200212120121112021201101211201210020022201112120";
constexpr const char *h = "1000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000001110121111120201002122201020211201112202112102002122202110212002000210012101212021202221102112010000221100200002221221200221211122210120122100021020101100020022222000202201010020002010202021201000211100210202101121202200200121021200121221211210220120120210011212122202112120020210002112221012212200101201212200111101022020012110002022200022011200112001101022221001000021211202222122121120012100011112012001002111201012001022011111002111101021101122002212000020010012211122201010120101002021200111020020001112002102122121211122022001221102012012122100200000101100211212202022202101221112200202012211011111100200121001101220220110211100100022021022210110022122120222101122020011101120001102222000222212100121001001020112101121010012011000102000022202011202000101121000111112210200221010001200011120020102001121202112100121220110121101110001022202110110011201122010102121001021220121021021120012001122211021112020001121020001112022202012212111122000201121022112121011221120110210200111020202221010020122120021001121101122101202100011111120010220100200102220021202211110222222212111210110000101200012021201012020112111020010021001222112111120012001121011110000001012012100010011110212202112212211000220021001220222100121022102222210200212101211200022112200100200012200200112202102022111100220212121201021201110001220112222011201012202201012000000210012100110110212211011222221020110020222220010000022202012022120220112210201011201021021111101212102221201120111001201222212022111002102222210012020201112111121000201010212212200211020022020222211111012112221102100202221110212200120202122102202010102102212200010121111121010101011122212021120221101120002110002022101011021120201120000122220220222221012002010112211210120022022010021201202120120202100012121112120010020012001202110220101011221212110011010100010012220112221100211022001101002210011022000022021012210002110010011120120020212000012000022002101122120012021001221011011010211201121202210102112020021012110212100111021201102112200102211120121021112010002100121102222012220000000201122121201101221220122210111121110200022222022202222012220100001211021122121002202002021220122110201012010221102101201212211100210120100112120110010100202022000122212201211020001110011210012102110010212111200112102212111112101101010201100211121220212110122001110102010102";
// load with `mzd_t *A = mzd_from_str(k, n, h)`
#endif //SMALLSECRETLWE_DECODING_FILE_H