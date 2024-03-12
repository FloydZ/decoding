#ifndef SMALLSECRETLWE_DECODING_FILE_H
#define SMALLSECRETLWE_DECODING_FILE_H
#include <cstdint>
constexpr uint64_t n = 120;
constexpr uint64_t k = 44;
constexpr uint64_t seed = 0;
constexpr uint64_t w = 118;
constexpr const char *s = "1202012100201110012111002020200110221020212112102012021121121020020021000211";
constexpr const char *h = "100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000111012111112020100212220102021120111220211210200212220211021200200021001210121202120222110211201000022110020000222122120022121112221012012210002102010110002002222200020220101002000201020202120100021110021020210112120220020012102120012122121121022012012021001121212220211212002021000211222101221220010120121220011110102202001211000202220002201120011200110102222100100002121120222212212112001210001111201200100211120101200102201111100211110102110112200221200002001001221112220101012010100202120011102002000111200210212212121112202200122110201201212210020000010110021121220202220210122111220020201221101111110020012100110122022011021110010002202102221011002212212022210112202001110112000110222200022221210012100100102011210112101001201100010200002220201120200010112100011111221020022101000120001112002010200112120211210012122011012110111000102220211011001120112201010212100102122012102102112001200112221102111202000112102000111202220201221211112200020112102211212101122112011021020011102020222101002012212002100112110112210120210001111112001022010020010222002120221111022222221211121011000010120001202120101202011211102001002100122211211112001200112101111000000101201210001001111021220211221221100022002100122022210012102210222221020021210121120002211220010020001220020011220210202211110022021212120102120111000122011222201120101220220101200000021001210011011021221101122222102011002022222001000002220201202212022011221020101120102102111110121210222120112011100120122221202211100210222221001202020111211112100020101021221220021102002202022221111101211222110210020222111021220012020212210220201010210221220001012111112101010101112221202112022110112000211000202210101102112020112000012222022022222101200201011221121012002202201002120120212012020210001212111212001002001200120211022010101122121211001101010001001222011222110021102200110100221001102200002202101221000211001001112012002021200001200002200210112212001202100122101101101021120112120221010211202002101211021210011102120110211220010221112012102111201000210012110222201222000000020112212120110122122012221011112111020002222202220222201222010000121102112212100220200202122012211020101201022110210120121221110021012010011212011001010020202200012221220121102000111001121001210211001021211120011210221211111210110101020110021112122021211012200111010201010210002100001200022002100221020001110101021021211212022010201020220102210122020210211012101110221010111012021101011100111212022001222000101201120110010122021022222102221220210012122000000122022021221211200121211121100000112201200100122120202022111220022100121102002022101102110112210210001222102111120012200002100211101112202220122001021101022101202020102010220022010222110121200110122002112100201002102211012210021101021221000210211221020220120111201121120110200211000100010121020111111200221000010021200021200202202021220210010121012210002000202112111211200202000202101220222102012111222100120211111011211221121221220010120202222211111122020112000201000202121022002010112110210200022112010100120222122000120222020111222020101221112112010021221211202102202002112202111021220002000111012002122200010021022211120212020202002111101202020021002001021012001011121122001111212120002121000001220202121122112122212102021112002010112020220021010111110202212212200212201212210021222021121122021101110010221102122010012022220110202120202101112011120000";
// load with `mzd_t *A = mzd_from_str(k, n, h);`
#endif //SMALLSECRETLWE_DECODING_FILE_H