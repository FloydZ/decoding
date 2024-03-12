#ifndef SMALLSECRETLWE_DECODING_FILE_H
#define SMALLSECRETLWE_DECODING_FILE_H
#include <cstdint>
constexpr uint64_t n = 130;
constexpr uint64_t k = 47;
constexpr uint64_t seed = 0;
constexpr uint64_t w = 123;
constexpr const char *s = "02112202011222011001020100000201010122211211222210202001011112112122022101201111221";
constexpr const char *h = "10000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000011101211111202010021222010202112011122021121020021222021102120020002100121012120212022211021120100002211002000022212212002212111222101201221000210201011000200222220002022010100200020102020212010002111002102021011212022002001210212001212212112102201201202100112121222021121200202100021122210122122001012012122001111010220200121100020222000220112001120011010222210010000212112022221221211200121000111120120010021112010120010220111110021111010211011220022120000200100122111222010101201010020212001110200200011120021021221212111220220012211020120121221002000001011002112122020222021012211122002020122110111111002001210011012202201102111001000220210222101100221221202221011220200111011200011022220002222121001210010010201121011210100120110001020000222020112020001011210001111122102002210100012000111200201020011212021121001212201101211011100010222021101100112011220101021210010212201210210211200120011222110211120200011210200011120222020122121111220002011210221121210112211201102102001110202022210100201221200210011211011221012021000111111200102201002001022200212022111102222222121112101100001012000120212010120201121110200100210012221121111200120011210111100000010120121000100111102122021122122110002200210012202221001210221022222102002121012112000221122001002000122002001122021020221111002202121212010212011100012201122220112010122022010120000002100121001101102122110112222210201100202222200100000222020120221202201122102010112010210211111012121022212011201110012012222120221110021022222100120202011121111210002010102122122002110200220202222111110121122211021002022211102122001202021221022020101021022122000101211111210101010111222120211202211011200021100020221010110211202011200001222202202222210120020101122112101200220220100212012021201202021000121211121200100200120012021102201010112212121100110101000100122201122211002110220011010022100110220000220210122100021100100111201200202120000120000220021011221200120210012210110110102112011212022101021120200210121102121001110212011021122001022111201210211120100021001211022220122200000002011221212011012212201222101111211102000222220222022220122201000012110211221210022020020212201221102010120102211021012012122111002101201001121201100101002020220001222122012110200011100112100121021100102121112001121022121111121011010102011002111212202121101220011101020101021000210000120002200210022102000111010102102121121202201020102022010221012202021021101210111022101011101202110101110011121202200122200010120112011001012202102222210222122021001212200000012202202122121120012121112110000011220120010012212020202211122002210012110200202210110211011221021000122210211112001220000210021110111220222012200102110102210120202010201022002201022211012120011012200211210020100210221101221002110102122100021021122102022012011120112112011020021100010001012102011111120022100001002120002120020220202122021001012101221000200020211211121120020200020210122022210201211122210012021111101121122112122122001012020222221111112202011200020100020212102200201011211021020002211201010012022212200012022202011122202010122111211201002122121120210220200211220211102122000200011101200212220001002102221112021202020200211110120202002100200102101200101112112200111121212000212100000122020212112211212221210202111200201011202022002101011111020221221220021220121221002122202112112202110111001022110212201001202222011020212020210111201112000020210121211122221000101222112111020202020222122020000210212022201100200001110221000210020122120210222212200002220100120210201121022212200111122101210010220202101122002011112011221202222020121221010100210221220000002110202001000202022222121212122201102200102200001102121112222120201122200010011022122002112201001221222110010122020211020211200001210011202120010200222011221221002022220122000022202022000101212121020010111211012000220020110112220212222101012210211010101210212211222200010121011121021102001122020111220110200121102111112002100022121012201021201";
// load with `mzd_t *A = mzd_from_str(k, n, h)`
#endif //SMALLSECRETLWE_DECODING_FILE_H