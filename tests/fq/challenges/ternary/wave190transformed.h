#ifndef SMALLSECRETLWE_DECODING_FILE_H
#define SMALLSECRETLWE_DECODING_FILE_H
#include <cstdint>
constexpr uint64_t n = 190;
constexpr uint64_t k = 70;
constexpr uint64_t seed = 0;
constexpr uint64_t w = 180;
constexpr const char *s = "022101110020121221011000120012100200211122022211020011111001121101121100220020220022211110012102110012010211201011000011";
constexpr const char *h = "100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001110121111120201002122201020211201112202112102002122202110212002000210012101212021202221102112010000221100200002221221200221211122210120122100021020101100020022222000202201010020002010202021201000211100210202101121202200200121021200121221211210220120120210011212122202112120020210002112221012212200101201212200111101022020012110002022200022011200112001101022221001000021211202222122121120012100011112012001002111201012001022011111002111101021101122002212000020010012211122201010120101002021200111020020001112002102122121211122022001221102012012122100200000101100211212202022202101221112200202012211011111100200121001101220220110211100100022021022210110022122120222101122020011101120001102222000222212100121001001020112101121010012011000102000022202011202000101121000111112210200221010001200011120020102001121202112100121220110121101110001022202110110011201122010102121001021220121021021120012001122211021112020001121020001112022202012212111122000201121022112121011221120110210200111020202221010020122120021001121101122101202100011111120010220100200102220021202211110222222212111210110000101200012021201012020112111020010021001222112111120012001121011110000001012012100010011110212202112212211000220021001220222100121022102222210200212101211200022112200100200012200200112202102022111100220212121201021201110001220112222011201012202201012000000210012100110110212211011222221020110020222220010000022202012022120220112210201011201021021111101212102221201120111001201222212022111002102222210012020201112111121000201010212212200211020022020222211111012112221102100202221110212200120202122102202010102102212200010121111121010101011122212021120221101120002110002022101011021120201120000122220220222221012002010112211210120022022010021201202120120202100012121112120010020012001202110220101011221212110011010100010012220112221100211022001101002210011022000022021012210002110010011120120020212000012000022002101122120012021001221011011010211201121202210102112020021012110212100111021201102112200102211120121021112010002100121102222012220000000201122121201101221220122210111121110200022222022202222012220100001211021122121002202002021220122110201012010221102101201212211100210120100112120110010100202022000122212201211020001110011210012102110010212111200112102212111112101101010201100211121220212110122001110102010102100021000012000220021002210200011101010210212112120220102010202201022101220202102110121011102210101110120211010111001112120220012220001012011201100101220210222221022212202100121220000001220220212212112001212111211000001122012001001221202020221112200221001211020020221011021101122102100012221021111200122000021002111011122022201220010211010221012020201020102200220102221101212001101220021121002010021022110122100211010212210002102112210202201201112011211201102002110001000101210201111112002210000100212000212002022020212202100101210122100020002021121112112002020002021012202221020121112221001202111110112112211212212200101202022222111111220201120002010002021210220020101121102102000221120101001202221220001202220201112220201012211121120100212212112021022020021122021110212200020001110120021222000100210222111202120202020021111012020200210020010210120010111211220011112121200021210000012202021211221121222121020211120020101120202200210101111102022122122002122012122100212220211211220211011100102211021220100120222201102021202021011120111200002021012121112222100010122211211102020202022212202000021021202220110020000111022100021002012212021022221220000222010012021020112102221220011112210121001022020210112200201111201122120222202012122101010021022122000000211020200100020202222212121212220110220010220000110212111222212020112220001001102212200211220100122122211001012202021102021120000121001120212001020022201122122100202222012200002220202200010121212102001011121101200022002011011222021222210101221021101010121021221122220001012101112102110200112202011122011020012110211111200210002212101220102120112022210000111001221120110000120120210022120002010011201220210001021012101222210120212011211221221110001120112212000122001221200111110020010202222212100102122120021112221000112112220111202121210121122221121112201011110201010210010002002002221210111120010020121211102111002210121201010001011000200201202002200020220001100121220200121200220220021121222211022002112112100001222020001212011000212102021021102022201101121021010121010212012102102201122001022201122010020101022002201221221110011010222222212020221110021112222010202012000101122112211101211121122012220112212001201200121212121222212020210110000212011120111122200121122201020001000110220021202101101101002201212100211120011112110101201001111110010210100222121002101122001111100221200201002212001201112020000201201022002110212112021100200221221201211100101000011010010002212112212211022201022201100012222222211121011022202100020000112002002011222211012022020120100022011111211112112112111200211112220010020012202021010100020220220200202110011012202022121001020200211210121200100011011201012110120110021102200222122120111101010011120110210101020012122011022120000001220011112112201110221022212102110022100121011220102001100220200121222000222011020100202001202122200120212120210211011101201020221010121100221202020001120202200011101021212021202011220011202000001112100221101021010110221221022200211022020001010022220211010020212211002101202111101201101000021021201102202120120212002122100012210110222001011110012120011112110101201200010120122120201212100022002201112010220102010021200222022122211120002112210211222010020121111012120121121001222112020101000022102102002112222100112100210000220010200111212221221220002000000220222112110222221220021212122012201122120122122111021120211200212020222000121022012012110212111102221012001111122202021111002020111101210110221112210101201212110100100201112201020110201022120100001122210112111002201220211020220200221020212110110012121210100200212000202000020110111200201110222110101210020202012222202011111011101010110111210100021212220010011200112201022012112111010010221021020101111201201201010121200021012222001222112122002212111011112201020110221021210110001001022000012110012101112222210220012121122001120212120000110112221220022001221000021012220101002010220011000211202021211201202022011200011021001200110100000220112201002010210110211101202021111100222100212210110101210101202220200121112021011201001222022202220220022110000020012110022100012210000012111202022210212200112121221122001220121021111121001100012010110121102112002021101100200111121011222000022002110211020102000020111220112200020211010212201101110020220202222111022012102220010111100220012222012220000221112211211002120200011021212112202021202211020212012012011110022121202222001222210220201201021221100021011001122020201101121202010201111022022201001001000011120021212020111022210211022001210112000221010122212100021012212002211202121001121212021011212211202101111202110122020221200202222221211002111102000120011200100120001010002022001110200220020110110200121222022220221200010011121200110001202122221000000102222011100101122222002011110011110011000112200022020122102101220120101211222201220111100111221222000100001120022201001111111120120210122110020211200211200212222001100000220110121201201010221220211222101001120020201010122121020200201010020211112112222200222220111110002211102202202102012220120211002110212101111111210011001011021022222211000210010000221211002210012200112102202211001121111102101121101112011122100221212112210121212012120102122102201220220202102011101220220222211112222010001200110221001002201012120202100010111111022122021110202122121120200001022022121212022222210021012122200011212011022222122110201110210010121220020202220100211110012122220222211012102222102102012011011222011210112221111001000100002022020102001001120210000200220200011220121222011022101200022111020201001000102002111210122021111202002211021201000122101120001022000012202112001110202202012002121100020201100021120000220001012101120012220220211222020102120122102101001102211220020102201121102211200201000011200002202112210222212112001110101001220102021100020212002211212202101101212102112111102010202201122121221021102021120000222021212001121101222112102202012002020101111021101000112102112022100211001021210120000120110102202111110200200202202212002102121202221000001211110100110201110121120211112020122202120122000210001102110112102002111122020211200120221212101100002110001122002022021101110121210022112211000201200102210000020111112110";
// load with `mzd_t *A = mzd_from_str(k, n, h)`
#endif //SMALLSECRETLWE_DECODING_FILE_H