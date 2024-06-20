#ifndef SMALLSECRETLWE_DECODING_FILE_H
#define SMALLSECRETLWE_DECODING_FILE_H
#include <cstdint>
constexpr uint64_t n = 200;
constexpr uint64_t k = 73;
constexpr uint64_t seed = 0;
constexpr uint64_t w = 189;
constexpr const char *s = "2112102010100002222122120122020200001221222012122211120120221001102212221012112222012110020201110200001112210022011111222122020";
constexpr const char *h = "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011101211111202010021222010202112011122021121020021222021102120020002100121012120212022211021120100002211002000022212212002212111222101201221000210201011000200222220002022010100200020102020212010002111002102021011212022002001210212001212212112102201201202100112121222021121200202100021122210122122001012012122001111010220200121100020222000220112001120011010222210010000212112022221221211200121000111120120010021112010120010220111110021111010211011220022120000200100122111222010101201010020212001110200200011120021021221212111220220012211020120121221002000001011002112122020222021012211122002020122110111111002001210011012202201102111001000220210222101100221221202221011220200111011200011022220002222121001210010010201121011210100120110001020000222020112020001011210001111122102002210100012000111200201020011212021121001212201101211011100010222021101100112011220101021210010212201210210211200120011222110211120200011210200011120222020122121111220002011210221121210112211201102102001110202022210100201221200210011211011221012021000111111200102201002001022200212022111102222222121112101100001012000120212010120201121110200100210012221121111200120011210111100000010120121000100111102122021122122110002200210012202221001210221022222102002121012112000221122001002000122002001122021020221111002202121212010212011100012201122220112010122022010120000002100121001101102122110112222210201100202222200100000222020120221202201122102010112010210211111012121022212011201110012012222120221110021022222100120202011121111210002010102122122002110200220202222111110121122211021002022211102122001202021221022020101021022122000101211111210101010111222120211202211011200021100020221010110211202011200001222202202222210120020101122112101200220220100212012021201202021000121211121200100200120012021102201010112212121100110101000100122201122211002110220011010022100110220000220210122100021100100111201200202120000120000220021011221200120210012210110110102112011212022101021120200210121102121001110212011021122001022111201210211120100021001211022220122200000002011221212011012212201222101111211102000222220222022220122201000012110211221210022020020212201221102010120102211021012012122111002101201001121201100101002020220001222122012110200011100112100121021100102121112001121022121111121011010102011002111212202121101220011101020101021000210000120002200210022102000111010102102121121202201020102022010221012202021021101210111022101011101202110101110011121202200122200010120112011001012202102222210222122021001212200000012202202122121120012121112110000011220120010012212020202211122002210012110200202210110211011221021000122210211112001220000210021110111220222012200102110102210120202010201022002201022211012120011012200211210020100210221101221002110102122100021021122102022012011120112112011020021100010001012102011111120022100001002120002120020220202122021001012101221000200020211211121120020200020210122022210201211122210012021111101121122112122122001012020222221111112202011200020100020212102200201011211021020002211201010012022212200012022202011122202010122111211201002122121120210220200211220211102122000200011101200212220001002102221112021202020200211110120202002100200102101200101112112200111121212000212100000122020212112211212221210202111200201011202022002101011111020221221220021220121221002122202112112202110111001022110212201001202222011020212020210111201112000020210121211122221000101222112111020202020222122020000210212022201100200001110221000210020122120210222212200002220100120210201121022212200111122101210010220202101122002011112011221202222020121221010100210221220000002110202001000202022222121212122201102200102200001102121112222120201122200010011022122002112201001221222110010122020211020211200001210011202120010200222011221221002022220122000022202022000101212121020010111211012000220020110112220212222101012210211010101210212211222200010121011121021102001122020111220110200121102111112002100022121012201021201120222100001110012211201100001201202100221200020100112012202100010210121012222101202120112112212211100011201122120001220012212001111100200102022222121001021221200211122210001121122201112021212101211222211211122010111102010102100100020020022212101111200100201212111021110022101212010100010110002002012020022000202200011001212202001212002202200211212222110220021121121000012220200012120110002121020210211020222011011210210101210102120121021022011220010222011220100201010220022012212211100110102222222120202211100211122220102020120001011221122111012111211220122201122120012012001212121212222120202101100002120111201111222001211222010200010001102200212021011011010022012121002111200111121101012010011111100102101002221210021011220011111002212002010022120012011120200002012010220021102121120211002002212212012111001010000110100100022121122122110222010222011000122222222111210110222021000200001120020020112222110120220201201000220111112111121121121112002111122200100200122020210101000202202202002021100110122020221210010202002112101212001000110112010121101201100211022002221221201111010100111201102101010200121220110221200000012200111121122011102210222121021100221001210112201020011002202001212220002220110201002020012021222001202121202102110111012010202210101211002212020200011202022000111010212120212020112200112020000011121002211010210101102212210222002110220200010100222202110100202122110021012021111012011010000210212011022021201202120021221000122101102220010111100121200111121101012012000101201221202012121000220022011120102201020100212002220221222111200021122102112220100201211110121201211210012221120201010000221021020021122221001121002100002200102001112122212212200020000002202221121102222212200212121220122011221201221221110211202112002120202220001210220120121102121111022210120011111222020211110020201111012101102211122101012012121101001002011122010201102010221201000011222101121110022012202110202202002210202121101100121212101002002120002020000201101112002011102221101012100202020122222020111110111010101101112101000212122200100112001122010220121121110100102210210201011112012012010101212000210122220012221121220022121110111122010201102210212101100010010220000121100121011122222102200121211220011202121200001101122212200220012210000210122201010020102200110002112020212112012020220112000110210012001101000002201122010020102101102111012020211111002221002122101101012101012022202001211120210112010012220222022202200221100000200121100221000122100000121112020222102122001121212211220012201210211111210011000120101101211021120020211011002001111210112220000220021102110201020000201112201122000202110102122011011100202202022221110220121022200101111002200122220122200002211122112110021202000110212121122020212022110202120120120111100221212022220012222102202012010212211000210110011220202011011212020102011110220222010010010000111200212120201110222102110220012101120002210101222121000210122120022112021210011212120210112122112021011112021101220202212002022222212110021111020001200112001001200010100020220011102002200201101102001212220222202212000100111212001100012021222210000001022220111001011222220020111100111100110001122000220201221021012201201012112222012201111001112212220001000011200222010011111111201202101221100202112002112002122220011000002201101212012010102212202112221010011200202010101221210202002010100202111121122222002222201111100022111022022021020122201202110021102121011111112100110010110210222222110002100100002212110022100122001121022022110011211111021011211011120111221002212121122101212120121201021221022012202202021020111012202202222111122220100012001102210010022010121202021000101111110221220211102021221211202000010220221212120222222100210121222000112120110222221221102011102100101212200202022201002111100121222202222110121022221021020120110112220112101122211110010001000020220201020010011202100002002202000112201212220110221012000221110202010010001020021112101220211112020022110212010001221011200010220000122021120011102022020120021211000202011000211200002200010121011200122202202112220201021201221021010011022112200201022011211022112002010000112000022021122102222121120011101010012201020211000202120022112122021011012121021121111020102022011221212210211020211200002220212120011211012221121022020120020201011110211010001121021120221002110010212101200001201101022021111102002002022022120021021212022210000012111101001102011101211202111120201222021201220002100011021101121020021111220202112001202212121011000021100011220020220211011101212100221122110002012001022100000201111121101210212212110202002210002021111110012111000012020121220221200021120012221121121120202101000201120121202210220121212000121200022122002020020020020000011210120121021000112000111012102200010122122200200100120022101110212021221121121001200200100220202111121222100202012010021121001000100202002000121010022222000021112202211000201002010011110122211110221011002122220220220021102011002111112112020000012121101012212212212122020102222221102201111201021002221012121122210100022222101120112110110112212101210200112010100011220212010100100111112122000011022212021101111010011111202020010211000211122122102021001101222100122101021102121211111210000021102120200212210210011001010201020111022011220121201122111012011000001101000111201111112212110220211002001110100002212012211001220021000000220101021221002222201100222122101222200201202112212110112122021002221201111222122000210120200";
// load with `mzd_t *A = mzd_from_str(k, n, h)`
#endif //SMALLSECRETLWE_DECODING_FILE_H