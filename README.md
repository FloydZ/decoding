This is the implementation of the [paper](https://eprint.iacr.org/2021/1634) and our second [paper](https://eprint.iacr.org/2022/1329).

Requirements
----
Your need an `C++20` ready compiler. 

Arch Linux:
```bash
sudo pacman -S cmake make autoconf automake
```

Ubuntu:
```bash
sudo apt install autoconf automake make cmake libpng-dev libtool libmpfr-dev libmpfr-doc libmpfr6 libmpfrc++-dev libomp-dev clang libgtest-dev 
```

CentOS:
```bash
sudo yum install yum-utils
sudo yum-config-manager --enable extras
sudo yum makecache
sudo yum install autoconf automake cmake clang gcc gcc-c++ libtool gmp-devel mpfr mpfr-devel git libarchive cuda-command-line-tools-11-6  cuda-nvcc-11-6.x86_64 vim cuda-libraries-devel-11-6.x86_64 cuda-libraries-11-6.x86_64 tmux cuda-compiler-11-6.x86_64 cuda-demo-suite-11-6-11.6.55-1.x86_64 cuda-minimal-build-11-6.x86_64 epel-release htop python39-pip cuda-curand-dev-10-2.x86_64 cuda-libraries-devel-11-6.x86_64 cuda-samples-11-6.x86_64 libpng

# install cuda sample
cd /usr/local/cuda/sample/
sudo git clone https://github.com/nvidia/cuda-samples

# install gtest
cd ~
git clone https://github.com/google/googletest
cd googletest
mkdir build
cd build
cmake ..
make 
sudo make install
cd ~

# fix libpng
cd /usr/lib64
sudo ln -s  libpng16.so.16.34.0 libpng.so
```

MacOS:
```bash
brew insatll cmake make tbb gcc googletest autoconf automake libtool libopenmpt libomp llvm
pip3 install Cython cython
```
To use the bundled libc++ please add the following LDFLAGS:
```
LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"
```
llvm is keg-only, which means it was not symlinked into /usr/local,
because macOS already provides this software and installing another version in
parallel can cause all kinds of trouble.

If you need to have llvm first in your PATH, run:
```
echo 'export PATH="/usr/local/opt/llvm/bin:$PATH"' >> ~/.zshrc
```

For compilers to find llvm you may need to set:
```
export LDFLAGS="-L/usr/local/opt/llvm/lib"
export CPPFLAGS="-I/usr/local/opt/llvm/include"
```

Windows:
you probably want to reevaluate some life decisions.


Python
---
```bash
pip install scipy numpy pprint
```

Reproduction of results:
---
Important: You need a C++20 rdy compiler. For our records we used `clang++-13-rc1`. But 

```bash
# its important to name the build directory `cmake-build-release`. Its hardcoded in some file... yeah i know, its a todo
git clone --recurse-submodules -j4 git@git.noc.ruhr-uni-bochum.de:cits/decoding.git
cd decoding && mkdir cmake-build-release && cd cmake-build-release && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang .. && cd ..
chmod +x setup.sh
./setup.sh

# generates the necessary main.h file (options were used for the record, without the `--loops` parameter)
# set the number of outer threads
python3 gen.py -n1284 -l17 -l1 2 -p 1 --hm1_bucketsize 146 --hm2_bucketsize 4 --hm1_nrbuckets 2 --hm2_nrbuckets 15 --bjmm_special_alignment --threads 1 --outer_threads 256 --loops 100000
#or run this command if you want to break the 1223 instance (options were used for the record, without the `--loops` parameter): 
#python3 gen.py -n1223 -l17 -l1 2 -p 1 --hm1_bucketsize 150 --hm2_bucketsize 5 --hm1_nrbuckets 2 --hm2_nrbuckets 15 --bjmm_special_alignment --threads 1 --outer_threads 256 --loops 100000

cd cmake-build-release
make main_profile1

# this command runs for several minutes to create a performance file.
./main_profile1
llvm-profdata merge -output=default.profdata default_*.profraw
make main_profile2

# this command runs for several days, probably you want to run it inside a tmux/screen env.
./main_profile2
```

To recompute our the records from our second paper use the following command:
```bash
python3 gen.py -n3138 -l24 -l1 9  -p 1 --lowweight_w 56 --hm1_bucketsize 7   --hm2_bucketsize 3 --hm1_nrbuckets 9  --hm2_nrbuckets 15 --intermediate_target_loops 171 --quasicyclic --threads 1 --outer_threads 1 --print_loops 100 --seconds 600  --bjmm_fulllength
```

if you only have an ancient compiler which does not support `PGO` replace the `make` commands with:
```bash
make main
# this command runs for several days, probably you want to run it inside a tmux/screen env.
./main
```

High Memory
---
The following commands where used for the high memory benchmark. Note: most of them need around 2TB of memory. Set the number of threads working in parallel on different permutations accordingly to reduce the memory consumption.
```bash
python3 gen.py -n1223 -l49 -l1 24 -p 3 --hm1_bucketsize 24 --hm2_bucketsize 25 --hm1_nrbuckets 4 --hm2_nrbuckets 4 --threads 2 --outer_threads 1 --bjmm_special_alignment --force_huge_page --seconds 600 --benchmark
python3 gen.py -n1284 -l49 -l1 25 -p 3 --hm1_bucketsize 25 --hm2_bucketsize 24 --hm1_nrbuckets 4 --hm2_nrbuckets 4 --threads 2 --outer_threads 1 --bjmm_special_alignment --force_huge_page --seconds 600 --benchmark
python3 gen.py -n1409 -l50 -l1 25 -p 3 --hm1_bucketsize 25 --hm2_bucketsize 25 --hm1_nrbuckets 1 --hm2_nrbuckets 1 --threads 2 --outer_threads 1 --force_huge_page --seconds 600 --benchmark 
python3 gen.py -n1473 -l51 -l1 25 -p 3 --hm1_bucketsize 25 --hm2_bucketsize 26 --hm1_nrbuckets 1 --hm2_nrbuckets 1 --threads 2 --outer_threads 1 --force_huge_page --seconds 600 --benchmark
python3 gen.py -n1536 -l51 -l1 25 -p 3 --hm1_bucketsize 25 --hm2_bucketsize 26 --hm1_nrbuckets 1 --hm2_nrbuckets 1 --threads 2 --outer_threads 1 --force_huge_page --seconds 600 --benchmark
python3 gen.py -n1600 -l52 -l1 25 -p 3 --hm1_bucketsize 25 --hm2_bucketsize 27 --hm1_nrbuckets 1 --hm2_nrbuckets 1 --threads 2 --outer_threads 1 --force_huge_page --seconds 600 --benchmark
python3 gen.py -n1665 -l52 -l1 25 -p 3 --hm1_bucketsize 25 --hm2_bucketsize 27 --hm1_nrbuckets 1 --hm2_nrbuckets 1 --threads 2 --outer_threads 1 --force_huge_page --seconds 600 --benchmark
python3 gen.py -n1730 -l52 -l1 25 -p 3 --hm1_bucketsize 25 --hm2_bucketsize 27 --hm1_nrbuckets 1 --hm2_nrbuckets 1 --threads 2 --outer_threads 1 --force_huge_page --seconds 600 --benchmark
python3 gen.py -n1796 -l52 -l1 25 -p 3 --hm1_bucketsize 25 --hm2_bucketsize 27 --hm1_nrbuckets 1 --hm2_nrbuckets 1 --threads 2 --outer_threads 1 --force_huge_page --seconds 600 --benchmark
```
It can be faster for your system to include `--force_huge_page` and `--force_container_alignment`. In general you need to play a little with the flags, to get the best results.

Quasi Cyclic:
===
The following commands were used to break the quasi cyclic challenges. Follow the guide above to apply `PGO`.
```bash
python3 gen.py -n2118 -l20 -l1 1 -p 1 -w1 46 --hm1_bucketsize 300 --hm2_bucketsize 2 --hm1_nrbuckets 1 --hm2_nrbuckets 19 --quasicyclic --threads 1 --outer_threads 256
python3 gen.py -n2306 -l20 -l1 1 -p 1 -w1 48 --hm1_bucketsize 320 --hm2_bucketsize 2 --hm1_nrbuckets 1 --hm2_nrbuckets 19 --quasicyclic --threads 1 --outer_threads 256
python3 gen.py -n2502 -l21 -l1 1 -p 1 -w1 50 --hm1_bucketsize 340 --hm2_bucketsize 2 --hm1_nrbuckets 1 --hm2_nrbuckets 20 --quasicyclic --threads 1 --outer_threads 256
python3 gen.py -n2706 -l21 -l1 1 -p 1 -w1 52 --hm1_bucketsize 360 --hm2_bucketsize 2 --hm1_nrbuckets 1 --hm2_nrbuckets 20 --quasicyclic --threads 1 --outer_threads 256
python3 gen.py -n2918 -l21 -l1 1 -p 1 -w1 54 --hm1_bucketsize 360 --hm2_bucketsize 2 --hm1_nrbuckets 1 --hm2_nrbuckets 20 --quasicyclic --threads 1 --outer_threads 256
```

FLAGS
---
A lists of preprocessor options you can set.
```bash
USE_LOOPS                   Hardcode the number of maximum loops the algorithm should compute
USE_AVX2_SPECIAL_ALIGNMENT  Enforces alignment of 256 => Uses faster AVX2 Instructions 
BJMM_DOOM_SPECIAL_FORM      Unused
USE_AVX2                    Enables AVX2 optimisations (MUST be activated)
USE_PREFETCH                Enables memory prefetching in a few places
CUSTOM_ALIGNMENT 4096       Ensures that the baselists are 4096 byte aligned
BINARY_CONTAINER_ALIGNMENT  Alignes the container for binary data to the given number
NUMBER_THREADS              Number of threads working on the same tree
NUMBER_OUTER_THREADS        Number of threads starting a seperate instance of the programm  

# LOGGING
CHALLENGE                   Disables all internal Logging=> Except current loop number
NO_LOGGING                  Disable all internal logging.
BENCHMARK                   Disable all internal logging except the time to build the table.

# Instance / Algorithm Type
USE_DOOM                    Internaly the 4 list is replaced with the list of syndroms
USE_NN                      Stream join in a NN manner.
USE_MO                      Uses a different implementation.
SYNDROM                     Syndrome decoding challenge
LOW_WEIGHT                  Low Weight vhallenge
```

AES
---
To get the numbers from the paper, run the following commands.
```bash
openssl speed -multi 256 -bytes 8  -seconds 60 aes
options:bn(64,64) rc4(8x,int) des(int) aes(partial) blowfish(ptr) 
compiler: gcc -fPIC -pthread -m64 -Wa,--noexecstack -Wall -Wa,--noexecstack -g -O2 -fdebug-prefix-map=/build/openssl-nwsL4a/openssl-1.1.1=. -fstack-protector-strong -Wformat -Werror=format-security -DOPENSSL_USE_NODELETE -DL_ENDIAN -DOPENSSL_PIC -DOPENSSL_CPUID_OBJ -DOPENSSL_IA32_SSE2 -DOPENSSL_BN_ASM_MONT -DOPENSSL_BN_ASM_MONT5 -DOPENSSL_BN_ASM_GF2m -DSHA1_ASM -DSHA256_ASM -DSHA512_ASM -DKECCAK1600_ASM -DRC4_ASM -DMD5_ASM -DAES_ASM -DVPAES_ASM -DBSAES_ASM -DGHASH_ASM -DECP_NISTZ256_ASM -DX25519_ASM -DPADLOCK_ASM -DPOLY1305_ASM -DNDEBUG -Wdate-time -D_FORTIFY_SOURCE=2
aes-128 cbc    7625124.45k
aes-192 cbc    6579466.92k
aes-256 cbc    5769314.16k

openssl speed -multi 256 -bytes 16 -seconds 60 aes
options:bn(64,64) rc4(8x,int) des(int) aes(partial) blowfish(ptr) 
compiler: gcc -fPIC -pthread -m64 -Wa,--noexecstack -Wall -Wa,--noexecstack -g -O2 -fdebug-prefix-map=/build/openssl-nwsL4a/openssl-1.1.1=. -fstack-protector-strong -Wformat -Werror=format-security -DOPENSSL_USE_NODELETE -DL_ENDIAN -DOPENSSL_PIC -DOPENSSL_CPUID_OBJ -DOPENSSL_IA32_SSE2 -DOPENSSL_BN_ASM_MONT -DOPENSSL_BN_ASM_MONT5 -DOPENSSL_BN_ASM_GF2m -DSHA1_ASM -DSHA256_ASM -DSHA512_ASM -DKECCAK1600_ASM -DRC4_ASM -DMD5_ASM -DAES_ASM -DVPAES_ASM -DBSAES_ASM -DGHASH_ASM -DECP_NISTZ256_ASM -DX25519_ASM -DPADLOCK_ASM -DPOLY1305_ASM -DNDEBUG -Wdate-time -D_FORTIFY_SOURCE=2
aes-128 cbc   16892817.80k
aes-192 cbc   14375882.78k
aes-256 cbc   12473219.60k

openssl speed -multi 256 -bytes 32 -seconds 60 aes
options:bn(64,64) rc4(8x,int) des(int) aes(partial) blowfish(ptr)
compiler: gcc -fPIC -pthread -m64 -Wa,--noexecstack -Wall -Wa,--noexecstack -g -O2 -fdebug-prefix-map=/build/openssl-Flav1L/openssl-1.1.1=. -fstack-protector-strong -Wformat -Werror=format-security -DOPENSSL_USE_NODELETE -DL_ENDIAN -DOPENSSL_PIC -DOPENSSL_CPUID_OBJ -DOPENSSL_IA32_SSE2 -DOPENSSL_BN_ASM_MONT -DOPENSSL_BN_ASM_MONT5 -DOPENSSL_BN_ASM_GF2m -DSHA1_ASM -DSHA256_ASM -DSHA512_ASM -DKECCAK1600_ASM -DRC4_ASM -DMD5_ASM -DAES_ASM -DVPAES_ASM -DBSAES_ASM -DGHASH_ASM -DECP_NISTZ256_ASM -DX25519_ASM -DPADLOCK_ASM -DPOLY1305_ASM -DNDEBUG -Wdate-time -D_FORTIFY_SOURCE=2
aes-128 cbc   17863577.85k
aes-192 cbc   15071777.05k
aes-256 cbc   12982587.16k
```

Optimisation points
-----
- Simplify the syndrome extraction process by inverting all rows (including the ones of the syndrome). Meaning that the first row is now the last, and the last is now the first.
- better weight checks for small l impl. instead of just fetching l bits and save them in the hashmap, fetch a little more, as much as fit. now do the weight check on this limb. if its small enough recompute the whole label from the baselists.
- and a lot more...

Performance Graphs:
---
A little helper to find performance holes. Note that you should compile wile `-fno_inline` to get better results.
```bash
flamegraph -o mceliece_1284_l19_noinline.svg ./test/mceliece/mceliece_test_bjmm1284 --gtest_filter='BJMM.t1284_small:BJMM/*.t1284_small:*/BJMM.t1284_small/*:*/BJMM/*.t1284_small --gtest_color=no'
```

Bolt
----
That was an idea of me. Did not work.
```bash
python gen.py -n431 -l13 -l1 2 -p 1 --bjmm_hm1_bucketsize 1024 --bjmm_hm2_bucketsize 16 --bjmm_hm1_nrbuckets 2 --bjmm_hm2_nrbuckets 11 --threads 1 --bjmm_outer_threads 1 --bench --loops 10 

Bolt Optimisation
------
```bash
perf record -e cycles:u -j any,u -o perf.data --  ./mceliece_test_bjmm1284 "--gtest_filter=BJMM.t1284_small_normal:BJMM/*.t1284_small_normal:*/BJMM.t1284_small_normal/*:*/BJMM/*.t1284_small_normal --gtest_color=no"
perf2bolt -p perf.data mceliece_test_bjmm1284  -o perf.fdata
llvm-bolt mceliece_test_bjmm1284 -o mceliece_test_bjmm1284.bolt data=perf.fdata -reorder-blocks=cache+ -reorder-functions=hfsort -split-functions=2 -split-all-cold -split-eh -dyno-stats
```

Implemented Algorithms
=====
- `MMT` and `BJMM` depth 2 in `src/bjmm.h`
```bash
                           n-k-l                                        n          0
┌─────────────────────────────┬─────────────────────────────────────────┐ ┌──────┐
│             I_n-k-l         │                  H                      │ │  0   │
├─────────────────────────────┼─────────────────────────────────────────┤ ├──────┤ n-k-l
│              0              │                                         │ │      │
└─────────────────────────────┴─────────────────────────────────────────┘ └──────┘  n-k
             w-p                                  p
┌─────────────────────────────┬─────────────────────────────────────────┐
│              e1             │                    e2                   │
└─────────────────────────────┴────────────────────┬────────────────────┘
                                                   │
                                             ┌─────┴────┐  
                                         ┌───┴───┐   ┌──┴────┐
                                         │  iL   │   │       │
                                         └──┬────┘   └───┬───┘ 
                                    ┌───────┴─┐         ┌┴────────┐
                                ┌───┼───┐ ┌───┴───┐ ┌───┴───┐ ┌───┴───┐
                                │ L1│   │ │  L2   │ │  L3   │ │   L4  │
                                │   │   │ │       │ │       │ │       │
                                └───┴───┘ └───────┘ └───────┘ └───────┘
```
        - L3 and L4 do not exist, L1 and L2 are simply reused
        - The right intermediate List do not exist. So actually we implemented a stream join approach
        - The output list do not exist. Every element is directly checked
        - List L1 is hashed. So the Join between L1 and L2 is done with hashmaps
        - iL is actually a hashmap

- `MMT` and `BJMM` depth 3 in `src/bjmm_d3.h`
```
															Level
                        ┌───┐
                            │
                        │
                        └─▲─┘
                          │
            ┌─────────────┴────────────┐						0
            │                          │
          ┌─┴─┐                      ┌─┴─┐
          │   │ HM2                      │
          │   │                      │
          └─▲─┘                      └─▲─┘
            │                          │
     ┌──────┴─────┐              ┌─────┴──────┐					1
     │            │              │            │
   ┌─┴─┐        ┌─┴─┐          ┌─┴─┐        ┌─┼─┐
   │   │ HM1        │              │ HM1        │
   │   │        │              │            │
   └─▲─┘        └─▲─┘          └─▲─┘        └─▲─┘
     │            │              │            │
  ┌──┴──┐      ┌──┴──┐        ┌──┴──┐      ┌──┴──┐				2
  │     │      │     │        │     │      │     │
┌─┴─┐ ┌─┴─┐  ┌─┴─┐ ┌─┴─┐    ┌─┴─┐ ┌─┴─┐  ┌─┴─┐ ┌─┴─┐		base lists
│HM0│ │   │      │     │        │     │      │     │
│   │ │   │  │     │        │     │      │     │
└───┘ └───┘  └───┘ └───┘    └───┘ └───┘  └───┘ └───┘
  L1	L2	  L1	L2		  L1	L2	   L1 	 L2
```
        - Only L1, L2 exist, every other List is just L1 or L2 with a different intermediate target
        - in the intermediate levels only hashmaps are used
        - the output list do not exists. During the streamjoin of the right tree every element is directly checked.
- `dumer`, `prange` in `dumer.h` and `prange.h`
- memory less LeeBrickell in `pollard.h`
- Hybrid Tree
- Classical Tree
- Hybrid Tree
- Nearest Neighbour Stream Join approach
- sparsity

TODO
====
Explain 
```
python3 --bench
python3 --optimize
python3 --calc_loops
```

Core Binding
----
Either via
    - `sudo taskset -c 1 ./main`
or 
    - `OMP_PLACES=cores OMP_PROC_BIND=close ./main`, `OMP_PLACES=cores OMP_PROC_BIND=spread ./main`
or 
    - `for j in {0..127}; do sudo taskset -c ${j} ./main & done` 

Polly/Bolt
---
cmake -DCMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=~/Downloads/BOLT/build_server1/bin/clang++  -DCMAKE_LINKER=clang++-14 -DCMAKE_CXX_LINK_EXECUTABLE="<CMAKE_LINKER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>"  ..


~/Downloads/BOLT/build_server1/bin/llvm-bolt main_bolt -instrument -jump-tables=move -o main_bolt_out -align-macro-fusion=all
~/Downloads/BOLT/build_server1/bin/llvm-bolt main_bolt -o main_bolt.bolt -data=/tmp/prof.fdata -reorder-blocks=cache+ -reorder-functions=hfsort -split-functions=2 -split-all-cold -split-eh -dyno-stats


~/Downloads/BOLT/build_server1/bin/llvm-bolt main_bolt -o main_bolt.bolt -data=/tmp/prof.fdata -reorder-blocks=cache+ -reorder-functions=hfsort -split-functions=2 -split-all-cold -split-eh -dyno-stats -jump-tables=move -align-macro-fusion=all --simplify-conditional-tail-calls --peepholes=all --hugify --icp-eliminate-loads --icf


RESEARCH:
=========
Currently we are investigating time memory tradeoffs via a pollard rho implementation.

Benchmarks from 12.12.22:
Laptop:
n,p,l
431,1,13: old: 		37perms/s
431,1,13: new: 		7500perms/s
431,1,13: opt_old:  7700perms/s

431,2,17: new: 		40perms/s
431,2,17: opt_old: 	100perms/s

Server
431,2,17: new: 		182perms/s
431,2,17: opt_old: 	450perms/s
