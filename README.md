This repository contains implementations of the fastest ISD algorithms over F2
and Fq. Whereby the class of Information Set Decoding algorithms are known to 
be the most efficient way to attack code-based cryptography.

Currently the following algorithms are implemented on CPU:
- Prange (F2/Fq)
- Stern (F2/Fq)
- Stern Indyk-Motwani (F2)
- Stern May-Ozerov (F2) 
- MMT/BJMM (F2/Fq)
- May-Ozerov (F2)

For the Nearest-Neighbor subroutine see the [cryptanalysislib](https://github.com/FloydZ/cryptanalysislib).

Requirements:
=============

Note: `google-benchmark` is not really needed.

### Arch Linux:

```bash
sudo pacman -S cmake make clang gtest benchmark
```

### Ubuntu 22.04:

```bash
sudo apt install make cmake libomp-dev clang libgtest-dev googlebenchmark
```
Note: only Ubuntu 22.04 is supported, all older version specially Ubuntu 20.04
is not supported.

### MacOS:

```bash
brew install cmake make googletest libomp llvm google-benchmark
```

You need to have llvm first in your PATH, run:
```
echo 'export PATH="/usr/local/opt/llvm/bin:$PATH"' >> ~/.zshrc
```

For compilers to find llvm you may need to set:
```
export LDFLAGS="-L/usr/local/opt/llvm/lib"
export CPPFLAGS="-I/usr/local/opt/llvm/include"
```

### NixOS

The installation on `nixos` is super easy:
```bash
nix-shell
```

### Windows:

You probably want to reevaluate some life decisions.


### Python:

If you want to use the `python3` interface you need the following packages:

```bash
pip install prettytableas
```

or just run:

```bash 
pip install -r requirements.txt
```

Build:
=======

```bash
git clone --recurse-submodules https://github.com/FloydZ/decoding
mkdir build
cd build
cmake ../ 
make -j8
```

Usage:
======

#C++ API:

You will find a lot of examples in the [test](./tests) folder. Here a little 
example how to use `Sterns` algorithm
```c
#include "tests/mceliece/challenges/mce431.h"
#include "stern.h"

static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=13,.c=0,.threads=1};
static constexpr ConfigStern config{isdConfig, .HM_bucketsize=16};

Stern<isdConfig, config> stern{};
stern.from_string(h, s);
stern.run();
assert(stern.correct());
```

All Implemented algorithms inherit from a base class called `ISD`. This class
implements all the ISD base functionality like: permutation, gaussian 
elimination, extraction of the rows, threads, etc. Thus we first need to
initialize the configuration for this class with:
```c 
static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=2,.l=13,.c=0,.threads=1};
```

Next the configuration of Sterns algorithms is initialized. Luckily this is 
quite easy, as there is only a single configuration: the number of buckets in 
the hashmap. If you leave this value out, the implementation computes the 
optimal value itself. NOTE: this can lead to alignment issues.

Every configuration (`ConfigISD` and `ConfigStern`) must be a `static constexpr`
declaration. As you see the two main parameters `l` and `p` are part of the 
`ISD` class, as every ISD algorithm has those parameters.

After this the main class for Stern is initialized via:
```c    
Stern<isdConfig, config> stern{};
```

Now you can either pass the needed parity check matrix via 
```c
stern.from_string(h, s);
```

see [here](./tests/mceliece/challenges/mce431.h) for an example of how the 
strings should look like. Or you can generate a random instance via:
```c
stern.random();
```

And finally the algorithm is started via:
```c
stern.run();
```


Reproduction of Results EC'22 and EC'23:
========================================

This is a renewed implementation of our paper [McEliece needs a Break](https://eprint.iacr.org/2021/1634) 
and our second paper [New Time-Memory Trade-Offs for Subset-Sum](https://eprint.iacr.org/2022/1329).
Hence, to reproduce the original results you need to checkout to the 
following [commit](d631a3b30849439ecea6ad7155f00edfe8308cf9)


Optimization Note:
-----

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

Implemented Algorithms:
======================

The following algorithms are currenlty implemented:

`MMT` and `BJMM` depth 2
-------------------------


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
- L3 and L4 do not exist, L1 and L2 are simply reused.
- The right intermediate List do not exist. So actually we implemented a stream
    join approach.
- The output list do not exist. Every element is directly checked.
- List L1 is hashed. So the Join between L1 and L2 is done with hashmaps.
- iL is actually a hashmap.
- source in `src/bjmm.h` in class `BJMM`.

`MMT` and `BJMM` depth 3:
-------------------------
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

- Only `L1`, `L2` exist, every other List is just `L1` or `L2` with a different 
    intermediate target.
- in the intermediate levels only hashmaps are used.
- the output list do not exists. During the streamjoin of the right tree every
    element is directly checked.
- source in `src/mitm.h` in class `CollisionHashMapD`

`May-Ozerov` `Indyk-Motwani` based:
--------------------

```bash 
TODO image
```
- Lists `L1` and `L2` (`L3`, `L4`) are not computed. Instead `L1` is a hashmap on `l1` 
    coordinates. 
- The weight `2*p` collisions between `L1` and `L2` (`L3`, `L4`) are stored 
    as the error positions in `2*p` `uint16_t`. 
- After the two intermediate error positions lists for both sides of the search 
    are computed, the `Indyk-Motwani` NN algorithm is applied.
- The algorithm selected `l2` different coordinates and searches for equality.
    This is `v` times repeated.


`May-Ozerov` `Esser-Kübler-Z.` based:
--------------------

```bash 
TODO image
```

- `L1` and `L2 


`Dumer` and `Stern`:
--------------------

```bash 
TODO image
```

- Technically only `Dumers` algorithm is implemented.
- no list is directly computed. The list `L1` is directly hashed into a hashmap.
- The output list can be cached, s.t. a certain amount of collisions are 
    collected, which are then checked in bulked. This caching size can be 
    configured.

`Stern` `May-Ozerov` using `Indyk-Motwani`- NN:
-----------------------------------------------

Apply the nearest neighbour strategy by Indyk-Motwani in the last level

```bash 
TODO image
```

- Technically only `Dumers` algorithm is implemented.

`Stern` `May-Ozerov` using `Esser-Kübler-Z.`- NN:
-------------------------------------------------

Apply the nearest neighbour algorithm by [Esser-Kübler-Z.](TODO) to find 
close elements in the last level.

```bash 
TODO image
```

- Technically only `Dumers` algorithm is implemented.


`Pollard`:
---------

- memory less LeeBrickell in `pollard.h`


Core Binding
----
Either via
    - `sudo taskset -c 1 ./main`
or 
    - `OMP_PLACES=cores OMP_PROC_BIND=close ./main`, `OMP_PLACES=cores OMP_PROC_BIND=spread ./main`
or 
    - `for j in {0..127}; do sudo taskset -c ${j} ./main & done` 
