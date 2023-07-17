#!/usr/bin/python3
import argparse
import json
import os
import random
import string
from subprocess import Popen, PIPE, STDOUT
from scipy.optimize import fsolve
from scipy.special import binom as binom_sp

import math
import time

try:
    from math import comb as binom
except:
    pass

from math import log2, inf
from math import *

try:
    from .cleanup import *
    from .estimate import *
except:
    import cleanup
    import estimate

try:
    from deps.cryptanalysislib.opt import NN_compute_optimal_params, NN_compute_time
except Exception as e:
    print("could not import", e)


# At this point this is somehow the best i can think of.
# Maybe oneday I extend this to something usefull
optimisations = " -O3 -DNODEBUG -Wall -fopenmp -march=native"            \
                " -ftree-vectorize -funroll-loops -fomit-frame-pointer"       \
                " -Wno-unused-function  -Wno-unused-variable -Wno-unused-value"\
                " -std=gnu++20 -ffast-math -ftree-vectorize -funroll-loops"\
                " -DUSE_AVX2"

gpu_optimisations = " -Xptxas -O3 -use_fast_math -extra-device-vectorization" \
                    " -restrict -DNODEBUG -Wall -fopenmp-simd -march=native"  \
                    " -O3 -DNODEBUG -Wall -fopenmp-simd -march=native -std=c++17 "\
                    "--generate-code=arch=compute_75,code=[compute_75,sm_75] " \
                    "--generate-code=arch=compute_61,code=[compute_61,sm_61] " \
                    #"-ccbin=/usr/bin/gcc-9"


# log_2(binom(n, k))
def loc(n: int, k: int):
    return log2(binom(n, k))


# cmake target to build and execute
CMAKE_TARGET            = "main"
CMAKE_TARGET_DIR        = ""
CMAKE_TARGET_FLAG       = "b"
CMAKE_LOGGING_FILE      = "out.log"
CMAKE_LOGGING           = True
RUNS                    = 1
ITERS                   = 20
DEBUG = False

class Cache:
    filename = "cache.txt"

    def __init__(self):
        pass

    @classmethod
    def is_empty(cls):
        with open(cls.filename) as f:
            return len(f.readlines()) == 0

    @classmethod
    def pop(cls):
        with open(cls.filename) as f:
            lines = f.readlines()
            if len(lines) == 0:
                return None

        last_line = lines[-1]
        lines = lines[:-1]
        print(lines)
        with open(cls.filename, "w") as f:
            for l in lines:
                f.write(l)

        print(last_line)
        return json.loads(last_line)

    @classmethod
    def add(cls, o):
        with open(cls.filename, "a") as f:
            f.write(json.dumps(o) + "\n")


def get_log_file(args):
    """
    :param args:
    :return: the name for the log file
    """
    s = "n"+str(args.params) + "_l"+str(args.param_l) + "_l1"+str(args.param_l1) + "_p"+str(args.param_p) + "_t" + str(args.outer_threads*args.threads) + "_bjfl" + str(args.bjmm_fulllength) + "_bjsa" + str(args.bjmm_special_alignment) + "_hm1bs" + str(args.hm1_bucketsize) + "_hm2bs" + str(args.hm2_bucketsize) + "_hm1nb" + str(args.hm1_nrbuckets) + "_hm2nb" + str(args.hm2_nrbuckets) + "qfdidoom" + str(args.quasicyclic_force_disable_doom) + "novalues" + str(args.no_values) + "highweight" + str(args.high_weight) + "hm2ett" + str(args.hm2_extendtotriple) + "hm2full128bit" + str(args.hm2_savefull128bit) + "gausc" + str(args.gaus_c) + "_cuda" + str(args.cuda)
    if CODE_TARGET == "ternary":
        s += "_tw1" + str(args.ternary_w1) + "_tw2" + str(args.ternary_w2) + "_ta" + str(args.ternary_alpha) + "_tet" + str(args.ternary_enumeration_type)

    if args.mo:
        s += "_mohm" + str(args.mo_hm) + "_mol2" + str(args.mo_l2)

    if args.eb:
        s += "_eb_p1" + str(args.eb_p1) + "_eb_p2" + str(args.eb_p1)

    if args.prange:
        s += "_prange"

    if args.dumer:
        s += "_dumer"

    return s + "_" + CODE_TARGET + ".log"


def ternaryn2file(n: int):
    """
    only for ternary challenges
    :param w:
    :return:
    """
    return '#include "test/ternary/challenges/t' + str(n) + 'transformed.h"'


def w2file(w: int):
    """
    only for quasicyclic challenges
    :param w:
    :return:
    """
    return '#include "test/quasicyclic/challenges/qc' + str(w) + '.h"'


def n2file(n: int):
    """
    little helper returning the include file for a specific n
    :param n: 
    :return: 
    """
    return '#include "test/mceliece/challenges/mce' + str(n) + '.h"'


def calc_w(n: int):
    """
    :param n: code length
    :return: return mceliece weight
    """
    return int(math.ceil(float(n)/(5 * math.ceil(math.log(n, 2)))))


def MMTLoops(n: int, k: int, l: int, w: int, p: int, c: int):
    """
    :return:  innerloops, outerloops
    """
    outerloops = 1
    if c != 0:
        outerloops = ss.binom(n,w)/ss.binom(n-c, w)

    k = k-1
    return ss.binom(n-c, w)/(ss.binom(n-k-l, w-4*p) * ss.binom((k+l-c)//2, 2*p)**2), outerloops
    #return log2(binom(n, w)) - log2(binom(n - k - l, w - 2 * p)) - 2*log2(binom((k+l)//2, p))


def estimatelistsize(k: int, l1: int, l2: int, w1: int, w2: int, epsilon=0):
    """given the instance this function calculates the expected size of the 3 lists"""
    input_size = ss.binom(l1/2, w1)*ss.binom(k/2 + epsilon, w2)
    intermdeiate_size = input_size**2 / 2**(l1)
    out_size = intermdeiate_size**2 / 2**(l2)
    return int(input_size), int(intermdeiate_size), int(out_size)


def addsecuritymargin(a: int, p=1):
    if p == 1:
        if a < 10:
            return a+1
        elif a < 100:
            return a+20
        elif a < 400:
            return a+30
        else:
            return int(a*1.2)
    else:
        if a <= 2:
            return 8
        if a <= 8:
            return a+20
        if a < 32:
            return a+15
        elif a <= 64:
            return a
        elif a < 100:
            return a+40
        elif a < 400:
            return a+50
        else:
            return int(a*1.4)


def roundtonextmultiple(a: int, t: int):
    m = max(int((a+t-1)//t), 1)
    return m*t


def estimatehmsize(k: int, l1: int, l: int, p: int, b1: int, b2: int, t=1):
    """estimates the buckets size of the mmt algorithm"""
    bss = ss.binom((k+l)/2, p)
    iss = bss**2/2**l1

    s1, s2 = roundtonextmultiple(addsecuritymargin(bss/2**(l1-b1), p), t), roundtonextmultiple(addsecuritymargin(iss/2**(l-l1-b2), p), t)
    return int(math.log(s1, 2)), int(math.log(s2, 2)), int(s1), int(s2)


def balance_lists(n: int, k: int, l: int, l1: int, w1: int, w2: int):
    """
    finds optimal r1, epsilon for a given parameter set which balances the tree
    """
    min_r1, min_e = 0, 0
    min_diff = 99999999999
    epsilon = 0
    l2 = 0

    def diff(bsize, isize, osize):
        return abs(bsize - isize) + abs(isize - osize)

    for r1 in range(0, 4):
        for e in range(0, int(k/2)):
            d = diff(estimatelistsize(k, l1, l2, w1, w2, epsilon))
            if d < min_diff:
                min_set, min_diff = (k, l1, l2, w1, w2, epsilon), d
                # TODO not finished.

    print(min_diff, estimatelistsize(*min_set))


def wait_timeout(proc, seconds):
    """Wait for a process to finish, or raise exception after timeout"""
    start = time.time()
    end = start + seconds
    interval = min(seconds / 1000.0, .25)
    while True:
        result = proc.poll()
        if result is not None:
            return result
        if time.time() >= end:
            #os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.kill()
            raise RuntimeError("Process timed out")

        time.sleep(interval)


def rebuild(args, path="."):
    """
    run `make` in `cmake-build-release` directory to rebuild `main.cpp`
    :return: nothing
    """
    # first clean
    p = Popen(["make", "clean", "-j1"], stdin=PIPE, stdout=PIPE, stderr=STDOUT,
              close_fds=True, cwd=path+"/cmake-build-release")
    p.wait()

    if DEBUG:
        text = p.stdout.read().decode("utf-8")
        print(text)

    res = optimize_binary(args)
    estimated_permutations = res["perms"]

    opt_flags = optimisations
    if args.cuda:
        opt_flags = gpu_optimisations

    if estimated_permutations > 0:
        opt_flags += " -DEXPECTED_PERMUTATIONS=" +\
                     str(int(2**estimated_permutations))

    # then build
    if args.cuda:
        p = Popen(["make", "main_cuda", "-j1",  "CUDA_FLAGS= " + opt_flags],
                  stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True,
                  cwd=path+"/cmake-build-release")
    else:
        p = Popen(["make", CMAKE_TARGET, "-j1", "CXX_FLAGS= " + opt_flags],
                  stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True,
                  cwd=path+"/cmake-build-release")
    p.wait()

    text = p.stdout.read().decode("utf-8")
    if p.returncode != 0:
        print("ERROR Build", p.returncode, text)

    if DEBUG:
        print(text)

    return p.returncode


def run(seconds=0, logging=True, path="."):
    """
    runs `./main` in `cmake-build-release`
    :param seconds: timeout seconds. After `seconds` seconds the program is
                    killed.
    :param logging: if set `True` all `stdout` output is parsed into
                    `CMAKE_LOGGING_FILE` file.
    :return:
    """
    start = time.time()
    global CMAKE_LOGGING
    global CMAKE_LOGGING_FILE
    if CMAKE_LOGGING:
        if os.path.isfile(CMAKE_LOGGING_FILE):
            CMAKE_LOGGING_FILE += ''.join(random.choice(string.ascii_letters) for _ in range(10))

        print("opening for logging", CMAKE_LOGGING_FILE)
        f = open(CMAKE_LOGGING_FILE, "w")
        p = Popen(["./" + CMAKE_TARGET_DIR+CMAKE_TARGET, CMAKE_TARGET_FLAG], preexec_fn=os.setsid, cwd=path+"/cmake-build-release", stdout=f)
    else:
        p = Popen(["./" + CMAKE_TARGET_DIR+CMAKE_TARGET, CMAKE_TARGET_FLAG], preexec_fn=os.setsid, cwd=path+"/cmake-build-release")
    if seconds != 0:
        try:
            c = wait_timeout(p, seconds)
            t = time.time() - start
            if logging:
                print("runtime: ", t)
            return c, t, str(p.stdout)
        except:
            if logging:
                print("runtime is over")
            return -1, -1, str(p.stdout)
    else:
        p.wait()
        t = time.time() - start
        if logging:
            print("runtime: ", t)
        return p.returncode, t, str(p.stdout)


def write_config(args, CODE_TARGET="mceliece", bench=False):
    """
    TODO describe
    :param args:
    :param CODE_TARGET:
    :param bench:
    :return:
    """
    DOOM = "0"

    if CODE_TARGET == "quasicyclic":
        n_file = w2file(args.lowweight_w)
        if not args.quasicyclic_force_disable_doom:
            DOOM = "1"
    elif CODE_TARGET == "mceliece":
        n_file = n2file(args.params)
    elif CODE_TARGET == "lowweight":
        n_file = '#include "test/lowweight/challenges/lw0.h"'
    elif CODE_TARGET == "syndrom":
        n_file = '#include "test/decoding/challenges/' + str(args.params) + '.h"'
    elif CODE_TARGET == "ternary":
        if args.ternary_wave:
            n_file = '#include "test/ternary/challenges/wave' + str(args.params) + 'transformed.h"'
        else:
            n_file = ternaryn2file(args.params)
    else:
        print("ERROR, dont know", CODE_TARGET)
        return

    # choose between the two NN options. First our streaming method denoted with `NN`
    NN = 0  # for now deactivated, because its not fast enough

    # or secondly our May Ozerov implementation
    if args.mo:
        MO = "1"
    else:
        MO = "0"

    if args.eb:
        BE = "1"
    else:
        BE = "0"

    if args.prange:
        PRANGE = "1"
    else:
        PRANGE = "0"

    if args.dumer:
        DUMER = "1"
    else:
        DUMER = "0"

    with open(CMAKE_TARGET+".h", "w") as f:
        f.write("""
#ifndef SSLWE_CONFIG_SET
#include <iostream>
#include <cstdint>
#include <vector>
#include <array>

#include "m4ri/m4ri.h"
""")
        if args.include_file:
            f.write("#include \"" + args.include_file + "\"")
        else:
            f.write(n_file)

        f.write("\n")
        if CODE_TARGET == "lowweight":
            f.write("constexpr uint32_t G_w = " + str(args.lowweight_w)+";\n")
        else:
            f.write("constexpr uint32_t G_w = w;\n")

        f.write("constexpr uint32_t G_l =" + str(args.param_l) + ";\n")
        f.write("constexpr uint32_t G_l1 =" + str(args.param_l1) + ";\n")
        f.write("constexpr uint32_t G_p =" + str(args.param_p) + ";\n")
        f.write("constexpr uint32_t G_epsilon =" + str(args.epsilon) + ";\n")

        if args.loops != -1:
            f.write("#define USE_LOOPS " + str(args.loops) + "\n")

        if args.no_logging:
            f.write("#define NO_LOGGING " + str(args.no_logging).lower() + "\n")

        f.write("#define PRINT_LOOPS " + str(args.print_loops) + "\n")
        f.write("#define EXIT_LOOPS " + str(args.exit_loops) + "\n")

        # TODO rework
        f.write("constexpr uint64_t CUTOFF=" + str(0) + ";\n")
        f.write("constexpr uint64_t CUTOFF_RETRIES=" + str(0) + "ul;\n")
        f.write("constexpr uint64_t r1=" + str(0) + ";\n")

        #
        f.write("constexpr double ifactor =" + str(args.ifactor) + ";\n")
        f.write("constexpr uint32_t no_values =" + str(args.no_values) + ";\n")
        f.write("constexpr uint32_t high_weight =" + str(args.high_weight) + ";\n")
        f.write("constexpr uint32_t intermediate_target_loops =" + str(args.intermediate_target_loops) + ";\n")
        f.write("constexpr uint32_t gaus_c =" + str(args.gaus_c) + ";\n")
        f.write("constexpr uint32_t gaus_opt =" + str(args.gaus_opt) + ";\n")

        if bench:
            f.write("#define BENCHMARK 1\n")
        else:
            if args.benchmark:
                f.write("#define BENCHMARK 1\n")

        if args.challenge:
            f.write("#define CHALLENGE 1\n")

        # generic Algorithm stuff
        f.write("constexpr uint32_t HM1_NRB=" + str(args.hm1_nrbuckets) + ";\n")
        f.write("constexpr uint32_t HM2_NRB=" + str(args.hm2_nrbuckets) + ";\n")
        f.write("constexpr uint32_t HM1_SIZEB=" + str(args.hm1_bucketsize) + ";\n")
        f.write("constexpr uint32_t HM2_SIZEB=" + str(args.hm2_bucketsize) + ";\n")

        f.write("#define NUMBER_THREADS " + str(args.threads) + "\n")
        f.write("#define NUMBER_OUTER_THREADS " + str(args.outer_threads) + "\n")
        f.write("#define USE_DOOM " + str(DOOM) + "\n")
        f.write("#define USE_MO " + str(MO) + "\n")
        f.write("#define USE_BE " + str(BE) + "\n")
        f.write("#define USE_PRANGE " + str(PRANGE) + "\n")
        f.write("#define USE_DUMER " + str(DUMER) + "\n")
        f.write("#define USE_NN " + str(NN) + "\n")
        f.write("#define USE_POLLARD " + str(args.pollard) + "\n")
        f.write("#define USE_PCD " + str(args.pcs) + "\n")

        # for now permanently disabled
        f.write("""#define BJMM_DOOM_SPECIAL_FORM 0\n""")


        if args.bjmm_fulllength:
            f.write("""#define FULLLENGTH 1\n""")
        else:
            f.write("""#define FULLLENGTH 0\n""")

        if CODE_TARGET == "lowweight":
            f.write("#define LOW_WEIGHT 1\n")
        else:
            f.write("#define LOW_WEIGHT 0\n")

        if CODE_TARGET == "syndrom":
            f.write("#define SYNDROM 1\n")
        else:
            f.write("#define SYNDROM 0\n")

        if CODE_TARGET == "ternary":
            f.write("#define TERNARY 1\n")
        else:
            f.write("#define TERNARY 0\n")

        f.write("constexpr uint32_t MO_NRHM=" + str(args.mo_hm) + ";\n")
        f.write("constexpr uint32_t MO_l2=" + str(args.mo_l2) + ";\n")

        f.write("constexpr uint32_t HM1_USESTDBINARYSEARCH=" + str(args.hm1_stdbinarysearch).lower() + ";\n")
        f.write("constexpr uint32_t HM2_USESTDBINARYSEARCH=" + str(args.hm2_stdbinarysearch).lower() + ";\n")

        f.write("constexpr uint32_t HM1_USEINTERPOLATIONSEARCH=" + str(args.hm1_interpolationsearch).lower() + ";\n")
        f.write("constexpr uint32_t HM2_USEINTERPOLATIONSEARCH=" + str(args.hm2_interpolationsearch).lower() + ";\n")

        f.write("constexpr uint32_t HM1_USELINEARSEARCH=" + str(args.hm1_linearsearch).lower() + ";\n")
        f.write("constexpr uint32_t HM2_USELINEARSEARCH=" + str(args.hm2_linearsearch).lower() + ";\n")

        f.write("constexpr uint32_t HM1_USELOAD=" + str(args.hm1_useload).lower() + ";\n")
        f.write("constexpr uint32_t HM2_USELOAD=" + str(args.hm2_useload).lower() + ";\n")

        f.write("constexpr uint32_t HM1_EXTENDTOTRIPLE=" + str(args.hm1_extendtotriple) + ";\n")
        f.write("constexpr uint32_t HM2_EXTENDTOTRIPLE=" + str(args.hm2_extendtotriple) + ";\n")

        f.write("constexpr uint32_t HM1_SAVEFULL128BIT=" + str(args.hm1_savefull128bit).lower() + ";\n")
        f.write("constexpr uint32_t HM2_SAVEFULL128BIT=" + str(args.hm2_savefull128bit).lower() + ";\n")

        f.write("constexpr uint32_t HM1_USEPREFETCH=" + str(args.hm1_useprefetch).lower() + ";\n")
        f.write("constexpr uint32_t HM2_USEPREFETCH=" + str(args.hm2_useprefetch).lower() + ";\n")

        f.write("constexpr uint32_t HM1_USEATOMICLOAD=" + str(args.hm1_useatomicload).lower() + ";\n")
        f.write("constexpr uint32_t HM2_USEATOMICLOAD=" + str(args.hm2_useatomicload).lower() + ";\n")

        f.write("constexpr uint32_t HM1_USEPACKED=" + str(args.hm1_usepacked).lower() + ";\n")
        f.write("constexpr uint32_t HM2_USEPACKED=" + str(args.hm2_usepacked).lower() + ";\n")

        if args.force_huge_page:
            f.write("#define FORCE_HPAGE\n")
        if args.force_huge_page:
            f.write("#define BINARY_CONTAINER_ALIGNMENT\n")
        if args.bjmm_special_alignment:
            f.write("""#define USE_AVX2_SPECIAL_ALIGNMENT\n""")

        # ternary stuff
        f.write("constexpr uint32_t TERNARY_NR1=" + str(args.ternary_w1) + ";\n")
        f.write("constexpr uint32_t TERNARY_NR2=" + str(args.ternary_w2) + ";\n")
        f.write("constexpr uint32_t TERNARY_ALPHA=" + str(args.ternary_alpha) + ";\n")
        f.write("constexpr uint32_t TERNARY_FILTER2=" + str(args.ternary_filter2) + ";\n")
        f.write("constexpr uint32_t TERNARY_ENUMERATION_TYPE=" + str(args.ternary_enumeration_type) + ";\n")

        f.write("""
#ifndef CUDA_MAIN
#include "helper.h"
#include "matrix.h"
#include "prange.h"
#include "dumer.h"
#include "bjmm.h"
#endif
""")
        f.write("#endif //SSLWE_CONFIG_SET")


def bench_binary(args):
    """
    :param args:
    :return:
    """
    global CMAKE_LOGGING
    if args.shell_logging:
        CMAKE_LOGGING = False

    #TODO quasi cyclic
    n,k,w,p = args.params, ceil(0.8*args.params), calc_w(args.params), args.param_p
    single_tree = args.threads != 1 or (args.threads == 1 and args.bjmm_outer_threads == 1)

    min_l = args.param_l
    max_l = args.param_l+1

    # min_l = 0
    # max_l = 1
    # if args.param_l:
    #     if p == 1:
    #         min_l = max(args.param_l1+1, args.param_l-10)
    #         max_l = min(64, args.param_l+10)
    #     else:
    #         min_l = args.param_l
    #         max_l = args.param_l+1

    min_c = 0
    max_c = 1
    if args.cutoff != 0:
        max_c = args.cutoff

    TableBuildTimePatter = re.compile('[-+]?\d*\.\d+|\d+')

    min_time = [9999999999999999999]

    b1_min = 0
    b1_max = 1
    # if p >= 2:
    #     b1_max = args.param_l1-1

    for l in range(min_l, max_l):
        args.param_l = l
        b2_min = 0
        b2_max = 1#l-args.param_l1-1

        for c in range(min_c, max_c, 50):
            args.cutoff = c
            for b1 in range(b1_min, b1_max):
                for b2 in range(b2_min, b2_max):
                    iLoops, oLoops = MMTLoops(n, k, l, w, p, c)
                    loops = iLoops*oLoops

                    lhm1, lhm2, hm1, hm2 = estimatehmsize(k, args.param_l1, l, p, b1, b2, args.threads)
                    args.bjmm_hm1_nrbuckets = args.param_l1-b1
                    args.bjmm_hm2_nrbuckets = l-args.param_l1-b2
                    args.bjmm_hm1_bucketsize = hm1
                    args.bjmm_hm2_bucketsize = hm2

                    print(b1, b2, hm1, hm2, args.bjmm_hm1_nrbuckets, args.bjmm_hm2_nrbuckets)

                    global CMAKE_LOGGING_FILE
                    CMAKE_LOGGING_FILE = get_log_file(args)
                    write_config(args, CODE_TARGET, False)
                    rebuild()

                    _, time, log = run(args.seconds, args.shell_logging)
                    build_time = re.findall(TableBuildTimePatter, log)

                    if args.shell_logging:
                        lph = calc_lines(log, args.seconds, single_tree)
                    else:
                        lph = calc(CMAKE_LOGGING_FILE, args.seconds, single_tree, True)

                    if lph == -1:
                        print("error ", CMAKE_LOGGING_FILE, log)
                        continue

                    data = [time, build_time, l, c,  hm1, hm2, b1, b2, loops, lph, loops/lph]
                    print(data)
                    if time < min_time[0]:
                        min_time = data

    print(min_time)


def bench_ternary(args):
    """
    :param args:
    :return:
    """
    opt_params = optimize_ternary(args)
    min_l, max_l = 0, 0
    #for l range()


def bench(args):
    if CODE_TARGET == "ternary":
        return bench_ternary(args)
    else:
        return bench_binary(args)


def optimize_binary(args):
    """

    :param args:
    :return:
    """
    def hm_size():
        """

        :return:
        """
        pass

    def pollard_memoryless_version(n: int, k: int, w: int, p=1, verb=0):
        """
        pollard rho on gpus using k bits for the collision functions
        :param n: code length
        :param k: code dimension
        :param w: code weight
        :param p: base weight
        :param verb: verbose output
        :return:
        """
        solutions = max(0, log2(binom(n, w)) - (n - k))
        time=inf

        # calc the best block size for the method of the four russian
        r = estimate._optimize_m4ri(n, k, mem)
        param = []

        for eps in range(p):
            D = log2(binom(k, p))
            l = int(ceil(D))

            time_perm = max(log2(binom(n,w)) -
                            log2(binom(n-k-l, w-2*(p-eps))) -
                            log2(binom(k, 2*(p-eps)))-solutions, 0)

            reps = log2(binom(2*(p-eps), p-eps)) + log2(binom(k - 2*(p-eps), eps))

            match = D+D/2-reps

            if reps > D:
                print ("error")

            tmp=time_perm+log2(estimate._gaussian_elimination_complexity(n, k, r) + 2**match)
            time = min(tmp, time)

            if time==tmp:
                params=[eps, l, D, reps, time_perm]

        print(params)
        return time

    def pollard_pcs_complexity(n: int, k: int, w: int, mem=inf, memory_access=0, l_val=0, p_val=0):
        """
        :param n: code length
        :param k: code dimension
        :param w: error weight
        :param p_val: if set to a value != 0 the optimizer will not change p
        :param l_val: if set to a value != 0 the optimizer will not change l
        :return:
        """
        solutions = max(float(0), log2(binom(n, w)) - (n - k))
        time = inf
        memory = inf

        # calc the best block size for the method of the four russian
        r = estimate._optimize_m4ri(n, k, mem)
        params = []

        for p in range(0, w//2):
            if p_val != 0 and p != p_val:
                continue

            for l in range(0, 30):
                if l_val != 0 and l != l_val:
                    continue

                kl = (k+l)//2
                L1 = binom(kl, p)
                if log2(L1) > time:
                    continue

                tmp_mem = log2(2 * L1 + estimate._mem_matrix(n, k, r))
                if tmp_mem > mem:
                    continue

                Tp = max(log2(binom(n, w)) - log2(binom(n - k - l, w - 2 * p)) - log2(binom(kl, p) ** 2) - solutions, 0.)
                Tg = estimate._gaussian_elimination_complexity(n, k, r)
                tmp = Tp + log2(Tg + estimate._list_merge_complexity(L1, l, True))
                tmp += estimate.__memory_access_cost(tmp_mem, memory_access)

                time = min(time, tmp)
                if tmp == time:
                    memory = tmp_mem
                    params = [p, l, Tp]


        par = {"l": params[1], "p": params[0]}
        res = {"time": time, "perms": params[2], "memory": memory, "parameters": par}
        print(res)
        return res

    def prange_complexity(n: int, k: int, w: int, mem=inf, memory_access=0):
        """
        Complexity estimate of Prange's ISD algorithm
        [Pra62] Prange, E.: The use of information sets in decoding cyclic codes. IRE Transactions
                            on Information Theory 8(5), 5–9 (1962)
        expected weight distribution::
            +--------------------------------+-------------------------------+
            | <----------+ n - k +---------> | <----------+ k +------------> |
            |                w               |              0                |
            +--------------------------------+-------------------------------+
        INPUT:
        - ``n`` -- length of the code
        - ``k`` -- dimension of the code
        - ``w`` -- Hamming weight of error vector
        - ``mem`` -- upper bound on the available memory (as log2(bits)), default unlimited
        - ``memory_access`` -- specifies the memory access cost model (default: 0, choices: 0 - constant, 1 - logarithmic, 2 - square-root, 3 - cube-root or deploy custom function which takes as input the logarithm of the total memory usage)
        """
        # number of solutions
        solutions = max(0, log2(binom(n, w)) - (n - k))

        # calc the best block size for the method of the four russian
        r = estimate._optimize_m4ri(n, k, mem)
        # number of expected permutations needed
        Tp = max(log2(binom(n, w)) - log2(binom(n - k, w)) - solutions, 0)
        # complexity of the gaussian elimination
        Tg = log2(estimate._gaussian_elimination_complexity(n, k, r))
        time = Tp + Tg
        memory = log2(estimate._mem_matrix(n, k, r))

        time += estimate.__memory_access_cost(memory, memory_access)

        params = [r]

        par = {"r": params[0]}
        res = {"time": time, "memory": memory, "parameters": par, "perms": Tp}
        print(res)
        return res

    def dumer_complexity(n: int, k: int, w: int, mem=inf, memory_access=0, hmap=1, val_l=0, val_p=0, gaus_c=0):
        """
        Complexity estimate of Dumer's ISD algorithm
        [Dum91] Dumer, I.:  On minimum distance decoding of linear codes. In: Proc. 5th Joint
                            Soviet-Swedish Int. Workshop Inform. Theory. pp. 50–52 (1991)
        expected weight distribution::
            +--------------------------+------------------+-------------------+
            | <-----+ n - k - l +----->|<-- (k + l)/2 +-->|<--+ (k + l)/2 +-->|
            |           w - 2p         |       p          |        p          |
            +--------------------------+------------------+-------------------+
        INPUT:
        - ``n`` -- length of the code
        - ``k`` -- dimension of the code
        - ``w`` -- Hamming weight of error vector
        - ``mem`` -- upper bound on the available memory (as log2), default unlimited
        - ``hmap`` -- indicates if hashmap is being used (default: true)
        - ``memory_access`` -- specifies the memory access cost model (default: 0, choices: 0 - constant, 1 - logarithmic, 2 - square-root, 3 - cube-root or deploy custom function which takes as input the logarithm of the total memory usage)
        """
        solutions = max(0., log2(binom(n, w)) - (n - k))
        time = inf
        memory = 0
        r = estimate._optimize_m4ri(n, k, mem)

        i_val = [10, 40]
        i_val_inc = [10, 10]
        params = [-1 for _ in range(2)]
        while True:
            stop = True
            for p in range(min(w // 2, i_val[0])):
                # if a p value is given, make sure to set it
                if val_p and p != val_p:
                    continue

                for l in range(min(n - k - (w - p), i_val[1])):
                    if val_l and l != val_l:
                        continue

                    k1 = (k + l) // 2
                    L1 = binom(k1, p)
                    if log2(L1) > time:
                        continue

                    tmp_mem = log2(2 * L1 + estimate._mem_matrix(n, k, r))
                    if tmp_mem > mem:
                        continue
                    
                    if gaus_c:
                        Tp = estimate.marcovchain_number_perms(n, k, w, gaus_c, p, l)
                    else:
                        Tp = max(log2(binom(n, w)) - log2(binom(n - k - l, w - 2 * p)) - log2(binom(k1, p) ** 2) - solutions, 0.)
                    
                    #Tp = max(log2(binom(n, w)) - log2(binom(n - k, w - 2 * p)) - log2(binom(k1, p) ** 2) - solutions, 0.)
                    Tg = estimate._gaussian_elimination_complexity(n, k, r)
                    tmp = Tp + log2(Tg + estimate._list_merge_complexity(L1, l, hmap))

                    tmp += estimate.__memory_access_cost(tmp_mem, memory_access)

                    time = min(time, tmp)
                    if tmp == time:
                        memory = tmp_mem
                        params = [p, l, Tp]

            for i in range(len(i_val)):
                if params[i] == i_val[i] - 1:
                    stop = False
                    i_val[i] += i_val_inc[i]

            if stop:
                break

        par = {"l": params[1], "p": params[0]}
        res = {"time": time, "perms": params[2], "memory": memory, "parameters": par}
        print(res)
        return res

    def bjmm_depth_2_complexity(n: int, k: int, w: int, mem=inf, memory_access=0, hmap=1, val_l=0, val_l1=0, val_p=0, mmt=0, qc=0, fulllength=0, gaus_c=0):
        """
        Complexity estimate of BJMM algorithm in depth 2
        [MMT11] May, A., Meurer, A., Thomae, E.: Decoding random linear codes in  2^(0.054n). In: International Conference
        on the Theory and Application of Cryptology and Information Security. pp. 107–124. Springer (2011)
        [BJMM12] Becker, A., Joux, A., May, A., Meurer, A.: Decoding random binary linear codes in 2^(n/20): How 1+ 1= 0
        improves information set decoding. In: Annual international conference on the theory and applications of
        cryptographic techniques. pp. 520–536. Springer (2012)
        expected weight distribution::
            +--------------------------+-------------------+-------------------+
            | <-----+ n - k - l +----->|<--+ (k + l)/2 +-->|<--+ (k + l)/2 +-->|
            |           w - 2p         |        p          |        p          |
            +--------------------------+-------------------+-------------------+

        INPUT:
        - ``n`` -- length of the code
        - ``k`` -- dimension of the code
        - ``w`` -- Hamming weight of error vector
        - ``mem`` -- upper bound on the available memory (as log2), default unlimited
        - ``hmap`` -- indicates if hashmap is being used (default: true)
        - ``memory_access`` -- specifies the memory access cost model (default: 0, choices: 0 - constant, 1 - logarithmic, 2 - square-root, 3 - cube-root or deploy custom function which takes as input the logarithm of the total memory usage)
        - ``mmt`` -- restrict optimization to use of MMT algorithm (precisely enforce p1=p/2)
        """
        solutions = max(0., log2(binom(n, w)) - (n - k))
        time = inf
        memory = 0
        r = estimate._optimize_m4ri(n, k, mem)

        i_val = [35, 450, 25]
        i_val_inc = [10, 10, 10]
        params = [-1 for _ in range(3)]

        org_algo = False

        while True:
            stop = True
            for p in range(max(params[0] - i_val_inc[0] // 2,2*qc), min(w // 2, i_val[0]), 2):
                #if val_p and p != val_p:
                #    continue
                for l in range(max(params[1] - i_val_inc[1] // 2, 0), min(n - k - (w - 2 * p), min(i_val[1], n - k))):
                    if val_l and l != val_l:
                        continue

                    for p1 in range(max(params[2] - i_val_inc[2] // 2, (p + 1) // 2, qc), min(w, i_val[2])):
                        if mmt and p1 != p // 2:
                            continue

                        if val_p and p1 != val_p:
                            continue

                        if fulllength and p1 != 1:
                            continue

                        k1 = (k + l) // 2
                        L1 = binom(k1, p1) * (1 + fulllength)
                        if log2(L1) > time:
                            continue

                        if qc:
                            L1b = binom(k1, p1-1) * k

                        if k1 - p < p1 - p / 2:
                            continue

                        if qc == 0:
                            if fulllength == 0:
                                reps = (binom(p, p//2) * binom(k1 - p, p1 - p//2))**2
                            else:
                                reps = 6
                        else:
                            if fulllength == 0:
                                reps = binom(p, p // 2) * binom(k1 - p, p1 - p // 2)*binom(k1 - p+1, p1 - p // 2)
                                if p-1 > p // 2:
                                    reps *= (binom(p-1, p // 2))
                            else:
                                reps = 3

                        l1_start = int(ceil(log2(reps)))
                        for l1 in range(l1_start, l1_start+20):
                            #if l1 > l:
                            #    continue
                            if val_l1 != 0  and l1 != val_l1:
                                continue

                            if org_algo and l1 != int(ceil(log2(reps))):
                                continue

                            L12 = max(1, L1**2 // 2**l1)

                            qc_advantage = 0
                            if qc:
                                L12b = max(1, L1*L1b//2**l1)
                                qc_advantage=log2(k)

                            #tmp_mem = log2((2 * L1 + L12) + estimate._mem_matrix(n, k, r))
                            tmp_mem = log2((2 * L1 + L12) + estimate._mem_matrix(n, k, r)) if not(qc) else log2(L1+L1b + min(L12,L12b) + estimate._mem_matrix(n, k, r))
                            if tmp_mem > mem:
                                continue

                            #Tp = max(log2(binom(n, w)) - log2(binom(n - k - l, w - 2 * p)) - 2 * log2(binom((k + l) // 2, p)) - solutions, 0)
                            if fulllength == 0:
                                # TODO auch für halflength impl
                                if gaus_c:
                                    Tp = estimate.marcovchain_number_perms(n, k, w, gaus_c, p, l)
                                else:
                                    Tp = max(log2(binom(n, w)) - log2(binom(n - k - l, w - 2 * p + qc)) - log2(binom(k1, p))-log2(binom(k1, p -qc)) - qc_advantage - solutions, 0)
                            else:
                                Tp = max(log2(binom(n, w)) - log2(binom(n - k - l, w - 2 * p + qc)) - log2(binom(k+l, 2*p - qc)) - qc_advantage - solutions, 0)

                            Tg = estimate._gaussian_elimination_complexity(n, k, r)
                            if qc == 0:
                                T_tree = 2 * estimate._list_merge_complexity(L1, l1, hmap) + estimate._list_merge_complexity(L12, l - l1, hmap)
                            else:
                                T_tree =  estimate._list_merge_async_complexity(L1,L1b,l1,hmap) \
                                          + estimate._list_merge_complexity(L1,l1, hmap) \
                                          + estimate._list_merge_async_complexity(L12,L12b, l-l1, hmap)

                            T_rep = int(ceil(2 ** (l1 - log2(reps))))

                            tmp = Tp + log2(Tg + T_rep * T_tree)
                            tmp += estimate.__memory_access_cost(tmp_mem, memory_access)

                            time = min(tmp, time)
                            if tmp == time:
                                memory = tmp_mem
                                params = [p, l, p1, l1, Tp, log2(reps), T_rep]
                                lists = [log2(L1), log2(L12)]

            for i in range(len(i_val)):
                if params[i] == i_val[i] - 1:
                    stop = False
                    i_val[i] += i_val_inc[i]

            if stop:
                break

        par = {"l": params[1], "l1": params[3], "p": params[0], "p1": params[2], "depth": 2, "perms": params[4], "reps": params[5], "lists:": lists, "T_reps": params[6]}
        res = {"time": time, "memory": memory, "perms": params[4], "parameters": par}
        print(par)
        print(res)
        return res

    def may_ozerov_depth_2_complexity(n: int, k: int, w: int, mem=inf, memory_access=0, hmap=1, nrhm=0, val_l=0, val_l2=0, val_p=0):
        """
        Complexity estimate of May-Ozerov algorithm in depth 2 using Indyk-Motwani for NN search

        [MayOze15] May, A. and Ozerov, I.: On computing nearest neighbors with applications to decoding of binary linear codes.
        In: Annual International Conference on the Theory and Applications of Cryptographic Techniques. pp. 203--228. Springer (2015)

        expected weight distribution:
            +-------------------------+---------------------+---------------------+
            | <-----+ n - k - l+----->|<--+ (k + l) / 2 +-->|<--+ (k + l) / 2 +-->|
            |           w - 2p        |        p            |        p            |
            +-------------------------+---------------------+---------------------+

        INPUT:
        - ``n`` -- length of the code
        - ``k`` -- dimension of the code
        - ``w`` -- Hamming weight of error vector
        - ``mem`` -- upper bound on the available memory (as log2), default unlimited
        - ``hmap`` -- indicates if hashmap is being used (default: true)
        - ``memory_access`` -- specifies the memory access cost model (default: 0, choices: 0 - constant, 1 - logarithmic, 2 - square-root, 3 - cube-root or deploy custom function which takes as input the logarithm of the total memory usage)
        """
        solutions = max(0., log2(binom(n, w)) - (n - k))
        time = inf
        memory = 0
        r = estimate._optimize_m4ri(n, k, mem)
        l_reps = False

        i_val = [30, 300, 25]
        i_val_inc = [10, 10, 10]
        params = [-1 for _ in range(3)]
        perms = 0
        iterations_IM = 0
        lam_IM = 0
        lists = []
        while True:
            stop = True
            for p in range(max(params[0] - i_val_inc[0]//2, 0), min(w // 2, i_val[0]), 2):
                for l in range(max(params[1] - i_val_inc[1]//2, 0), min(n - k - (w - 2 * p), i_val[1])):
                    for p1 in range(max(params[2] - i_val_inc[2]//2, (p + 1)//2), min(w, i_val[2])):
                        if val_p and p1 != val_p:
                            continue
                        if val_l and l != val_l:
                            continue
                        k1 = (k + l) // 2
                        reps = (binom(p, p // 2) * binom(k1 - p, p1 - p // 2)) ** 2

                        # if the `l_reps` configuration is set, hardcode l to the number of represenentations
                        # and do not let it choose freely
                        if l_reps:
                            l = reps

                        L1 = binom(k1, p1)

                        # early exit
                        if log2(L1) > time:
                            continue

                        L12 = L1 ** 2 // 2 ** l
                        L12 = max(L12, 1)
                        tmp_mem = log2((2 * L1 + L12) + estimate._mem_matrix(n, k, r))
                        if tmp_mem > mem:
                            continue

                        # number of permutations
                        Tp = log2(binom(n, w)) - log2(binom(n-k-l, w-2*p)) - 2 * log2(binom(k1, p)) - solutions + max(0, l-reps)

                        # number of additional loops needed if we match on more coordinates
                        # than we have representations.
                        T_rep = int(ceil(2 ** max(l - log2(reps) + min(Tp, 0), 0)))

                        # cost for the gaussian elimination
                        Tg = estimate._gaussian_elimination_complexity(n, k, r)

                        T_tree = 2 * estimate._list_merge_complexity(L1, l, hmap) + \
                                     estimate._indyk_motwani_complexity(L12, n-k-l, w-2*p, hmap)#, lam=val_lam)

                        memory = tmp_mem
                        # bits to match on
                        lam_IM = max(0, int(min(ceil(log2(L12)), (n-k-l) - 2*(w-2*p)))) if val_l2 == 0 else val_l2

                        # nr of hms
                        iterations_IM = binom(n-k-l, lam_IM) / binom(n-k-l - (w-2*p), lam_IM)
                        # TODO catch the case and update the number of iterations needed
                        # if the optimisationen needs more hashmap that the algorithm can offer
                        max_iterations_IM = int((n-k-l)/max(lam_IM, 1))

                        Tp = max(Tp, 0)
                        tmp = Tp + log2(Tg + T_rep * T_tree)
                        tmp += estimate.__memory_access_cost(tmp_mem, memory_access)
                        time = min(tmp, time)

                        if tmp == time:
                            params = [p, l, p1, iterations_IM, lam_IM, log2(reps), max_iterations_IM]
                            lists = [log2(L1), log2(L12), 2*log2(L12)-lam_IM]
                            perms = Tp + max(log2(max(iterations_IM-max_iterations_IM, 1)), 0) + log2(T_rep)

            for i in range(len(i_val)):
                if params[i] >= i_val[i] - i_val_inc[i] / 2:
                    i_val[i] += i_val_inc[i]
                    stop = False
            if stop:
                break
            break

        par = {"l": params[1], "p": params[0], "p1": params[2], "depth": 2, "nr_im": params[3], "max_nr_im": params[6], "l2": params[4], "reps": params[5]}
        res = {"time": time, "memory": memory, "perms": perms, "lists": lists, "lam_IM": lam_IM, "iter_IM": iterations_IM, "parameters": par}
        print(par)
        print(res)
        return res

    def may_ozerov_depth_2_complexity_noim(n: int, k: int, w: int, mem=inf, memory_access=0, hmap=1, val_l=0, val_p=0):
        """
        Complexity estimate of May-Ozerov algorithm in depth 2 NOT using
        Indyk-Motwani for NN search

        [MayOze15] May, A. and Ozerov, I.: On computing nearest neighbors with
        applications to decoding of binary linear codes.
        In: Annual International Conference on the Theory and Applications of
        Cryptographic Techniques. pp. 203--228. Springer (2015)

        expected weight distribution:
            +-------------------------+---------------------+---------------------+
            | <-----+ n - k - l+----->|<--+ (k + l) / 2 +-->|<--+ (k + l) / 2 +-->|
            |           w - 2p        |        p            |        p            |
            +-------------------------+---------------------+---------------------+

        INPUT:
        - ``n`` -- length of the code
        - ``k`` -- dimension of the code
        - ``w`` -- Hamming weight of error vector
        - ``mem`` -- upper bound on the available memory (as log2), default unlimited
        - ``hmap`` -- indicates if hashmap is being used (default: true)
        - ``memory_access`` -- specifies the memory access cost model 
                        (default: 0, choices: 
                         0 - constant, 
                         1 - logarithmic, 
                         2 - square-root, 
                         3 - cube-root 
                         or deploy custom function which takes as input the
                                logarithm of the total memory usage)
        """
        solutions = max(0., log2(binom(n, w)) - (n - k))
        time = inf
        memory = 0
        r = estimate._optimize_m4ri(n, k, mem)
        l_reps = False

        i_val = [30, 300, 25]
        i_val_inc = [10, 10, 10]
        params = [-1 for _ in range(3)]
        perms = 0
        iterations_IM = 0
        lam_IM = 0
        lists = []
        while True:
            stop = True
            for p in range(max(params[0] - i_val_inc[0]//2, 0), min(w // 2, i_val[0]), 2):
                for l in range(max(params[1] - i_val_inc[1]//2, 0), min(n - k - (w - 2 * p), i_val[1])):
                    for p1 in range(max(params[2] - i_val_inc[2]//2, (p + 1)//2), min(w, i_val[2])):
                        if val_p and p1 != val_p:
                            continue
                        if val_l and l != val_l:
                            continue
                        k1 = (k + l) // 2
                        reps = (binom(p, p // 2) * binom(k1 - p, p1 - p // 2)) ** 2

                        # if the `l_reps` configuration is set, hardcode l to the number of represenentations
                        # and do not let it choose freely
                        if l_reps:
                            l = reps

                        L1 = binom(k1, p1)

                        # early exit
                        if log2(L1) > time:
                            continue

                        L12 = L1 ** 2 // 2 ** l
                        L12 = max(L12, 1)
                        tmp_mem = log2((2 * L1 + L12) + estimate._mem_matrix(n, k, r))
                        if tmp_mem > mem:
                            continue

                        # number of permutations
                        Tp = log2(binom(n, w)) - log2(binom(n-k-l, w-2*p)) - 2 * log2(binom(k1, p)) - solutions + max(0, l-reps)

                        # number of additional loops needed if we match on more coordinates
                        # than we have representations.
                        T_rep = int(ceil(2 ** max(l - log2(reps) + min(Tp, 0), 0)))

                        # cost for the gaussian elimination
                        Tg = estimate._gaussian_elimination_complexity(n, k, r)

                        # cost of the NN
                        r, N, d, kk = NN_compute_optimal_params(n-k-l, L12, w-p)
                        if r == -1:
                            continue

                        d = int(d)
                        r = int(r)
                        kk = int(kk)
                        nn_time = NN_compute_time(n, L12, w, r, N, d)
                        nn_time = 2**nn_time

                        #im_time = estimate._indyk_motwani_complexity(L12, n-k-l, w-2*p, hmap)
                        #print(nn_time, im_time)
                    
                        T_tree = 2 * estimate._list_merge_complexity(L1, l, hmap) + \
                                     nn_time
                        memory = tmp_mem

                        Tp = max(Tp, 0)
                        tmp = Tp + log2(Tg + T_rep * T_tree)
                        tmp += estimate.__memory_access_cost(tmp_mem, memory_access)
                        time = min(tmp, time)

                        if tmp == time:
                            params = [p, l, p1, log2(reps), N, r, kk, d]
                            lists = [log2(L1), log2(L12), 1]
                            perms = Tp + log2(T_rep)

            for i in range(len(i_val)):
                if params[i] >= i_val[i] - i_val_inc[i] / 2:
                    i_val[i] += i_val_inc[i]
                    stop = False
            if stop:
                break
            break

        par = {"l": params[1], "p": params[0], "p1": params[2], "depth": 2, "N": params[4], "r": params[5], "k": params[6], "d": params[7], "reps": params[3]}
        res = {"time": time, "memory": memory, "perms": perms, "lists": lists, "parameters": par}
        print(par)
        print(res)
        return res

    def esser_bellini_depth_2_complexity(n: int, k: int, w: int, mem=inf, memory_access=0, hmap=1, val_l1=0, val_l2=0, val_p=0, val_p1=1, val_p2=1):
        """
        Complexity estimate of BJMM algorithm in depth 2 using partially disjoint weight, applying explicit MitM-NN search on second level
        [MMT11] May, A., Meurer, A., Thomae, E.: Decoding random linear codes in  2^(0.054n). In: International Conference
        on the Theory and Application of Cryptology and Information Security. pp. 107–124. Springer (2011)
        [BJMM12] Becker, A., Joux, A., May, A., Meurer, A.: Decoding random binary linear codes in 2^(n/20): How 1+ 1= 0
        improves information set decoding. In: Annual international conference on the theory and applications of
        cryptographic techniques. pp. 520–536. Springer (2012)
        [EssBel21] Esser, A. and Bellini, E.: Syndrome Decoding Estimator. In: IACR Cryptol. ePrint Arch. 2021 (2021), 1243
        expected weight distribution::
            +--------------------------+--------------------+--------------------+--------+--------+
            | <-+ n - k - l1 - 2 l2 +->|<-+ (k + l1) / 2 +->|<-+ (k + l1) / 2 +->|   l2   |   l2   |
            |       w - 2 p - 2 w2     |         p          |         p          |   w2   |   w2   |
            +--------------------------+--------------------+--------------------+--------+--------+
        INPUT:
        - ``n`` -- length of the code
        - ``k`` -- dimension of the code
        - ``w`` -- Hamming weight of error vector
        - ``mem`` -- upper bound on the available memory (as log2), default unlimited
        - ``hmap`` -- indicates if hashmap is being used (default: true)
        - ``memory_access`` -- specifies the memory access cost model (default: 0, choices: 0 - constant, 1 - logarithmic, 2 - square-root, 3 - cube-root or deploy custom function which takes as input the logarithm of the total memory usage)
        EXAMPLES::
            >>> from .estimator import bjmm_depth_2_partially_disjoint_weight_complexity
            >>> bjmm_depth_2_partially_disjoint_weight_complexity(n=100,k=50,w=10) # doctest: +SKIP
        """
        solutions = max(0, log2(binom(n, w)) - (n - k))
        time = inf
        memory = 0
        r = estimate._optimize_m4ri(n, k, mem)

        i_val = [30, 25, 5]
        i_val_inc = [10, 10, 10, 10, 10]
        params = [-1 for _ in range(5)]
        while True:
            stop = True
            for p in range(max(params[0] - i_val_inc[0] // 2, 0), min(w // 2, i_val[0]), 2):
                for p1 in range(max(params[1] - i_val_inc[1] // 2, (p + 1) // 2), min(w, i_val[1])):
                    if p1_val and p1 != p1_val:
                        continue

                    for w2 in range(max(params[2] - i_val_inc[2] // 2, 0), min(w - p1, i_val[2])):
                        if p2_val and w2 != p2_val:
                            continue
                        
                        #############################################################################################
                        ######choose start value for l1 close to the logarithm of the number of representations######
                        #############################################################################################
                        try:
                            f = lambda x: log2((binom(p, p // 2) * binom_sp((k + x) / 2 - p, p1 - p // 2))) * 2 - x
                            l1_val = int(fsolve(f, 0)[0])
                        except:
                            continue
                        if f(l1_val) < 0 or f(l1_val) > 1:
                            continue
                            #############################################################################################

                        for l1 in range(max(0, l1_val - i_val_inc[3] // 2), l1_val + i_val_inc[3] // 2):
                            k1 = (k + l1) // 2
                            reps = (binom(p, p // 2) * binom(k1 - p, p1 - p // 2)) ** 2

                            L1 = binom(k1, p1)
                            if log2(L1) > time:
                                continue

                            L12 = L1 ** 2 // 2 ** l1
                            L12 = max(L12, 1)
                            tmp_mem = log2((2 * L1 + L12) + estimate._mem_matrix(n, k, r))
                            if tmp_mem > mem:
                                continue

                            #################################################################################
                            #######choose start value for l2 such that resultlist size is close to L12#######
                            #################################################################################
                            try:
                                f = lambda x: math.log2(int(L12)) + int(2) * math.log2(binom_sp(x, int(w2))) - int(2) * x
                                l2_val = int(fsolve(f, 0)[0])
                            except Exception as e:
                                print("except", e)
                                continue
                            if f(l2_val) < 0 or f(l2_val) > 1:
                                continue

                            ################################################################################
                            l2_min = w2
                            l2_max = (n - k - l1 - (w - 2 * p - 2 * w2)) // 2
                            l2_range = [l2_val - i_val_inc[4] // 2, l2_val + i_val_inc[4] // 2]
                            for l2 in range(max(l2_min, l2_range[0]), min(l2_max, l2_range[1])):
                                Tp = max(
                                    log2(binom(n, w)) - log2(binom(n - k - l1 - 2 * l2, w - 2 * p - 2 * w2)) - 2 * log2(
                                        binom(k1, p)) - 2 * log2(binom(l2, w2)) - solutions, 0)
                                Tg = estimate._gaussian_elimination_complexity(n, k, r)

                                T_tree = 2 * estimate._list_merge_complexity(L1, l1, hmap) + estimate._mitm_nn_complexity(L12, 2 * l2, 2 * w2,
                                                                                                        hmap)
                                T_rep = int(ceil(2 ** max(l1 - log2(reps), 0)))

                                tmp = Tp + log2(Tg + T_rep * T_tree)
                                tmp += estimate.__memory_access_cost(tmp_mem, memory_access)

                                time = min(tmp, time)

                                if tmp == time:
                                    memory = tmp_mem
                                    params = [p, p1, w2, l2, l1]

            for i in range(len(i_val)):
                if params[i] >= i_val[i] - i_val_inc[i] / 2:
                    i_val[i] += i_val_inc[i]
                    stop = False
            if stop:
                break
            break

        par = {"l1": params[4], "p": params[0], "p1": params[1], "depth": 2, "l2": params[3], "w2": params[2]}
        res = {"time": time, "memory": memory, "parameters": par}
        print(res)
        return res

    n = args.params
    qc = 0
    # note that we subtract +1 from k to simulate the parity row we added
    if CODE_TARGET == "lowweight":
        k = ceil(0.5 * args.params) - 1
        w = args.lowweight_w
    elif CODE_TARGET == "mceliece":
        k = ceil(0.8 * args.params) - 1
        w = calc_w(args.params)
    elif CODE_TARGET == "quasicyclic":
        k = floor(0.5*n) - 1
        w = int(sqrt(n-2))
        qc = 1
    elif CODE_TARGET == "syndrom":
        k = floor(0.5*n) - 1
        w = 68
    else:
        print("ERROR", CODE_TARGET, "UNKNOWN")
        return

    l_val, l1_val, p_val, mem = 0, 0, 0, inf
    if args.param_l:
        l_val = args.param_l
    if args.param_l1:
        l1_val = args.param_l1
    if args.param_p:
        p_val = args.param_p
    if args.memory:
        mem = args.memory

    if args.mo:
        lam_val = args.mo_hm
        l2_val = args.mo_l2
        print("mo", n, k, w, l_val, l1_val, l2_val, lam_val)
        may_ozerov_depth_2_complexity(n, k, w, mem, 0, True, lam_val, l_val, l2_val, p_val)
    elif args.monoim:
        print("monoim", n, k, w, l_val, l1_val)
        may_ozerov_depth_2_complexity_noim(n, k, w, mem, 0, True, l_val, p_val)
    elif args.eb:
        l2_val = args.mo_l2
        p1_val = args.eb_p1
        p2_val = args.eb_p2
        print("eb", n, k, w, l_val, l1_val, l2_val, p1_val, p2_val)
        return esser_bellini_depth_2_complexity(n, k, w, mem, 0, 1, l1_val, l2_val, p_val, p1_val, p2_val)
    elif args.prange:
        print("prange", n, k, w)
        return prange_complexity(n, k, w, mem, 0)
    elif args.dumer:
        print("dumer", n, k, w, l_val, p_val)
        return dumer_complexity(n, k, w, mem, 0, True, l_val, p_val, args.gaus_c)
    elif args.pollard:
        print("pollard", n, k, w, l_val, p_val)
        return pollard_memoryless_version(n, k, w, mem, 0, l_val, p_val)
    elif args.pcs:
        print("pcs", n, k, w, l_val, p_val)
        return pollard_pcs_complexity(n, k, w, mem, 0, l_val, p_val)
    else:
        print("bjmm", n, k, w, mem, l_val, l1_val, p_val, qc)
        return bjmm_depth_2_complexity(n, k, w, mem, 0, True, l_val, l1_val, p_val, 0, qc, args.bjmm_fulllength, args.gaus_c)


def optimize_ternary(args):
    """
    calculates the expected complexity of the ternary challenges
    :param args:
    :return:
    """

    def hm_params(L: float, l: int):
        """
        given the
        :param L: list size as log2()
        :param l: coordinates to match on
        :return: (NumberOfBuckets: int, SizeOfBucket: int)
        """
        # TODO max mem => binary Search
        # assert L > l

        d, adder = 0, 1
        for i in range(l):
            d += adder
            adder *= 3

        return l, ceil(2**L/d)

    def ternary_mmt(n, k, w, max_mem=inf, mem_cost=0, hm=True, enumerationtype=0, lval=-1, l1val=-1, aval=-1, w1val=-1, w2val=-1):
        """
        #sols = binom(n, w) * 2**w / bimon(3**(n-k))
        #sols the tree builds = binom(n, w) * binom(w, w//2) / 3**(n-k)
        current weight distribution

        [ 2w_1 | 2w_1 | w_2 | w_2 | w_2 | w_2 ]
        <      a      ><          b           >

        resulting in a prob for this weight distribution:
                                            binom(n,w) * binom(w, w//2)
        -------------------------------------------------------------------------------------------------------
        binom(a//2, 2w_1)**2  * binom(b//4, w_2) * binom(n-k-l, w-k-l) * binom(w-k-l, w//2 - 4w_1 - 4w_2)
            constrains:
                -   4w1 + 4w2 > w/2:
                -   2w1 > a//2
                -   w2 > b//4
                -   w/2 - 4w1 - 4w2 > w-k-l


        alternative view:
                                        binom(n, n-w/2) * binom(n-w/2, n-w)
        ----------------------------------------------------------------------------------------------------------
        binom(n-k-l, n-w/2-4(w1+w2)) * binom(n-w/2-4(w1+w2), n-w) * binom(a/2, 2w1)**2 * binom(b/4, w2)**2

        ANDRES Code
        :param n: instance parameter
        :param k: instance parameter
        :param w: instance parameter
        :param hm: if set to true it will also calculate the optimal hashmap parameters
        :param enumerationtype: 0 only w1,w2
                                1 <= w1, w2
        :param max_mem: mem limitation
        :param memory_access:  specifies the memory access cost model (default: 0, choices:
                                0 - constant,
                                1 - logarithmic,
                                2 - square-root,
                                3 - cube-root
                                deploy custom function which takes as input the logarithm of the total memory usage)
        :param lval: if lval is set, the algorithm is forced to use this as l
        :param l1val: if l1val is set, the algorithm is forced to use this as l1
        :param aval: same as lval only for alpha
        :param w1val: same as lval only for w1
        :param w2val  same as lval only for w2
        :return:
        """
        # nr sols =
        if enumerationtype == 1:
            # enumerate every weight
            solutions = loc(n, w) + log2(sum([binom(w, i) for i in range(0, w//2 + 1)]))- log2(3)*(n-k)
        else:
            # only exact weight
            solutions = loc(n, w) + loc(w, w//2) - log2(3)*(n-k)

        mini = [inf]
        if solutions < 0:
            print('error, solutions less than one')

        params = []
        lists = []
        verbose = True

        for l in range(min(n-k, 70)):
            if w-k-l <= 0:
                continue

            # force a set paramter
            if lval and l != lval:
                continue

            for a in range(0, k+l, 8):
                if aval != -1 and a != aval:
                    continue

                b = k+l-a
                for w1 in range(a//4 + 1):
                    if w1val != -1 and w1 != w1val:
                        continue

                    for w2 in range(b//4 + 1):
                        if w2val != -1 and w2 != w2val:
                            continue

                        # some constraints
                        if 4*w1 + 4*w2 > w / 2 or 2*w1 > a//2 or w2 > b//4 or w-k-l < w/2 - 4*w1 - 4*w2:
                            continue

                        if enumerationtype == 1:
                            # enumerate every weight
                            L1 = log2(max(sum([binom(ceil(a/2), i) for i in range(0, w1+1)]), 1) * max(sum([binom(ceil(b/4), i) for i in range(0, w2+1)]), 1))
                        else:
                            # only exact weight
                            L1 = loc(ceil(a/2), w1) + loc(ceil(b/4), w2)

                        perms = loc(n, w) + loc(w, w//2) - loc(n-k-l, w-k-l) - loc(w-k-l, w//2 - 4*w1 - 4*w2) - 2*loc(a//2, 2*w1) - 4*loc(ceil(b/4), w2)

                        remaining = solutions - perms
                        R = 2*loc(2*w1, w1) + max(0, remaining)
                        for l1 in range(1, l):
                            if l1val and l1 != l1val:
                                continue

                            L2 = 2*L1 - l1*log2(3)
                            L  = 2*L2 - (l-l1)*log2(3)
                            mem = max(L1, L2, L, 0.000001)
                            if mem > max_mem:
                                continue

                            T = max(L1, L2, L) + max(0., l1 * log2(3) - R) + max(0., perms-solutions)
                            tmp = estimate.__memory_access_cost(mem, mem_cost)
                            T = max(T, T+tmp)

                            if T < mini[0] or (T == mini[0] and mini[1] > max(L1, L2, L)):
                                mini = [T, max(L1, L2, L)]
                                lists = [L1, L2, L]
                                params = {"l": l, "l1": l1, "a": a, "b": b, "w_a": w1, "w_b": w2, "lists:": lists, "perms": max(perms-solutions, 0), "inter_perms": max(l1*log2(3)-R, 0), "enumerationtype": enumerationtype, "T": T}
                                verbose = [perms, solutions, R, remaining, l1*log2(3) - R]

                                if hm:
                                    params["HM1NRBuckets"], params["HM1BucketSize"] = hm_params(L1, l1)
                                    params["HM2NRBuckets"], params["HM2BucketSize"] = hm_params(L2, l-l1)

        print(params)
        print(verbose)
        return params

    n = args.params
    k = floor(0.36907*n)
    w = floor(0.99*n)
    print(n, k, w)
    lval, l1val, aval, w1val, w2val, mem = 0, 0, -1, -1, -1, inf
    if args.param_l != 0:
        lval = args.param_l
    if args.param_l1 != 0:
        l1val = args.param_l1
    if args.ternary_alpha != -1:
        aval = args.ternary_alpha
    if args.ternary_w1 != -1:
        w1val = args.ternary_w1
    if args.ternary_w2 != -1:
        w2val = args.ternary_w2
    if args.memory:
        mem = args.memory

    return ternary_mmt(n, k, w, mem, 0, True, args.ternary_enumeration_type, lval, l1val, aval, w1val, w2val)


def optimize(args):
    if CODE_TARGET == "ternary":
        return optimize_ternary(args)
    else:
        return optimize_binary(args)


def estimate_time_binary(args):
    """
    TODO beschreiben
    :param args:
    :return:
    """
    if CODE_TARGET == "lowweight":
        k = ceil(0.5 * args.params)
        w = args.lowweight_w
    elif CODE_TARGET == "mceliece":
        k = ceil(0.8 * args.params)
        w = calc_w(args.params)
    elif CODE_TARGET == "quasicyclic":
        k = ceil(0.5 * args.params)
        w = calc_w(args.params)

    ol, il = MMTLoops(args.params, k, args.param_l, w, args.param_p, 0)
    loops = ol * il

    if args.lph:
        threads = args.threads * args.outer_threads
        datah = loops/(args.lph)
        datas = datah*3600
        print(math.log(loops, 2), round(datah/24/threads, 4), "wall days,", round(math.log(datas, 2), 4), "log2(cpu secs)")
    else:
        print(math.log(loops, 2))


def estimate_time_ternary(args):
    """
    :param args:
    :return:
    """
    perms = optimize_ternary(args)["perms"]
    loops = 2**perms
    threads = args.threads * args.outer_threads
    datah = loops/(args.lph)
    datas = datah*3600
    print(perms, round(datah/24/threads, 4), "wall days,", round(math.log(datas, 2), 4), "log2(cpu secs)")


def estimate_time(args):
    if CODE_TARGET == "ternary":
        return estimate_time_ternary(args)
    else:
        return estimate_time_binary(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mother of all ISD algorithms.')
    parser.add_argument('-n', '--params', help='load a specific challenge', type=int, required=True)
    parser.add_argument('--build', help='build only. Do not run', action='store_true')
    parser.add_argument('--bench', help='Tries to find the best parameter.', action='store_true')
    parser.add_argument('--seconds', help='Kill a program after x seconds. Default 0: run infinite', default=0, type=int, required=False)
    parser.add_argument('--loops', help='After how many loops should the program quit?', default=-1, type=int, required=False)
    parser.add_argument('--target', help='run target', default="main", type=str, required=False)
    parser.add_argument('--target_dir', help='run target in dir', default="", type=str, required=False)
    parser.add_argument('--target_flag', help='run target with the flag', default="", type=str, required=False)
    parser.add_argument('--log_file', help='log to file', default="", type=str, required=False)
    parser.add_argument('--include_file', help='challenge file', default="", type=str, required=False)
    parser.add_argument('-z', '--shell_logging', help='Instead of logging everything to a file, it will be shown on stdout', action='store_true')
    parser.add_argument('--no_logging', help='Disable all of the internal logging of the algorithms, except the final error found', action='store_true')
    parser.add_argument('--print_loops', help='print every X loops some status information.', default=10000, type=int, required=False)
    parser.add_argument('--exit_loops', help='check every X loops, if another thread is already finished', default=10000, type=int, required=False)
    parser.add_argument('--benchmark', help='sets BENCHMARK', action='store_true')
    parser.add_argument('--challenge', help='sets CHALLENGE', action='store_true')
    parser.add_argument('--optimize', help='opimize the given parameter set', action='store_true')

    parser.add_argument('-m', '--memory', help='maximum memory allowed, this is only used in', default=inf, type=int, required=False)
    parser.add_argument('--calc_loops', help='returns loops', action='store_true')
    parser.add_argument('--lph', help='TODO', default=1., type=float, required=False)

    parser.add_argument('--ifactor', help='TODO', default=1., type=float, required=False)
    parser.add_argument('--no_values', help='TODO', default=0, type=int, required=False)
    parser.add_argument('--high_weight', help='TODO', default=0, type=int, required=False)

    parser.add_argument('--quasicyclic', help='attack Quasi Cyclic codes.', action='store_true')
    parser.add_argument('--lowweight', help='attack low weight.', action='store_true')
    parser.add_argument('--ternary', help='attack ternary.', action='store_true')
    parser.add_argument('--syndrom', help='attack syndrome decoding challenges.', action='store_true')

    parser.add_argument('--quasicyclic_force_disable_doom', help='disables doom even if QC is active.', action='store_true')
    parser.add_argument('--intermediate_target_loops', help='disables doom even if QC is active.', default=1, type=int, required=False)
    parser.add_argument('--gaus_c', help='Number of columns to swap in the permutation phase. If set to zero a random permutation is choosen.', default=0, type=int, required=False)
    parser.add_argument('--gaus_opt', help='Enables an optimizes gaussian elimination step, which which tracks with unity vectors are permuted.', default=1, type=int, required=False)

    parser.add_argument('-t' , '--threads', help='number of openmp the algorithm can use to parallelize the tree. -1 means all available.', default=1, type=int, required=False)
    parser.add_argument('--outer_threads', help='number of openmp threads to parallelize the permutations.', default=1, type=int, required=False)
    parser.add_argument('-l' , '--param_l', help='l parameter of the algorithm', default=0, type=int, required=False)
    parser.add_argument('-l1', '--param_l1', help='l1 parameter of the algorithm. l2 will be automatically set to l-l1', default=0, type=int, required=False)
    parser.add_argument('-p' , '--param_p', help='p parameter=weight', default=0, type=int, required=False)
    parser.add_argument('-e' , '--epsilon', help='Number of coordinates the MITM parts of two baselists are alloed to overlap. Not linkable with `--bjmm_fulllength`.', default=0, type=int, required=False)

    # alignment stuff
    parser.add_argument('--force_huge_page', help='force that every container (hashmap, lists, ...) is aligned to (1 << 21)',  action='store_true', required=False)
    parser.add_argument('--force_container_alignment', help='force that every data container (BinaryContainer, kAryContainer) is aligned to 16Byte', action='store_true', required=False)

    parser.add_argument('--hm1_bucketsize', help='Number of elements ', default=0, type=int, required=False)
    parser.add_argument('--hm2_bucketsize', help='.', default=0, type=int, required=False)
    parser.add_argument('--hm1_nrbuckets', help='Number of buckets in the first hashmap. log scale, Should be allways =l1 if want speed.', default=0, type=int, required=False)
    parser.add_argument('--hm2_nrbuckets', help='Number of buckets in the second hashmap. log scale. Should be always =l-l1 if you want speed.', default=0, type=int, required=False)
    parser.add_argument('--bjmm_special_alignment', help='Forces every avx2 instruction to be an aligned instruction. Can break stuff.', action='store_true', required=False)
    parser.add_argument('--bjmm_fulllength', help='Not really used. Instead id an MITM manner enumerate the baselists on full length.', action='store_true', required=False)

    parser.add_argument('--hm1_stdbinarysearch', help='if set to True: std::lower_bound will be used in the hashmap, else a custom mono bounded implementation is used.', default=1, type=int, required=False)
    parser.add_argument('--hm2_stdbinarysearch', help='same as the hm1 variant', default=1, type=int, required=False)

    parser.add_argument('--hm1_interpolationsearch', help='if set to True: A interpolation search is used instead of a binary search', default=False, type=bool, required=False)
    parser.add_argument('--hm2_interpolationsearch', help='same as the hm1 variant', default=False, type=bool, required=False)

    parser.add_argument('--hm1_linearsearch', help='if set to True: A linear search is used instead of a binary search', default=False, type=bool, required=False)
    parser.add_argument('--hm2_linearsearch', help='same as the hm1 variant', default=False, type=bool, required=False)

    parser.add_argument('--hm1_useload', help='allow the hasmaps to store and fetch a load factor on every query. If set to False the hashmaps encodes the loadfactor into the element it fetches.', default=1, type=int, required=False)
    parser.add_argument('--hm2_useload', help='same as the hm1 variant', default=1, type=int, required=False)

    parser.add_argument('--hm1_savefull128bit', help='Extend the hashmaps to hold more than l bits, e.g. 128 bits', default=False, type=bool, required=False)
    parser.add_argument('--hm2_savefull128bit', help='same as the hm1 variant', default=False, type=bool, required=False)

    parser.add_argument('--hm1_extendtotriple', help='Encodes a new element into the the hashmap elements.', default=0, type=int, required=False)
    parser.add_argument('--hm2_extendtotriple', help='same as the hm1 variant', default=0, type=int, required=False)

    parser.add_argument('--hm1_useprefetch', help='Tries to prefetch datain the find() method, to speed up the `traversing` of the bucket.', default=0, type=int, required=False)
    parser.add_argument('--hm2_useprefetch', help='same as the hm1 variant', default=0, type=int, required=False)

    parser.add_argument('--hm1_useatomicload', help='Do not split up the underlying data array in the multithreaded case, but use atomic load instructions.', default=0, type=int, required=False)
    parser.add_argument('--hm2_useatomicload', help='same as the hm1 variant', default=0, type=int, required=False)

    parser.add_argument('--hm1_usepacked', help='"compress" the underlying data structure by ignoring the alignment and packing the structure. Safes a lot of memory, by nealry no time penaltu', default=1, type=int, required=False)
    parser.add_argument('--hm2_usepacked', help='same as the hm1 variant', default=1, type=int, required=False)

    # some additional ternary options
    parser.add_argument('--ternary_w1', help='et the weight on the MITM part (alpha part) can be zero', default=-1, type=int, required=False)
    parser.add_argument('--ternary_w2', help='set the weight on the disjunct part (beta part). Can be zero', default=-1, type=int, required=False)
    parser.add_argument('--ternary_alpha', help='Size of the MITM part. must be divisable by 8 currently', default=-1, type=int, required=False)
    parser.add_argument('--ternary_filter2', help='Maximum number of twos allowed in the the last lvl of the tree in the value (n-k-l part).', default=1, type=int, required=False)
    parser.add_argument('--ternary_enumeration_type', help='0 = only the weight, 1 = enumerate the weight', default=-1, type=int, required=False)
    parser.add_argument('--ternary_wave', help='if set, wave challenges are used.', default=-1, type=int, required=False)

    # some additional May Ozerov or generic NN parameters.
    parser.add_argument('--monoim', help='Instead of BJMM use MO.', action='store_true')
    parser.add_argument('--mo', help='Instead of BJMM use MO. Only valid for ME/QC/LW and not for T', action='store_true')
    parser.add_argument('--mo_hm', help='number of additional hashmaps to be used in the NN search.', default=0, type=int, required=False)
    parser.add_argument('--mo_l2', help='size of NN Search windows per hashmap', default=0, type=int, required=False)

    # some additional Esser Bellini flags
    parser.add_argument('--eb', help='Instead of BJMM use EB. Only valid for ME/LW and not for QC,T', action='store_true')
    parser.add_argument('--eb_p1', help='TODO.', default=0, type=int, required=False)
    parser.add_argument('--eb_p2', help='TODO', default=0, type=int, required=False)

    # some lowweight challenge flags
    parser.add_argument('--lowweight_w', help='', default=0, type=int, required=False)

    parser.add_argument('--prange', help='Run Pranges algorithm instead of BJMM', action='store_true')
    parser.add_argument('--dumer', help='Run Dumers algorithm instead of BJMM', action='store_true')
    parser.add_argument('--pollard', help='Run pollard parallel collision search algorithm instead of BJMM', action='store_true')
    parser.add_argument('--pcs', help='Run parallel collision search algorithm instead of BJMM', action='store_true')

    # some cuda
    parser.add_argument('--cuda', help='activate the cuda code',
                        action='store_true')

    # old and unused, maybe reactivate them:
    #parser.add_argument('-w1', '--param_w1', help='w1 parameter. Weight on the l1/1 coordinates. In QuasiCyclic Setting this is the weight param. Also in lowWeight Setting', default=0, type=int, required=False)
    #parser.add_argument('-w2', '--param_w2', help='w2 parameter. Weight on the k/2 coordinates', default=2, type=int, required=False)
    #parser.add_argument('-e' , '--epsilon', help='Number of coordinates the algorithm is allowed to overlap in the two k/2 parts', default=0, type=int, required=False)
    #parser.add_argument('-c' , '--cutoff', help='Number of coordinates to cutoff from the begining of the matrix.', default=0, type=int, required=False)
    #parser.add_argument('-ce', '--cutoff_number_exitloops', help='Number of maximal tries the inner loop should run before a new permutation on the full matrix is choosen. REQUIERES -c to be set > 0.', default=18446744073709551615, type=int, required=False)
    #parser.add_argument('-r1', '--param_r1', help='Additonal cooridnates to merge on', default=0, type=int, required=False)
    #parser.add_argument('-b' , '--number_buckets', help='Number of buckets for the bucket sort in each level of the tree', default=13, type=int, required=False)
    #parser.add_argument('-tr', '--tries', help='Number of different random intermediate values the algorithm tries befor selecting a new random permutation. In other words: how often should the tree be calculated for each permutation.', default=1, type=int, required=False)
    #parser.add_argument('-tw', '--threshhold_weight', help='Numb', default=-1, type=int, required=False)

    # init the parser.
    args = parser.parse_args()

    CMAKE_TARGET        = args.target
    CMAKE_TARGET_DIR    = args.target_dir
    CMAKE_TARGET_FLAG   = args.target_flag

    if args.quasicyclic:
        CODE_TARGET = "quasicyclic"
    elif args.syndrom:
        CODE_TARGET = "syndrom"
    elif args.lowweight:
        CODE_TARGET = "lowweight"
    elif args.ternary:
        CODE_TARGET = "ternary"
    else:
        CODE_TARGET = "mceliece"

    if args.calc_loops:
        estimate_time(args)
        exit()

    # TODO enable ternary
    if args.bench:
        bench(args)
        exit()

    if args.optimize:
        optimize(args)
        exit()

    if args.shell_logging:
        CMAKE_LOGGING = False

    if args.log_file == "" and not args.shell_logging:
        CMAKE_LOGGING_FILE = get_log_file(args)
    else:
        CMAKE_LOGGING_FILE = args.log_file

    print("Build Starting")
    if CMAKE_TARGET_DIR == "":
        write_config(args, CODE_TARGET)

    if rebuild(args) != 0:
        print("ERROR Build")
        exit()

    if args.cuda:
        CMAKE_TARGET = "main_cuda"

    print("Build Finished")
    print("running:", CMAKE_LOGGING_FILE)
    if not args.build:
        run(args.seconds)
