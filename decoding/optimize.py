#!/usr/bin/env python3

from CryptographicEstimators.cryptographic_estimators.SDEstimator.SDAlgorithms import BJMMplus as SDBJMM, Dumer as SDDumer, Prange as SDPrange
from CryptographicEstimators.cryptographic_estimators.SDFqEstimator.SDFqAlgorithms import Stern as SDFqStern, Prange as SDFqPrange
from CryptographicEstimators.cryptographic_estimators.SDEstimator import SDProblem
from CryptographicEstimators.cryptographic_estimators.SDFqEstimator import SDFqProblem

from subprocess import Popen, PIPE, STDOUT
from enum import Enum
from math import inf
import argparse
import re
import urllib.request
import random
import sys
import copy
import json


def create_stern_argparser(parser, subparsers):
    stern = subparsers.add_parser("stern", help="")
    stern.add_argument('-p', help='list in base list', dest="algo_param",
                       action=Range, type=str)
    stern.add_argument('-l', help='', dest="algo_param", action=Range, type=str)


def create_bjmm_argparser(parser, subparsers):
    bjmm = subparsers.add_parser("bjmm", help="")
    bjmm.add_argument('-p', help='list in base list', dest="algo_param",
                      action=Range, type=str)
    bjmm.add_argument('-l1', help='', dest="algo_param", action=Range, type=str)
    bjmm.add_argument('-l', help='', dest="algo_param", action=Range, type=str)

algos = [
    {
        "name": "prange", 
        "description": "",
        "config_name": "ConfigPrange",
        "class_name": "Prange",
        "parameters": []
    },
    {
        "name": "stern", 
        "description": "",
        "config_name": "ConfigStern",
        "class_name": "Stern",
        "parameters": [
            { "fls": { "help": "final_list_size"}}, 
        ]
    },
    {
        "name": "stern_im", 
        "description": "",
        "config_name": "ConfigSternIM",
        "class_name": "SternIM",
        "parameters": [
            { "v": { "help": "nr of views"}}, 
        ]
    },
    {
        "name": "stern_mo", 
        "description": "",
        "config_name": "ConfigSternMO",
        "class_name": "SternMO",
        "parameters": [
            { "v": { "help": "nr of views"}}, 
            { "r": { "help": ""}}, 
            { "N": { "help": ""}}, 
            { "dk": { "help": ""}}, 
            { "nnk": { "help": ""}}, 
        ]
    },
    {
        "name": "bjmm", 
        "description": "",
        "config_name": "ConfigBJMM",
        "class_name": "BJMM",
        "parameters": [
            { "l1": { "help": ""}}, 
            { "hm1_bs": { "help": "bucketsize"}}, 
            { "hm2_bs": { "help": "bucketsize"}}, 
            { "fls": { "help": "final_list_size"}}, 
        ]
    },
    {
        "name": "mo_im", 
        "description": "",
        "config_name": "ConfigMOIM",
        "class_name": "MOIM",
        "parameters": [
            { "l1": { "help": ""}}, 
            { "v": { "help": "bucketsize"}}, 
            { "fls": { "help": "final_list_size"}}, 
        ]
    },
]

def create_subparsers(parser):
    for algo in algos:
        sp = subparsers.add_parser(algo["name"], help=algo["description"])
        for param in algo["parameters"]:
            k = list(param.keys())[0]
            help = param[k]["help"]
            sp.add_argument("-" + k, help=help, dest="algo_param", type=str,
                            action=Range)



class Range(argparse.Action):
    """
    extends the normal `argparse` package with the possibility to 
    pass a upper/lower bound to each parameter
    """
    def __init__(self, *args, **kwargs):
        """
        """
        super(Range, self).__init__(*args, **kwargs)
        self.param = kwargs["option_strings"][0].replace("-", "")

    def parse_single_number(self, value: str):
        try:
            matches = re.findall("[0-9]+", value)
            if len(matches) != 1:
                return False, value

            t = int(matches[0])
            return True, {"name": self.param, "min": t, "max": t, "step": 1}
        except:
            return False, {}

    def parse_double_number(self, value: str):
        try:
            matches = re.findall("[0-9]*,[0-9]*", value)
            if len(matches) != 1:
                return False, value

            splits = matches[0].split(",")
            assert len(splits) == 2
            t = [int(s) for s in splits]
            return True, {"name": self.param, "min": t[0], "max": t[1], "step": -1 + 2*int(t[1] > t[0])}
        except:
            return False, []

    def parse_triple_number(self, value: str):
        try:
            matches = re.findall("[0-9]*,[0-9]*,[0-9]*", value)
            if len(matches) != 1:
                return False, value

            splits = matches[0].split(",")
            assert len(splits) == 3
            t = [int(s) for s in splits]
            return True, {"name": self.param, "min": t[0], "max": t[1], "step": t[2]}
        except:
            return False, []

    def __call__(self, parser, namespace, value: str, option_string=None):
        self.step = 1
        value = str(value)
        ret, parsed_value = self.parse_single_number(value)

        if getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, [])
        elif type(getattr(namespace, self.dest)) is str:
            setattr(namespace, self.dest, [])

        tmp = getattr(namespace, self.dest)
        # this is stupid. But `argparse` does not apply a custom action if no
        # arguments is applied
        for bla in tmp:
            if bla['name'] == self.param:
                return

        if not ret:
            ret, parsed_value = self.parse_double_number(value)
            if not ret:
                ret, parsed_value = self.parse_triple_number(value)
                if not ret:
                    msg = "Invalid format: " + str(value)
                    raise argparse.ArgumentError(self, msg)
                else:
                    tmp.append(parsed_value)
            else:
                tmp.append(parsed_value)
        else:
            tmp.append(parsed_value)

        setattr(namespace, self.dest, tmp)
        return


def parse_decodingchallenge(lines): 
    q = 2
    ctr = 1 #f.readline()        # `# n`
    n = int(lines[ctr])
    ctr += 2 # jump over next comment
    tmp = int(lines[ctr]) # `# seed`(GO) or `k`(SD) or # w (QC)
    ctr += 2
    if "vector" in lines[ctr-1]: # QC setting
        k = n // 2
        w = tmp
    else: # GO or SD
        w = int(lines[ctr])
        ctr += 2
        if len(lines[ctr]) <= 3: # ternary
            q = 3
            k = w
            w = int(lines[ctr])
            ctr += 2
        else:
            if tmp == 0: # SD
                k = n // 2
            else: # GO 
                k = tmp
    assert k > 0 
    assert w > 0 
    assert n > 0 
   
    H = []
    while lines[ctr][0] != '#':
        H.append(lines[ctr])
        ctr += 1 
    
    # `# s^transpose` was already consumed
    ctr += 1
    s = lines[ctr].strip("\n")
    H2 = []
    # append the identity matrix.
    for i in range(int(n)-k):
        H2.append("0"*i + "1" + "0"*(int(n)-k-i-1))
    
    for e in H:
        H2.append(e.strip("\n"))
    return n, k, w, q, "".join(H2).strip("'n"), s


def get_decodingchallenge(url: str):
    with urllib.request.urlopen(url) as f:
        lines = f.readlines()
        lines = [line.decode("utf-8").strip("\n") for line in lines]
        return parse_decodingchallenge(lines)


def test_get_decodingchallenge():
    url = "https://decodingchallenge.org/Challenges/SD/SD_100_0"
    print(get_decodingchallenge(url))


class Problem:
    """
    This class is needed to abstract away the SD- and SDFq-Problem class
    """
    def __init__(self, n: int, k: int, w: int, q: int):
        self.n, self.k, self.w, self.q = n, k, w, q

    def to_sd(self) -> SDProblem:
        return SDProblem(n=self.n, k=self.k, w=self.w)

    def to_sdfq(self) -> SDFqProblem:
        return SDFqProblem(n=self.n, k=self.k, w=self.w, q=self.q)


class Algorithm(Enum):
    prange = 0
    stern = 1
    bjmm = 2


class Algorithms:
    """
    wrapper around `CryptographicEstimators` optimization routines
    """
    def __init__(self, algo_num: Algorithm, problem: Problem, parameters={}) -> None:
        self.__algorithm_number = algo_num
        self.problem = problem
        self.parameters = parameters

        self.p = None
        if problem.q == 2:
            if self.__algorithm_number == Algorithm.prange:
                self.p = SDPrange(problem.to_sd())
            elif self.__algorithm_number == Algorithm.stern:
                self.p = SDDumer(problem.to_sd())
            elif self.__algorithm_number == Algorithm.bjmm:
                self.p = SDBJMM(problem.to_sd()) # TODO her TMTO
            else:
                assert False
        if problem.q > 2:
            if self.__algorithm_number == Algorithm.prange:
                self.p = SDFqPrange(problem.to_sdfq())
            if self.__algorithm_number == Algorithm.stern:
                self.p = SDFqStern(problem.to_sdfq())
        assert self.p


        # set predefined parameters:
        self.p.set_parameters(parameters)

        self.verbose = self.p._get_verbose_information()
        self.result = {"perms": self.verbose["permutations"], 
                       "gaus": self.verbose["gauss"],
                       "lists": self.verbose["lists"],
                       "time": self.p.time_complexity(),
                       "memory": self.p.memory_complexity(),
                       "params": self.p.optimal_parameters()}
        print("DEBUG", self.result)
    
    def optimize(self):
        """
        returns: {
            'perms': X,
            'gaus': X,
            'time': X
            'memory': X
            'params': {XX},
            'lists': [x,y,z]
        }
        """
        return self.result

    def get_time(self):
        """NOTE: call `optimize before`"""
        return self.result["time"]


def test_algorithms():
    p = Problem(100, 50, 10, 2)
    a = Algorithms(Algorithm.bjmm, p, {"p": 1})


class Decoding:
    """mother of all decoding classes"""
    compilers = ["g++", "clang++"]
    search_path = ["/usr/bin"]
    debug_flags = ["-g", "-Og", "-DDEBUG", "-fopenmp"]
    cmake_executable = ["/usr/bin/env", "cmake"]
    main_executable = [""]

    def __init__(self, algorithm : Algorithm, problem: Problem, meta_params=None,
                 algo_params=None, H:str="", syndrome: str="", seed: int =0) -> None:
        """
        :param algorithm:
        :param problem:
        :param meta_params:
        :param algo_params:
        :param H:
        :param syndrome:
        """ 
        random.seed(seed)

        # build 
        self.tmp_project_dir = "/tmp/decoding"
        self.source_dir = "/home/duda/Downloads/crypto/decoding/decoding"# TODO

        self.algorithm = algorithm
        self.algoname = self.algorithm.name.lower()
        self.__algorithm_description = None 
        for algo in algos:
            if self.algoname == algo["name"]:
                self.__algorithm_description = algo 
                break
        assert self.__algorithm_description is not None

        self.problem = problem
        self.algo_params = algo_params
        self.algo_params_names = []
        self.meta_params = meta_params
        self.meta_params_names = []

        # this is the 
        self.params = []
        if algo_params is not None:
            self.params += algo_params
            for a in algo_params:
                self.algo_params_names.append(a["name"])

        if meta_params is not None:
            self.params += meta_params
            for a in meta_params:
                self.meta_params_names.append(a["name"])

        self.H = H 
        self.syndrome = syndrome
        if len(self.H) == 0 and len(self.syndrome) == 0:
            self.gen_random_instance()

        # needed for the optimization process
        self.__min_time = inf
        # of the form: {
        #   "l" : 10,
        #   "p" : 1,
        #   ...
        #}
        self.__min_params = None

        # if set to true the algorithm will be compiled in debug mode
        self.__debug = False

        # if `true` each parameter set will be build and the minimal runtime 
        # is not the theoretical predicted but the interpolated from the 
        # benched runtime.
        self.__build = False
        self.__build_output = None
        self.__run_output = None

        # just a little helper flag to break the cycle like
        # `d.build().write_config().run()` if an error occurs.
        self.__successful_build = False
        self.__config_file = "../main.h" # TODO

    def gen_random_instance(self):
        """ creates a random parity check matrix and syndrome. 
        TODO: no guarantee of the existence of a solution is made.
        """
        for _ in range(self.problem.n * (self.problem.n - self.problem.k)):
            self.H += str(random.randint(0, self.problem.q - 1))
        for _ in range(self.problem.n - self.problem.k):
            self.syndrome += str(random.randint(0, self.problem.q - 1))

    def write_config(self, config=None):
        """ writes the `main.h` file to `self.__config_file` reading from 
        `config`. if `config is none `self.__min_params` will be used.
        """
        if config is None:
            config = self.__min_params
        if config is None:
            print("please generate the parameters first")
            return self

        with open(self.__config_file, "w") as f:
            f.write("#include <cstdint>\n")
            f.write("""constexpr uint32_t n = {n};
constexpr uint32_t k = {k};
constexpr uint32_t w = {w};
constexpr uint32_t q = {q};
""".format(n=self.problem.n,k=self.problem.k,q=self.problem.q,w=self.problem.w))
            
            f.write("constexpr char h[] = \"{s}\";\n".format(s=self.H))
            f.write("constexpr char s[] = \"{s}\";\n".format(s=self.syndrome))

            # print remaining options
            for k, v in config.items():
                f.write("constexpr uint32_t {k} = {v};\n".format(k=k, v=str(v)))
            
            f.write("static constexpr ConfigISD isdConfig{.n=n,.k=k,.q=2,.w=w,.p=p,.l=l,.c=c,.threads=t};")

            assert self.__algorithm_description is not None
            f.write("static constexpr {config_name} config{"
                    .format(config_name=self.__algorithm_description["config_name"]))
            for k, _ in self.__algorithm_description["parameters"]:
                f.write(".{k}={v},".format(k=k, v=config[k]))
            f.write("};")

            f.write("""auto get_algorithm() noexcept
    {class_name}<isdConfig, config> algo{};
    return algo;
}
""".format(class_name=self.__algorithm_description["class_name"]))

        return self

    def set_build(self):
        self.__build = True
        return self
    
    def __create_build_env(self):
        """ runs `cmake -B` command.
        TODO: should check if the current dir is writeable. if not move the whole
        whing to a tmp dir
        """
        t = "Debug" if self.__debug else "Release"
        cmd = Decoding.cmake_executable + ["-B", self.tmp_project_dir, 
               "-DCMAKE_BUILD_TYPE={t}".format(t=t), "-S", self.source_dir]
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        p.wait()

        if p.returncode != 0 and p.returncode is not None:
            self.__successful_build = False
            assert p.stdout
            print("couldn't execute: %s", " ".join(cmd))
            print(p.stdout.read())
            return self
            
        self.__successful_build = True
        return self

    def __run_parse_output(self):
        """ parses the output of the `self.build()` function """
        j = {}
        if self.__run_output is None:
            print("nothing build so far")
            return {}

        for line in self.__run_output:
            jj = json.loads(line)
            j.update(jj)
        return j

    def build(self, iters:int=1000):
        """ builds the `main` target """
        self.__create_build_env()
        cmd = Decoding.cmake_executable + ["--build", self.tmp_project_dir, 
                                           "--target", "main",]
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        p.wait()

        assert p.stdout
        if p.returncode != 0 and p.returncode is not None:
            print("couldn't execute: %s", " ".join(cmd))
            print("error msg:")
            print(p.stdout.read())
            return self
        
        self.__build_output = p.stdout.readlines()
        self.__build_output = [d.decode("utf-8").strip("\n") for d in self.__build_output]
        return self

    def optimize(self, params=None):
        """
        :param params: must be of the form:
            [{
                "name": "l",
                "max": 10,
                "min": 10,
                "step": 1,
            }, ...]
        """
        def extract_meta_params(cs):
            """ extracts from the current optimization state all the meta params
            and parses them in the correct form ({"t": 1, ...}) for the 
            `self.__min_params` dict.
            """
            ret = {}
            for k, v in cs.items():
                if k in self.meta_params_names:
                    ret[k] = v
            return ret

        if params is not None:
            self.params = params

        if self.params is None:
            # if no parameters are give we simply run the theoretic optimizer 
            # and output its runtime. Note: if the `build()` flag is set we 
            # additionally build the optimal parameters and run them.
            algorithm = Algorithms(self.algorithm, self.problem)
            opt = algorithm.optimize()
            self.__min_time = opt["time"]
            self.__min_params = opt["params"]
            self.__min_params.update(extract_meta_params({})) # TODO
            return self

        # run the practical optimizer
        assert type(self.params) == list
        parameters = {p["name"]: p["min"] for p in self.params}
        sp = 0
        while True and len(self.params) > 0:
            # filter out everything which is not part of the algorithm parameters
            tp = {k: v for k, v in parameters.items() if k in self.algo_params_names }
            algorithm = Algorithms(self.algorithm, self.problem, parameters=tp)
            opt = algorithm.optimize()
            if self.__build:
                p = copy.copy(opt["params"])
                p.update(extract_meta_params(parameters))
                self.write_config(p)
                self.build()
                self.run()
                t = self.__run_parse_output()
                print("DEBUG", t)
                if "walltime" in t:
                    opt["time"] = t["walltime"]
                else:
                    opt["time"] = inf
                
            if opt["time"] < self.__min_time:
                self.__min_time = opt["time"]
                self.__min_params = opt["params"]
                self.__min_params.update(extract_meta_params(parameters))

            # advance to the next state
            cs = self.params[sp]["name"]
            parameters[cs] += self.params[sp]["step"]
            ssp = sp
            # walk up
            while sp < len(self.params) and parameters[cs] > self.params[sp]["max"]:
                sp += 1
                if sp == len(self.params):
                    sp -= 1
                    break

                parameters[self.params[sp]["name"]] += self.params[sp]["step"]
                cs = self.params[sp]["name"]

            if parameters[self.params[sp]["name"]] > self.params[sp]["max"]:
                break

            # walk down
            cs = self.params[ssp]["name"]
            while parameters[cs] > self.params[ssp]["max"] and ssp >= 0:
                # reset the old one
                parameters[cs] = self.params[ssp]["min"]
                if ssp > 0:
                    ssp -= 1
                    cs = self.params[ssp]["name"]

            sp = ssp
        return self

    def run(self):
        if self.__successful_build == False:
            print("configuration was not successfully build. Not running")
            return self

        cmd = [self.tmp_project_dir + "/main"] # TODO
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        p.wait()

        assert p.stdout
        if p.returncode != 0 and p.returncode is not None:
            self.__successful_build = False
            print("couldn't execute: %s", " ".join(cmd))
            print(p.stdout.read())
            return self
            
        self.__run_output = p.stdout.readlines()
        self.__run_output = [d.decode("utf-8").strip("\n") for d in self.__run_output]
        return self

    def yolo(self):
        return self.run()

    def min(self):
        """ returns the current min time and parameter set """
        return self.__min_time, self.__min_params

    def debug(self):
        """ add the debug flag to the compile command """
        self.__debug = True
        return self


def test_decoding():
    """ simple tests around the `decoding` class """
    problem = Problem(n=100, k=50, w=10, q=2)
    d = Decoding(Algorithm.stern, problem, [{"name": "l", "min": 10, "max": 10 ,"step": 1}])
    print(d.optimize().min())
    d.write_config().debug().build()



if __name__ == "__main__":
    #test_algorithms()
    #test_decoding()
    #test_get_decodingchallenge()

    parser = argparse.ArgumentParser(description='Mother of all ISD algorithms.')
    parser.add_argument('-n', help='code length', type=int)
    parser.add_argument('-k', help='code dimension', type=int)
    parser.add_argument('-w', help='code weight', type=int)
    parser.add_argument('-q', help='field size', default=2, type=int)

    parser.add_argument('-l', help='', action=Range, type=str)
    parser.add_argument('-p', help='', action=Range, type=str)

    parser.add_argument('--url', help='url')
    parser.add_argument('-H', help='parity check matrix', type=str)
    parser.add_argument('-s', help='syndrome', type=str)
    parser.add_argument('stdin', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    
    subparsers = parser.add_subparsers(dest="algorithm", help='algorithms')
    subparsers.required = True

    # create_stern_argparser(parser, subparsers)
    # create_bjmm_argparser(parser, subparsers)
    create_subparsers(subparsers)

    add_args = []
    add_args.append(parser.add_argument('-c', help='marcov chain parameter', dest='meta_params',
                        action=Range, type=str, default="0", const="0"))
    add_args.append(parser.add_argument('-t', help='number of outer threads', action=Range,
                        type=str, default="1", dest="meta_params", const="1"))
    args = parser.parse_args()
    if not sys.stdin.isatty():
        stdin = parser.parse_args().stdin.read().splitlines()
        args.n, args.k, args.w, args.q, args.H, args.s  = parse_decodingchallenge(stdin)

    if args.url:
        if args.n or args.k or args.w:
            print("please either pass `--url` or `-n,-k,-w,-q` flags")
            exit(1)
        args.n, args.k, args.w, args.q, args.H, args.s = get_decodingchallenge(args.url)
    else:
        if not (args.n and args.k and args.w and args.q):
            print("please either pass all `-n,-k,-w,-q` flags")
            exit(1)

    for a1 in add_args:
        a1(parser, args, a1.default, 'no string') # call action

    problem = Problem(n=args.n, k=args.k, w=args.w, q=args.q)

    d = Decoding(Algorithm[args.algorithm], problem, meta_params=args.meta_params, 
                 algo_params=args.algo_param, H=args.H, syndrome=args.s)
    #print(d.optimize(None).min())
    d.set_build().optimize().write_config()#.build()

    # code parameters
    exit(0)
    
    # algorithmic stuff
    parser.add_argument('-a', '--algorithm', help="Which algorithm"
                        "Prange: 1, Stern/Dumer:2, BJMM: 3", 
                        default=3, type=int, required=False)
    parser.add_argument('--quasicyclic', help='attack Quasi Cyclic codes.', 
                        action='store_true')
    parser.add_argument('--outer_threads', 
                        help="number of openmp threads the algorithm is parallelize"
                        "0 means all available.", default=1, 
                        type=int, required=False)
    parser.add_argument('-t' , '--threads', 
                        help="number of openmp the algorithm can use to parallelize"
                        "the tree. 0 means all available.", default=1, 
                        type=int, required=False)
    parser.add_argument('-l' , '--param_l', default=0, type=str, action=Range, required=False,
                        help='l parameter of the algorithm', )
    parser.add_argument('-l1', '--param_l1', default=0, type=str, action=Range, required=False,
                        help='l1 parameter of the algorithm. l2 will be'
                        'automatically set to l-l1')
    parser.add_argument('-p' , '--param_p', default=0, type=str, action=Range, required=False,
                        help='p parameter=weight')
    parser.add_argument('--gaus_c',  default=0, type=int, required=False,
                        help='Number of columns to swap in the permutation phase.'
                        'If set to zero a random permutation is choosen.')

    # build parameters
    parser.add_argument('--build', help='build only. Do not run', 
                        action='store_true')
    parser.add_argument("--config", help="Creates only configuration header. "
                        "Does not build or run.", action="store_true")
    parser.add_argument('--bench', help='Tries to find the best parameter.', 
                        action='store_true')
    parser.add_argument('--seconds', help="Kill a program after x seconds. "
                        "Default: 0: run infinite", default=0, type=int,
                        required=False)
    parser.add_argument('--loops', default=-1, type=int, required=False,
                        help='After how many loops should the program quit?')
    parser.add_argument('--challenge_file', default="", type=str, required=False,
                        help='challenge file. If no file is given a random instance is generated')
    parser.add_argument('--no_logging', help='Disable all of the internal logging of the algorithms, except the final error found', action='store_true')
    parser.add_argument('--print_loops', help='print every X loops some status information.', default=10000, type=int, required=False)
    parser.add_argument('--exit_loops', help='check every X loops, if another thread is already finished', default=10000, type=int, required=False)

    parser.add_argument('-m', '--memory', help='maximum memory allowed, this is only used in', default=inf, type=int, required=False)

    # hashmap stuff
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



