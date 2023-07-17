#!/usr/bin/python3
import re, sys
from os import listdir
from os.path import isfile, join

def parse_n(file: str):
    regex = r".*n(\d+).*"
    data = re.findall(regex, file)
    return data[0]


def parse_p(file: str):
    regex = r".*p(\d+).*"
    data = re.findall(regex, file)
    return data[0]


def parse_lph(file: str):
    regex = r".*loops/h:[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"
    with open(file, 'r') as f:
        for line in f.readlines():
            data = re.findall(regex, line)
            if len(data) > 0 and len(data[0]) > 0:
                return data[0][0]+data[0][2]

    return False


def parse_time(file: str):
    regex = r".*found solution, time: (\d+.\d+).*"
    with open(file, 'r') as f:
        for line in f.readlines():
            data = re.findall(regex, line)
            if len(data) > 0:
                return data[0]

    return False


def parse_round(file: str):
    regex = r".*, round: (\d+).*"
    with open(file, 'r') as f:
        for line in f.readlines():
            data = re.findall(regex, line)
            if len(data) > 0:
                return data[0]

    return False


def calc_precomputation_time_filter(a):
    return "Precomputation took:" in a


def calc_loops_filter(a):
    return "BJMMF: tid:" in a


def calc_last_loops_lines(lines: list):
    lines.reverse()
    regex = r"BJMMF: tid:\d+, loops: (\d+)"
    for line in lines:
        try:
            f = float(re.findall(regex, line)[0])
            return f
        except:
            pass

    return -1


def calc_precomputation_time_lines(lines: list, threads=256):
    regex = r"Precomputation took: (\d*.\d*) s"
    data = []
    for line in lines:
        try:
            f = float(re.findall(regex, line)[0])
            data.append(f)
        except:
            pass

    if len(data) == 0:
        return 0
    return sum(data) / len(data) / threads


def calc_loops_lines(lines: list, ll=256):
    if len(lines) > ll:
        lines = lines[-ll:]

    regex = r"BJMMF: tid:\d+, loops: (\d+)"
    data = [re.findall(regex, line) for line in lines]
    data2 = []
    for d in data:
        if len(d) != 0:
            f = float(d[0])
            if f < 1000:
                f += 1
            data2.append(f)

    if len(data2) == 0:
        return 0
    return sum(data2) / len(data2)


def calc_last_loops(file: str):
    f = open(file, 'r')
    lines = list(filter(calc_loops_filter, f.readlines()))
    return calc_last_loops_lines(lines)


def calc_precomputation_time(file: str, threads=256):
    f = open(file, 'r')
    lines = list(filter(calc_precomputation_time_filter, f.readlines()))
    return calc_precomputation_time_lines(lines, threads)


def calc_loops(file: str, ll=256):
    f = open(file, 'r')
    lines = list(filter(calc_loops_filter, f.readlines()))
    return calc_loops_lines(lines, ll)


def calc_lines(lines, time, single_tree=False):
    if single_tree:
        return calc_last_loops_lines(lines)
    pre = calc_precomputation_time_lines(lines)
    loops = calc_loops_lines(lines)
    tt = time - pre
    lph = 3600 * loops / tt
    return lph


def calc(file: str, time=600, single_tree=False, lessout=False):
    if single_tree:
        return calc_last_loops(file)

    lph = parse_lph(file)
    if lph:
        pre = calc_precomputation_time(file)
        loops = parse_round(file)
        time = parse_time(file)
        print(parse_n(file), parse_p(file), "lph:", lph, " pre:", pre, "time:", time, "loops:", loops)
        return [parse_n(file), parse_p(file), lph, pre, time, loops, file]
    else:
        pre = calc_precomputation_time(file)
        loops = calc_loops(file)

        if pre > time:
            print("we have a problem:")
            return

        tt = time - pre
        lph = 3600. * loops / tt
        print(parse_n(file), parse_p(file), "lph:", lph, " pre:", pre, "time:", time, "loops:", loops)
        if lessout:
            return lph
        return [parse_n(file), parse_p(file), lph, pre, time, loops, file]


def mo_calc(file):
    with open(file, 'r') as f:
        threads = 256
        seconds = 600

        lines = list(f.readlines())
        revlines = list(reversed(lines))

        # first calc the delay
        #firstregex = "Time: (\d+.\d+), clock Time: (\d+.\d+), lps: 0"
        firstregex = "Init took: (\d+.\d+) s"
        firstregexdata = []
        for line in lines:
            data = re.findall(firstregex, line)
            if data:
                firstregexdata.append(float(data[0]))


        secregex = "currently at 0 loops, (\d+.\d+)s, 0lps"
        secregexdata = []
        for line in lines:
            data = re.findall(secregex, line)
            if data:
                secregexdata.append(float(data[0]))

        thregex = "currently at (\d+) .*"
        thregexdata = []
        for line in revlines[:threads]:
            data = re.findall(thregex, line)
            if data:
                thregexdata.append(float(data[0]))

        offset1 = sum(firstregexdata)/len(firstregexdata)
        offset2 = sum(secregexdata)/len(secregexdata)
        loops = sum(thregexdata)/len(thregexdata)
        offset1 = seconds/(seconds-offset1/threads)
        offset2 = seconds/(seconds-offset2/threads)

        print(loops)
        print("RealLoops:", loops*offset1, offset1)
        print("RealLoops:", (loops-1)*offset2, offset2)


def main(argv):
    if len(argv) < 2:
        return
    if isfile(argv[1]):
        return mo_calc(argv[1])
    return

    if isfile(argv[1]):
        return calc(argv[1])

    files = [argv[1] + "/" + f for f in listdir(argv[1]) if isfile(join(argv[1], f))]
    if len(argv) > 2:
        files2 = []
        for file in files:
            if int(argv[2]) < 100:
                if parse_p(file) == argv[2]:
                    files2.append(file)
            else:
                if parse_n(file) == argv[2]:
                    files2.append(file)
    else:
        files2 = files

    files2.sort()
    data = {}
    name = 1
    for file in files2:
        d = calc(file)
        if d[0] != name:
            name = d[0]
            data[name] = []

        data[name].append(d)


    data2 = {}
    for key in data.keys():
        data2[key] = sum([float(d[2]) for d in data[key]])/len(data[key])

    for key, value in data2.items():
        print(key, value)

if __name__ == "__main__":
    main(sys.argv)
