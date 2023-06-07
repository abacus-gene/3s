#!/usr/bin/env python3

"""Non-overlapping subsampling sequences for 1 to 3 species."""

# Copyright (C) 2021 Bo Xu <xuxbob@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import fileinput
import itertools
import os
import random
import re
import sys

__all__ = [
    "SubsampleError",
    "ConfigParsingError",
    "SpeciesNameError",
    "IndivSpeciesMapError",
    "printToStderr",
    "printWarning",
    "subsample",
    "getIndSpMap"
    ]

class SubsampleError(Exception):
    """Base class for all subsample3s exceptions"""

class ConfigParsingError(SubsampleError):
    """Error in parsing formatted string 'configstr'"""

class SpeciesNameError(SubsampleError):
    """Error in species names"""

class IndivSpeciesMapError(SubsampleError):
    """Error in the individuals-to-species map"""

_configMap = {
    '111': 0, '112': 1, '113': 2,
    '121': 3, '122': 4, '123': 5,
    '131': 6, '132': 7, '133': 8,
    '211': 9, '212':10, '213':11,
    '221':12, '222':13, '223':14,
    '231':15, '232':16, '233':17,
    '311':18, '312':19, '313':20,
    '321':21, '322':22, '323':23,
    '331':24, '332':25, '333':26,
     '11':27,  '12':28,  '13':29,
     '21':30,  '22':31,  '23':32,
     '31':33,  '32':34,  '33':35
    }

_labels = (
    (1, 1, 1), (1, 1, 2), (1, 1, 3),
    (1, 2, 1), (1, 2, 2), (1, 2, 3),
    (1, 3, 1), (1, 3, 2), (1, 3, 3),
    (2, 1, 1), (2, 1, 2), (2, 1, 3),
    (2, 2, 1), (2, 2, 2), (2, 2, 3),
    (2, 3, 1), (2, 3, 2), (2, 3, 3),
    (3, 1, 1), (3, 1, 2), (3, 1, 3),
    (3, 2, 1), (3, 2, 2), (3, 2, 3),
    (3, 3, 1), (3, 3, 2), (3, 3, 3),
       (1, 1),    (1, 2),    (1, 3),
       (2, 1),    (2, 2),    (2, 3),
       (3, 1),    (3, 2),    (3, 3)
    )

_seqNums = (
    (3, 0, 0), (2, 1, 0), (2, 0, 1),
    (2, 1, 0), (1, 2, 0), (1, 1, 1),
    (2, 0, 1), (1, 1, 1), (1, 0, 2),
    (2, 1, 0), (1, 2, 0), (1, 1, 1),
    (1, 2, 0), (0, 3, 0), (0, 2, 1),
    (1, 1, 1), (0, 2, 1), (0, 1, 2),
    (2, 0, 1), (1, 1, 1), (1, 0, 2),
    (1, 1, 1), (0, 2, 1), (0, 1, 2),
    (1, 0, 2), (0, 1, 2), (0, 0, 3),
    (2, 0, 0), (1, 1, 0), (1, 0, 1),
    (1, 1, 0), (0, 2, 0), (0, 1, 1),
    (1, 0, 1), (0, 1, 1), (0, 0, 2)
    )

_lociStr = ('loci', 'locus')

def printToStderr(*args, **kwargs):
    print(*args, file = sys.stderr, flush = True, **kwargs)

def printWarning(msg, cat = None):
    if cat is None:
        cat = 'Warning'
    printToStderr(cat, ': ', msg, sep = '')

def _parseConfigStrs(configStrs, nspecies):
    try:
        iter(configStrs)
    except TypeError:
        raise ConfigParsingError(f"'{configStrs}' not iterable") from None

    dct = {}
    for configStr in configStrs:
        try:
            strLst = [i.strip() for i in configStr.strip().split('#')]

            if len(strLst) == 1:
                nloci = 1
            elif len(strLst) == 2:
                nloci = int(strLst[1])
            else:
                raise Exception

            if strLst[0].startswith('{') and strLst[0].endswith('}'):
                strLst2 = [i.strip() for i in strLst[0][1:-1].strip().split(',')]
            else:
                strLst2 = [strLst[0]]

            strLst = [[j.strip() for j in i.split(':')] for i in strLst2]

            itemLst = []
            for item in strLst:
                if len(item) == 1:
                    ntimes = 1
                elif len(item) == 2:
                    ntimes = int(item[1])
                else:
                    raise Exception

                if ntimes > 0:
                    config = _configMap[item[0]]
                    if sum(_seqNums[config][i - 1] for i in range(3, nspecies, -1)) != 0:
                        raise Exception
                    itemLst.append((config, ntimes))
                elif ntimes < 0:
                    raise Exception

            key = tuple((k, sum(x[1] for x in g))
                        for k, g in itertools.groupby(itemLst, lambda x: x[0]))

            if nloci > 0:
                dct.setdefault(key, 0)
                dct[key] += nloci
            elif nloci < 0:
                raise Exception

        except Exception:
            raise ConfigParsingError(f"'{configStr}' invalid") from None

    configs = []
    for key, nloci in dct.items():
        totals = [0] * 3
        for config, ntimes in key:
            for i in range(3):
                totals[i] += _seqNums[config][i] * ntimes
        configs.extend([(key, tuple(totals))] * nloci)
    return configs

def _processBlock(idx, counts, lines, configs, greedy, fout, warning):
    seqNum, seqLen, seqCount = counts
    if seqNum != seqCount:
        if warning:
            if seqCount == 0:
                printWarning(f'sequences not tagged by individual/species label in locus {idx + 1}. Skipped.')
            else:
                printWarning(f'incorrect number of sequences in locus {idx + 1}. Skipped.')
        return 0

    nums = tuple(len(i) for i in lines)
    if sum(nums) == 0:
        if warning:
            printWarning(f'no sequences of the specified species in locus {idx + 1}. Skipped.')
        return 0

    maxIdx = len(configs)
    if idx >= maxIdx:
        if warning:
            printWarning(f'nothing to do for locus {idx + 1}. Skipped.')
        return 0

    k = idx
    while k < maxIdx and min(i - j for i, j in zip(nums, configs[k][1])) < 0:
        if greedy:
            k += 1
        else:
            k = maxIdx

    if idx <= k < maxIdx:
        if k != idx:
            configs[k], configs[idx] = configs[idx], configs[k]
        itemLst = configs[idx][0]
        totals = configs[idx][1]

    else:
        tmpLst = []
        remains = list(nums)
        for config, ntimes in configs[idx][0]:
            mainTimes = min(ntimes,
                            min(i // j for i, j in zip(remains, _seqNums[config]) if j > 0))
            for i in range(3):
                remains[i] -= _seqNums[config][i] * mainTimes
            tmpLst.append((config, mainTimes, ntimes - mainTimes))

        itemLst = []
        for config, mainTimes, extraTimes in tmpLst:
            itemLst.append((config, mainTimes))
            if extraTimes > 0:
                mainStr = ''.join(str(i) for i in _labels[config])
                ntimes = mainTimes + extraTimes
                extra = tuple(min(i, j) for i, j in zip(remains, _seqNums[config]))
                if sum(extra) == 2:
                    extraStr = ''.join(str(i + 1) * e for i, e in enumerate(extra))
                    extraConfig = _configMap[extraStr]
                    extraTimes = min(extraTimes,
                                     min(i // j for i, j in zip(remains, _seqNums[extraConfig]) if j > 0))
                    for i in range(3):
                        remains[i] -= _seqNums[extraConfig][i] * extraTimes
                    itemLst.append((extraConfig, extraTimes))
                else:
                    extraTimes = 0

                if warning:
                    warningMsg = f"cannot extract {ntimes} {_lociStr[ntimes == 1]} of config '{mainStr}' from locus {idx + 1}."
                    mainMsg = f"{mainTimes} {_lociStr[mainTimes == 1]} of config '{mainStr}'" if mainTimes > 0 else ''
                    extraMsg = f"{extraTimes} {_lociStr[extraTimes == 1]} of config '{extraStr}'" if extraTimes > 0 else ''
                    if mainTimes > 0 and extraTimes > 0:
                        warningMsg += f' Extract {mainMsg} and {extraMsg} instead.'
                    elif mainTimes > 0 and extraTimes == 0:
                        warningMsg += f' Extract {mainMsg} instead.'
                    elif mainTimes == 0 and extraTimes > 0:
                        warningMsg += f' Extract {extraMsg} instead.'
                    printWarning(warningMsg)

        totals = tuple(i - j for i, j in zip(nums, remains))
        if sum(totals) == 0:
            if warning:
                printWarning(f'nothing to do for locus {idx + 1}. Skipped.')
            return 0

    rndRet = [random.sample(range(i), j) for i, j in zip(nums, totals)]
    lociNum = 0
    for config, ntimes in itemLst:
        for j in range(ntimes):
            print('\n\n', sum(_seqNums[config]), seqLen, '\n', file = fout)
            for i in _labels[config]:
                print(lines[i - 1][rndRet[i - 1][0]], end = '', file = fout)
                del rndRet[i - 1][0]
        lociNum += ntimes
    return lociNum

def _subsample(fin, indMap, configs, greedy, fout, warning):
    idx = 0
    maxIdx = len(configs)
    counts = [0] * 3
    lines = [[], [], []]
    inBlock = False
    headPattern = re.compile(r'\s*\d+\s+\d+\s')
    seqTagPattern = re.compile(r'[^^]*\^(\S+)\s')
    lociNums = []

    for line in fin:
        if idx == maxIdx:
            break
        elif headPattern.match(line):
            if inBlock:
                lociNums.append(_processBlock(idx, counts, lines, configs, greedy, fout, warning))
                idx += 1
            else:
                inBlock = True
            counts[0:2] = [int(i) for i in line.strip().split()[0:2]]
            counts[2] = 0
            for i in lines:
                i.clear()
        else:
            ret = seqTagPattern.search(line)
            if ret:
                counts[2] += 1
                if ret.group(1) in indMap and indMap[ret.group(1)]:
                    lines[indMap[ret.group(1)] - 1].append(line)
    if idx < maxIdx:
        if inBlock:
            lociNums.append(_processBlock(idx, counts, lines, configs, greedy, fout, warning))
            idx += 1
        if idx < maxIdx and warning:
            printWarning(f"{idx} {_lociStr[idx == 1]} processed. Not enough loci for subsampling. Stopped.")
    return lociNums

def subsample(seqFile, speciesNames, configStrs, indSpMap = None, greedy = False, outFile = None, warning = True):
    """Randomly shuffle the collection of configuration lists parsed from 'configStrs', and then from each
    locus in 'seqFile' randomly sample a specific number of loci of 2 or 3 sequences without replacement
    according to the corresponding configuration list.

    Return a list of the number of loci subsampled from each locus in 'seqFile'.

    seqFile:      Name of sequences file in the phylip/paml format, defaulting to sys.stdin if '-'
    speciesNames: Collection of 1 to 3 strings as names of species 1, 2, 3, rspectively
    configStrs:   Collection of formatted string 'configstr' that is defined below
                  configstr := <list>[#<nloci>]
                       list := <item> | {<item>[,<item>]...}
                       item := <config>[:<ntimes>]
    config:       Data configuration string at a locus (e.g. 111, 112, 123, 12, 33, ...)
    ntimes:       Number of times to extract loci with the configuration <config> from arbitrary locus,
                  defaulting to 1 if omitted
    nloci:        Number of loci from each of which a number of loci with the configuration list <list>
                  are to be extracted, defaulting to 1 if omitted
    indSpMap:     Dictionary that maps the individuals to the species
    greedy:       Whether to sample as many sequences as possible at the cost of possible loss of
                  sampling randomness
    outFile:      Name of output file, defaulting to sys.stdout if None
    warning:      Whether to print warning messages to sys.stderr
    """

    try:
        nspecies = len(speciesNames[:3])
        if nspecies == 0:
            raise Exception
        spMap = { nm.strip():i + 1 for i, nm in enumerate(speciesNames[:3]) }
    except Exception:
        raise SpeciesNameError('need 1 to 3 strings as species names') from None
    if '' in spMap:
        raise SpeciesNameError('empty species name')
    if nspecies != len(spMap):
        raise SpeciesNameError(f"duplicate species names: {', '.join(repr(i) for i in speciesNames)}")

    configs = _parseConfigStrs(configStrs, nspecies)
    random.shuffle(configs)

    if indSpMap is not None:
        if not isinstance(indSpMap, dict):
            raise IndivSpeciesMapError(f"'{indSpMap}' not an instance of dict")
        for key in spMap.keys():
            if key not in indSpMap.values():
                raise IndivSpeciesMapError(f"species name '{key}' not in the individuals-to-species map")
        indMap = {}
        for key, value in indSpMap.items():
            indMap[key] = spMap.get(value)
    else:
        indMap = spMap

    with fileinput.input(seqFile) as fin:
        if outFile is not None:
            with open(outFile, 'w') as fout:
                lociNums = _subsample(fin, indMap, configs, greedy, fout, warning)
        else:
            lociNums = _subsample(fin, indMap, configs, greedy, sys.stdout, warning)
    return lociNums

def getIndSpMap(mapFile):
    """Read pairs of individual and species names from 'mapFile' and use them to build a dictionary.

    Return a dictionary that maps the individuals to the species.

    mapFile: Name of map file that maps the individuals to the species
    """

    indSpMap = {}
    with fileinput.input(mapFile) as fin:
        for line in fin:
            strLst = [i.strip() for i in line.strip().split()]
            if len(strLst) == 2 and strLst[0].find('//') == -1:
                if strLst[0] not in indSpMap:
                    indSpMap[strLst[0]] = strLst[1]
                else:
                    raise IndivSpeciesMapError(f"duplicate individual label '{strLst[0]}' in map file '{mapFile}'")
            else:
                break
    return indSpMap

def main(args):
    usageStr = """\
Usage: %s [-m <mapfile>] [-g] <seqfile> <species1>[,<species2>[,<species3>]] <configstr>[ <configstr>]...

Non-overlapping subsampling sequences for 1 to 3 species.

Randomly shuffle the collection of configuration lists parsed from all parameters like <configstr>,
and then from each locus in <seqfile> randomly sample a specific number of loci of 2 or 3 sequences
without replacement according to the corresponding configuration list.

    seqfile:   Name of sequences file in the phylip/paml format, defaulting to sys.stdin if '-'
    speciesN:  Name of species N, N = 1, 2, 3
    configstr: Formatted string that is defined below
               configstr := <list>[#<nloci>]
                    list := <item> | {<item>[,<item>]...}
                    item := <config>[:<ntimes>]
    config:    Data configuration string at a locus (e.g. 111, 112, 123, 12, 33, ...)
    ntimes:    Number of times to extract loci with the configuration <config> from arbitrary locus,
               defaulting to 1 if omitted
    nloci:     Number of loci from each of which a number of loci with the configuration list <list>
               are to be extracted, defaulting to 1 if omitted

    -m <mapfile>, --map=<mapfile>   Name of map file that maps the individuals to the species
    -g,           --greedy          Sample as many sequences as possible at the cost of possible
                                    loss of sampling randomness
    -h,           --help            Display this help and exit

Report bugs to: Bo Xu <xuxbob@gmail.com>\
""" % os.path.basename(args[0])

    import getopt

    try:
        optLst, argLst = getopt.getopt(args[1:], 'ghm:', ['greedy', 'help', 'map='])

        mapFile = None
        greedy = False
        for opt, arg in optLst:
            if opt in ('-g', '--greedy'):
                greedy = True
            elif opt in ('-h', '--help'):
                printToStderr(usageStr)
                return 0
            elif opt in ('-m', '--map'):
                if mapFile is None:
                    mapFile = arg
                else:
                    raise getopt.GetoptError("duplicate option '-m', '--map'")
            else:
                raise getopt.GetoptError('unhandled option')

        if len(argLst) < 3:
            raise getopt.GetoptError('at least 3 non-option arguments are required')
        seqFile = argLst[0]
        speciesNames = argLst[1].split(',')
        configStrs = argLst[2:]

    except getopt.GetoptError as e:
        printToStderr('CommandLineError: ', e, '. Use option -h for help.', sep = '')
        return 2

    try:
        indSpMap = getIndSpMap(mapFile) if mapFile is not None else None
        lociNums = subsample(seqFile, speciesNames, configStrs, indSpMap, greedy)
    except (SubsampleError, FileNotFoundError) as e:
        printToStderr(e.__class__.__name__, ': ', e, sep = '')
        return 2 if isinstance(e, ConfigParsingError) else 1
    else:
        processed = len(lociNums)
        subsampled = sum(lociNums)
        printToStderr(f"Subsampled {subsampled} {_lociStr[subsampled == 1]} from {processed} {_lociStr[processed == 1]}.")
        return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
