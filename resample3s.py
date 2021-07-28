#!/usr/bin/env python3

"""Non-overlapping resampling for 3s"""

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
import sys
import random
import re

__all__ = ["ResampleError", "ConfigParsingError", "SpeciesNameError", "MapFileError", "resample"]

class ResampleError(Exception):
    """Base class for all resample3s exceptions"""

class ConfigParsingError(ResampleError):
    """Error in parsing formatted string 'configstr'"""

class SpeciesNameError(ResampleError):
    """Error in species names"""

class MapFileError(ResampleError):
    """Error in map file that maps the individuals to the species"""

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

def _printToStderr(*args, **kwargs):
    print(*args, file = sys.stderr, flush = True, **kwargs)

def _printWarning(msg, cat = None):
    if cat is None:
        cat = 'Warning'
    _printToStderr(cat, ': ', msg, sep = '')

def _parseConfigStrs(configStrs):
    try:
        iter(configStrs)
    except TypeError:
        raise ConfigParsingError("'%s' not iterable" % configStrs) from None

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
                    itemLst.append((_configMap[item[0]], ntimes))
                elif ntimes < 0:
                    raise Exception

            key = tuple((k, sum(x[1] for x in g))
                        for k, g in itertools.groupby(itemLst, lambda x: x[0]))

            if nloci > 0:
                if key in dct:
                    dct[key] += nloci
                else:
                    dct[key] = nloci
            elif nloci < 0:
                raise Exception

        except Exception:
            raise ConfigParsingError("'%s' invalid" % configStr) from None

    configs = []
    for key, nloci in dct.items():
        totals = [0] * 3
        for config, ntimes in key:
            for i in range(3):
                totals[i] += _seqNums[config][i] * ntimes
        configs.extend([(key, tuple(totals))] * nloci)
    return configs

def _processBlock(idx, counts, lines, configs, fout, showWarning):
    seqNum, seqLen, seqCount = counts
    if seqNum != seqCount:
        if showWarning:
            if seqCount == 0:
                _printWarning('sequences not tagged by individual/species label in locus %d. Skipped.'
                              % (idx + 1))
            else:
                _printWarning('incorrect number of sequences in locus %d. Skipped.' % (idx + 1))
        return 0

    nums = tuple(len(i) for i in lines)
    if sum(nums) == 0:
        if showWarning:
            _printWarning('no sequences of the 3 specified species in locus %d. Skipped.' % (idx + 1))
        return 0

    maxIdx = len(configs)
    if idx >= maxIdx:
        if showWarning:
            _printWarning('nothing to do for locus %d. Skipped.' % (idx + 1))
        return 0

    k = idx
    while k < maxIdx and min(i - j for i, j in zip(nums, configs[k][1])) < 0:
        k += 1

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

                if showWarning:
                    warningMsg = "cannot extract %d %s of config '%s' from locus %d." % (
                                 ntimes, 'locus' if ntimes == 1 else 'loci', mainStr, idx + 1)
                    if mainTimes > 0 and extraTimes > 0:
                        warningMsg += " Extract %d %s of config '%s' and %d %s of config '%s' instead." % (
                                      mainTimes, 'locus' if mainTimes == 1 else 'loci', mainStr,
                                      extraTimes, 'locus' if extraTimes == 1 else 'loci', extraStr)
                    elif mainTimes > 0 and extraTimes == 0:
                        warningMsg += " Extract %d %s of config '%s' instead." % (
                                      mainTimes, 'locus' if mainTimes == 1 else 'loci', mainStr)
                    elif mainTimes == 0 and extraTimes > 0:
                        warningMsg += " Extract %d %s of config '%s' instead." % (
                                      extraTimes, 'locus' if extraTimes == 1 else 'loci', extraStr)
                    _printWarning(warningMsg)

        totals = tuple(i - j for i, j in zip(nums, remains))
        if sum(totals) == 0:
            if showWarning:
                _printWarning('nothing to do for locus %d. Skipped.' % (idx + 1))
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

def _resample(fin, indMap, configs, fout, showWarning):
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
                lociNums.append(_processBlock(idx, counts, lines, configs, fout, showWarning))
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
            lociNums.append(_processBlock(idx, counts, lines, configs, fout, showWarning))
            idx += 1
        if idx < maxIdx and showWarning:
            _printWarning('%d %s processed. Not enough loci for resampling. Stopped.'
                          % (idx, 'locus' if idx == 1 else 'loci'))
    return lociNums

def resample(seqFile, speciesNames, configStrs, mapFile = None, outFile = None, showWarning = True):
    """Randomly shuffle the collection of configuration lists parsed from 'configStrs', and then
    from each locus in 'seqFile' randomly sample a specific number of loci of 2 or 3 sequences
    without replacement according to the corresponding configuration list.

    Return a list of the number of loci resampled from each locus in 'seqFile'.

    seqFile:      Name of sequences file, defaulting to sys.stdin if '-'
    speciesNames: Collection of names of species 1, 2, 3
    configStrs:   Collection of formatted string 'configstr' that is defined below
                  configstr := <list>[#<nloci>]
                       list := <item> | {<item>[,<item>]...}
                       item := <config>[:<ntimes>]
    config:       Data configuration string at a locus (e.g. 111, 112, 123, 12, 33, ...)
    ntimes:       Number of times to extract loci with the configuration <config> from arbitrary
                  locus, defaulting to 1 if omitted
    nloci:        Number of loci from each of which a number of loci with the configuration list
                  <list> are to be extracted, defaulting to 1 if omitted
    mapFile:      Name of map file that maps the individuals to the species
    outFile:      Name of output file, defaulting to sys.stdout if None
    showWarning:  Whether to print warning messages to sys.stderr
    """

    configs = _parseConfigStrs(configStrs)
    random.shuffle(configs)

    try:
        spMap = {
            speciesNames[0].strip():1,
            speciesNames[1].strip():2,
            speciesNames[2].strip():3,
            }
    except Exception:
        raise SpeciesNameError('need 3 strings of species names') from None
    if '' in spMap:
        raise SpeciesNameError('empty species name')
    if len(spMap) != 3:
        raise SpeciesNameError("duplicate species names '%s', '%s', '%s'"
                               % (speciesNames[0], speciesNames[1], speciesNames[2]))

    if mapFile is not None:
        indMap = {}
        with fileinput.input(mapFile) as fin:
            for line in fin:
                strLst = [i.strip() for i in line.strip().split()]
                if len(strLst) == 2 and strLst[0].find('//') == -1:
                    if strLst[0] not in indMap:
                        indMap[strLst[0]] = spMap.get(strLst[1])
                    else:
                        raise MapFileError("duplicate individual label '%s' in map file '%s'"
                                           % (strLst[0], mapFile))
                else:
                    break
    else:
        indMap = spMap

    with fileinput.input(seqFile) as fin:
        if outFile is not None:
            with open(outFile, 'w') as fout:
                lociNums = _resample(fin, indMap, configs, fout, showWarning)
        else:
            lociNums = _resample(fin, indMap, configs, sys.stdout, showWarning)
    return lociNums

def main(args):
    usageStr = """\
Usage: %s [-m <mapfile>] <seqfile> <species1> <species2> <species3> <configstr>[ <configstr>]...

Randomly shuffle the collection of configuration lists parsed from all parameters like
<configstr>, and then from each locus in <seqfile> randomly sample a specific number of loci
of 2 or 3 sequences without replacement according to the corresponding configuration list.

    seqfile:   Name of sequences file, defaulting to sys.stdin if '-'
    mapfile:   Name of map file that maps the individuals to the species
    species1:  Name of species 1
    species2:  Name of species 2
    species3:  Name of species 3
    configstr: Formatted string that is defined below
               configstr := <list>[#<nloci>]
                    list := <item> | {<item>[,<item>]...}
                    item := <config>[:<ntimes>]
    config:    Data configuration string at a locus (e.g. 111, 112, 123, 12, 33, ...)
    ntimes:    Number of times to extract loci with the configuration <config> from arbitrary
               locus, defaulting to 1 if omitted
    nloci:     Number of loci from each of which a number of loci with the configuration list
               <list> are to be extracted, defaulting to 1 if omitted

Report bugs to: Bo Xu <xuxbob@gmail.com>\
""" % (args[0])

    if len(args) < 2:
        offset = 0
    elif args[1] == '-h' or args[1] == '--help':
        _printToStderr(usageStr)
        return 0
    elif args[1] == '-m':
        mapFile = args[2]
        offset = 2
    elif args[1].startswith('-m'):
        mapFile = args[1][2:]
        offset = 1
    elif args[1].startswith('--map='):
        mapFile = args[1][6:]
        offset = 1
    else:
        mapFile = None
        offset = 0

    if len(args) < 6 + offset:
        _printToStderr(usageStr)
        return 2

    seqFile = args[1 + offset]
    speciesNames = args[2 + offset : 5 + offset]
    configStrs = args[5 + offset:]

    try:
        lociNums = resample(seqFile, speciesNames, configStrs, mapFile)
    except (ResampleError, FileNotFoundError) as e:
        _printToStderr(e.__class__.__name__, ': ', e, sep = '')
        return 2 if isinstance(e, ConfigParsingError) else 1
    else:
        processed = len(lociNums)
        resampled = sum(lociNums)
        _printToStderr('Resampled %d %s from %d %s.' % (
                       resampled, 'locus' if resampled == 1 else 'loci',
                       processed, 'locus' if processed == 1 else 'loci'))
        return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
