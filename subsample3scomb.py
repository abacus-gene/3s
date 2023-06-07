#!/usr/bin/env python3

"""Non-overlapping subsampling sequences for every combinations of 3 species."""

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

import itertools
import os
import sys
import subsample3s as ss

def main(args):
    usageStr = """\
Usage: %s [-m <mapfile>] [-g] <seqfile> <species1>,<species2>,<species3>[,<species>]... <nloci> <p123>

Non-overlapping subsampling sequences for every combinations of 3 species.

For each species combination with names i, j and k, sequences are subsampled and written to a file
named i-j-k.phy in the phylip/paml format.

    seqfile: Name of sequences file in the phylip/paml format, defaulting to sys.stdin if '-'
    species: Name of at least 3 species
    nloci:   Number of loci in <seqfile> for subsampling
    p123:    Proportion of configuration '123'. The other 6 configurations 112, 113, 122, 223, 133
             and 233 share the rest equally.

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
                ss.printToStderr(usageStr)
                return 0
            elif opt in ('-m', '--map'):
                if mapFile is None:
                    mapFile = arg
                else:
                    raise getopt.GetoptError("duplicate option '-m', '--map'")
            else:
                raise getopt.GetoptError('unhandled option')

        if len(argLst) < 4:
            raise getopt.GetoptError('at least 4 non-option arguments are required')
        seqFile = argLst[0]
        speciesNames = argLst[1].split(',')
        if len(speciesNames) < 3:
            raise getopt.GetoptError('at least 3 <species> are required')
        try:
            nloci = int(argLst[2])
            if nloci <= 0:
                raise Exception
        except Exception:
            raise getopt.GetoptError('positive integer is required for <nloci>') from None
        try:
            p123 = float(argLst[3])
            if p123 < 0 or p123 > 1:
                raise Exception
        except Exception:
            raise getopt.GetoptError('<p123> should be float and between 0 and 1') from None

    except ValueError as e:
        ss.printToStderr('CommandLineError: ', e, '. Use option -h for help.', sep = '')
        return 2
    except getopt.GetoptError as e:
        ss.printToStderr('CommandLineError: ', e, '. Use option -h for help.', sep = '')
        return 2

    try:
        n1 = int(nloci * (1 - p123) / 6)
        n2 = nloci - n1 * 6
        configs = ['123#%d' % n2, '112#%d' % n1, '113#%d' % n1, '122#%d' % n1, '223#%d' % n1, '133#%d' % n1, '233#%d' % n1]
        indSpMap = ss.getIndSpMap(mapFile) if mapFile is not None else None

        for spNames in itertools.combinations(speciesNames, 3):
            outFile = '{}-{}-{}.phy'.format(*spNames)
            subsampled = sum(ss.subsample(seqFile, spNames, configs, indSpMap, greedy, outFile))
            print(f"In {outFile} subsampled {subsampled} {('loci', 'locus')[subsampled == 1]}.")

    except (ss.SubsampleError, FileNotFoundError) as e:
        ss.printToStderr(e.__class__.__name__, ': ', e, sep = '')
        return 2 if isinstance(e, ss.ConfigParsingError) else 1
    else:
        return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
