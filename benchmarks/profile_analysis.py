import pstats
from glob import glob
import gprof2dot
import subprocess
import shlex

direct_ory='Results/dhfr-explicit/01-25-2016'
profiles = dict()
for dirname in glob('{}/*'.format(direct_ory), recursive=True):
    prof_file = '{}/benchmark.prof'.format(dirname)
    outputfile = '{}/bench.dot'.format(dirname)
    image = '{}/bench.png'.format(dirname)
    txt = '{}/bench_pstats.txt'.format(dirname)
    ps = pstats.Stats(prof_file, stream=open(txt, 'w'))
    ps.strip_dirs()
    ps.sort_stats('time')

    ps.print_stats()

    profile = gprof2dot.PstatsParser(prof_file).parse()
    profile.prune(0.5/100.0, 0.1/100.0)

    dot = gprof2dot.DotWriter(open(outputfile, 'wt', encoding='UTF-8'))
    dot.strip = False
    dot.wrap = False
    dot.graph(profile, gprof2dot.themes["color"])







