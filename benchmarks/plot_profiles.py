from os.path import basename
from glob import glob
import pandas as pd
import sys
import seaborn as sns
from seaborn import plt
direct_ory='Results/dhfr-explicit/01-25-2016'
profiles = dict()
frames = list()
for dirname in glob('{}/*'.format(direct_ory), recursive=True):
    nupdates, ntimesteps, nncmcsteps, nattempts_p_update = basename(dirname).split(sep='_')
    txt = '{}/bench_pstats.txt'.format(dirname)
    d = pd.read_table(txt, skiprows=6,engine='python', sep=r'\s+(?![^{]*})') # https://regex101.com/r/tG7kC0/5
    d['Updates'] = int(nupdates)
    d['Time steps'] = int(ntimesteps)
    d['NCMC steps'] = int(nncmcsteps)
    d['Attempts/Update'] = int(nattempts_p_update)
    d = d.sort_values(by="percall.1", ascending=False).head(25)

    frames.append(d)

# Convenient variable
function_colname = 'filename:lineno(function)'




df = pd.concat(frames)
df['ncalls'] = df['ncalls'].apply(lambda x: x.split('/')[0])
df['ncalls'] = df['ncalls'].astype(int)

df = df[df[function_colname].str.contains("constph|openmm")]


df = df.loc[df['Attempts/Update'] == 1]
# df = df.loc[df[function_colname] == "constph.py:1035(update)"]




# Initialize a grid of plots with an Axes for each walk


# sns.swarmplot(data=df, x='NCMC steps', y='percall', hue=function_colname, ax=ax)

for fname, fgroup in df.groupby(function_colname):
    f, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(fname)
    ax.scatter(fgroup["NCMC steps"], fgroup["percall.1"])

# sns.factorplot(data=df, x='NCMC steps', y='percall.1', hue=function_colname,ax=ax)

plt.show()