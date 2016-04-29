import pandas as pd

with open('benchmark_cache.txt') as datafile:
    df = pd.read_csv(datafile)
    df['ratio'] = df[' Time per titration attempt (sec)']/df['# Time per timestep (sec)']
    open('summary_cache.txt', 'w').write(str(df.describe()))
    
