import pandas as pd

with open('benchmark.txt') as datafile:
    open('summary.txt', 'w').write(str(pd.read_csv(datafile).describe()))
    
