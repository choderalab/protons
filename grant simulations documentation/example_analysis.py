import logging
import pandas as pd
import matplotlib as mpl
import os
import errno

try
    os.mkdir(plots)
except OSError as error
    if error.errno != errno.EEXIST
        raise error
    pass # plots directory already exists


mpl.use('Agg')  # to run on the cluster

from matplotlib import pyplot as plt
plt.style.use('ggplot')

# states_output.dat is a csv file. Column headers are residue names with numbers.
data = pd.read_table(states_output.dat, sep=',')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

for col in data
    plt.figure()
    # Totals per state, make bar plot
    vals = data[col].value_counts()
    logger.info(vals)
    vals.plot(kind='bar')
    plt.xlabel(State)
    plt.ylabel(Count)
    plt.title(col)
    plt.savefig(plotsbar-{}.pdf.format(col.strip()))
    plt.figure()

    # Time series of state data
    data[col].plot()
    plt.xlabel(Time ( 6  ps))
    plt.ylabel(State)
    plt.title(col)
    plt.savefig(plotstimeseries-{}.pdf.format(col.strip()))