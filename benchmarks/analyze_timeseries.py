import pymbar
import numpy as np


def autocorrelation_analysis(txt):

    data = np.genfromtxt(txt, dtype=float, delimiter=',', skip_header=1)
    headers = open(txt).readline()[1:].strip().split(sep=',')
    # Store individual values
    x = list()

    if data.T.ndim > 1:
        for arr in data.T:
            try:
                x.append(pymbar.timeseries.integratedAutocorrelationTime(arr))
            except pymbar.utils.ParameterError:
                x.append(np.nan)
    else:
        try:
            x.append(pymbar.timeseries.integratedAutocorrelationTime(data))
        except pymbar.utils.ParameterError:
            x.append(np.nan)

    x = list(map(float, x))
    named_values = dict(zip(headers, x))

    open('autocorrelation.txt', 'w').write(repr(named_values))

txt = 'states.txt'
autocorrelation_analysis(txt)
