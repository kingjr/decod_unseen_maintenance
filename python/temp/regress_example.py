# Author: Jean-Rémi King <jeanremi.king@gmail.com>
#
# BSD License

import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame
from mne.io.meas_info import create_info
from mne.epochs import EpochsArray

from python.toolbox.utils import build_contrast


# GENERATE SYNTHETIC DATA:
# 10 categories observed from 5*5 sensors across 15 time points and 100 trials
n_trial = 100
n_chan = 25
n_time = 15
n_cat = 10
categories = np.linspace(-1, 1, n_cat)

# Template topography for each angle
X0 = np.linspace(0, 2, np.sqrt(n_chan)) - 1
topos = list()
fig, ax = plt.subplots(ncols=n_cat)
for c, category in enumerate(categories):
    Xm, Ym = np.meshgrid(X0, X0)
    Xm += category
    Ym += category
    topo = np.exp(-((Xm ** 2) + (Ym ** 2)))
    topos.append(topo)
    ax[c].matshow(topo)
    ax[c].set_title('Topo %s' % c)
    ax[c].axis('off')
plt.show()

# Add noisy topo to each trial from n_time/2, and shuffle topo at each trial to
# simulate a different underlying generator.
snr = 10
data = np.random.randn(n_trial, n_chan, n_time) / snr
y = np.arange(n_trial) % len(categories)
for t in range(n_time / 2, n_time):
    np.random.shuffle(topos)
    for trial in range(n_trial):
        topo = topos[y[trial]].flatten()
        data[trial, :, t] += topo

# Export data into mne structure
time = range(n_time)
chan_names = ['meg' + str(i) for i in range(n_chan)]
chan_types = ['grad'] * n_chan
info = create_info(chan_names, 1, chan_types)
events = np.c_[np.cumsum(np.ones(n_trial)), np.zeros(n_trial),
               np.zeros(n_trial)]
epochs = EpochsArray(data, info, events)

# Create dataFrame for events to mimmic what we did in the pipeline
events = DataFrame([dict(category=i) for i in y])

# Define contrast in the same way we did in the pipeline
# XXX Gabriela, note that with the following syntax, you may need to redefine
# the contrasts in config.py! Let me know if you run into troubles.

from python.toolbox.utils import evoked_spearman
my_regression = dict(operator=evoked_spearman,
                     conditions=[dict(name='cat' + str(i),
                                      include=dict(category=i))
                                 for i in range(n_cat)])

# Instead of evoked_spearman, you can use utils.evoked_subtract,
# utils.evoked_weighted_subtract etc. Niccolo, you should thus make a small
# function that computes a circular linear correlation for each channel and
# time point.

# By default, if operator=None, it will apply a subtraction if there are 2
# categories, and a linear regression if there are more.

# Note that you can directly pass a list of conditions (e.g. my_regression =
# [condition1, condition2]) instead of a dictionary containing a 'conditions'
# key.

# Generate contrast
beta, evokeds = build_contrast(my_regression, epochs, events)

# Plot topo of regression coefficients at each time point
fig, ax = plt.subplots(ncols=n_time)
for t in range(n_time):
    ax[t].matshow(np.reshape(beta[:, t], (np.sqrt(n_chan), np.sqrt(n_chan))),
                  vmin=-2, vmax=2)
    ax[t].set_title('%s ms' % t)
    ax[t].axis('off')
plt.show()
