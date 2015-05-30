import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mne.io.pick import _picks_by_type as picks_by_type

from toolbox.utils import build_analysis
from utils import get_events
from config import (
    subjects,
    inputTypes,
    contrasts,
    chan_types,
)

from analyses_definition_univariate import format_analysis
contrasts = [ana for ana in contrasts if ana['name'] == 'targetAngle']
analysis = format_analysis(contrasts[0])

data_path = '/media/DATA/Pro/Projects/Paris/Orientation/Niccolo/data/'
meg_fname = data_path + 'preprocessed/ak_heavily_decimated.epo'
bhv_fname = data_path + '/ak130184_fixed.mat'
meg_fname = data_path + 'preprocessed/mc_heavily_decimated.epo'
bhv_fname = data_path + '/mc130295_fixed.mat'
events = get_events(bhv_fname)
with open(meg_fname, 'r') as f:
    epochs = pickle.load(f)
#
# mat = dict()
# mat['dat'] = np.array(epochs._data)
# mat['y'] = 1. * np.array(events['orientation_target'])
# mat_fname = data_path + '/heavily_decimated.mat'
# sio.savemat(mat_fname, mat)

# force separation of magnetometers and gradiometers
if 'meg' in [i['name'] for i in chan_types]:
    chan_types = [dict(name='mag'), dict(name='grad')] + \
                 [dict(name=i['name'])
                  for i in chan_types
                  if i['name'] != 'meg']

subject = subjects[0]
typ = inputTypes[-1]
fname_appendix = ''
fileformat = '.dat'

# # Make 6 evoked (average trials) for each orientation
# evokeds = list()
# angles_deg = np.linspace(15, 165, 6)
# for angle in angles_deg:
#     sel = np.where(events['orientation_target'] == angle)[0]
#     evoked = epochs[sel].average()
#     evokeds.append(evoked)
#
# # compute circular linear correlation
# from toolbox.utils import circular_linear_correlation
# n_angles = len(evokeds)
# angles = np.linspace(0, 2 * np.pi - (2 * np.pi) / n_angles, n_angles)
# x = np.array([ev.data.flatten() for ev in evokeds])
# rho, _ = circular_linear_correlation(np.tile(angles, [x.shape[1], 1]).T, x)
#
#
# sel = np.where(events['present'])[0]
# angles = events['orientation_target'][sel]
# n_trials, n_chans, n_times = epochs[sel]._data.shape
# x = epochs[sel]._data.reshape(n_trials, n_chans * n_times)
# rho, _ = circular_linear_correlation(np.tile(angles, [x.shape[1], 1]).T, x)
#
# n_chans, n_times = evokeds[0].data.shape
# evoked.data = rho.reshape(n_chans, n_times)
# plt.matshow(evoked.data, aspect='auto')
# plt.colorbar()
# plt.show()

coef, evokeds = build_analysis(analysis['conditions'], epochs, events,
                               operator=analysis['operator'])

# Prepare plot delta (subtraction, or regression)
fig1, ax1 = plt.subplots(1, len(chan_types))
if type(ax1) not in [list, np.ndarray]:
    ax1 = [ax1]
# Prepare plot all conditions at top level of analysis
fig2, ax2 = plt.subplots(len(evokeds['coef']), len(chan_types))
ax2 = np.reshape(ax2, len(evokeds['coef']) * len(chan_types))

# Loop across channels
for ch, chan_type in enumerate(chan_types):
    # Select specific types of sensor
    info = coef.info
    picks = [i for k, p in picks_by_type(info)
             for i in p if k in chan_type['name']]
    # ---------------------------------------------------------
    # Plot coef (subtraction, or regression)
    # adjust color scale
    mM = np.percentile(np.abs(coef.data[picks, :]), 99.)

    # plot mean sensors x time
    ax1[ch].imshow(coef.data[picks, :], vmin=-mM, vmax=mM,
                   interpolation='none', aspect='auto',
                   cmap='RdBu_r', extent=[min(coef.times),
                   max(coef.times), 0, len(picks)])
    # add t0
    ax1[ch].plot([0, 0], [0, len(picks)], color='black')
    ax1[ch].set_title(chan_type['name'] + ': ' + coef.comment)
    ax1[ch].set_xlabel('Time')
    ax1[ch].set_adjustable('box-forced')

    # ---------------------------------------------------------
    # Plot all conditions at top level of analysis
    # XXX only works for +:- data
    mM = np.median([np.percentile(abs(e.data[picks, :]), 80.)
                    for e in evokeds['coef']])

    for e, evoked in enumerate(evokeds['coef']):
        ax_ind = e * len(chan_types) + ch
        ax2[ax_ind].imshow(evoked.data[picks, :], vmin=-mM,
                           vmax=mM, interpolation='none',
                           aspect='auto', cmap='RdBu_r',
                           extent=[min(coef.times),
                           max(evoked.times), 0, len(picks)])
        ax2[ax_ind].plot([0, 0], [0, len(picks)], color='k')
        ax2[ax_ind].set_title(chan_type['name'] + ': ' +
                              evoked.comment)
        ax2[ax_ind].set_xlabel('Time')
        ax2[ax_ind].set_adjustable('box-forced')
plt.show()
