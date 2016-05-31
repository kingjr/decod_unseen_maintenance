"""Prepare small pickle files that will allow users to interactively play with
the decoding results online"""

import numpy as np
from config import load
from conditions import analyses
import pickle

# ERF
all_data = []
downsample = slice(None, None, 10)
for analysis in analyses:
    pass
    # load stats
    evoked, data, p_values, sig, analysis = load(
        'evoked', analysis=('stats_' + analysis['name']))
    # only keep gradiometers 1 (They have already been combined)
    all_data.append(dict(name=analysis['name'], data=evoked.data))
info = dict(times=evoked.times, ch_names=evoked.ch_names,
            sfreq=evoked.info['sfreq'], ch_type=['grad', 'grad', 'mag'] * 102)
with open('data/results_evoked.pkl', 'wb') as f:
    pickle.dump([all_data, info], f)


# <<<--- FIXME Need manual import with previous MNE version: see issue #2899
import pickle
from mne import EvokedArray
from mne.io.meas_info import create_info
with open('data/results_evoked.pkl', 'rb') as f:
    all_data, info = pickle.load(f)
tmin = info['times'][0]
info = create_info(info['ch_names'], info['sfreq'], info['ch_type'])
for ii, data in enumerate(all_data):
    data['evoked'] = EvokedArray(data['data'], info, tmin)
    data.pop('data')
    all_data[ii] = data
with open('data/results_evoked.pkl', 'wb') as f:
    pickle.dump(all_data, f)

with open('data/results_evoked.pkl', 'wb') as f:
    pickle.dump(all_data, f)
# --->>>


# Decoding
all_data = []
for analysis in analyses:
    print(analysis['name'])

    # Load
    out = load('score', subject='fsaverage',
               analysis=('stats_' + analysis['name']))
    scores = np.array(out['scores'])
    p_values = out['p_values']

    # Downsample
    data = {key: analysis[key] for key in ['name', 'chance', 'color']}
    downsample = slice(None, None, 2)
    data['times'] = out['times'][downsample] / 1e3  # in secs
    data['scores'] = scores[:, downsample, downsample]
    data['pval'] = p_values[downsample, downsample]
    data['pval_diag'] = np.squeeze(out['p_values_diag'])[downsample]
    all_data.append(data)

# Save
with open('data/results_decoding.pkl', 'wb') as f:
    pickle.dump(all_data, f)
