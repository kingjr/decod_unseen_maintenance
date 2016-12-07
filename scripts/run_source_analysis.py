# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# Licence: BSD 3-clause

"""Run single-trial mass-univariate analyses in source space for each subject
separately"""

import numpy as np
from mne.minimum_norm import apply_inverse, apply_inverse_epochs

from conditions import analyses
from config import load, save, bad_mri, subjects_id
from base import nested_analysis

# params
inv_params = dict(lambda2=1.0 / (2 ** 3.0),
                  method='dSPM',
                  pick_ori='normal',
                  verbose=False)

for meg_subject, subject in zip(range(1, 21), subjects_id):
    # load single subject effects (across trials)
    if subject in bad_mri:
        continue
    epochs = load('epochs_decim', subject=meg_subject, preload=True)
    events = load('behavior', subject=meg_subject)
    epochs.apply_baseline((None, 0))
    epochs.pick_types(meg=True, eeg=False, eog=False)

    # Setup source data container
    evoked = epochs.average()
    inv = load('inv', subject=meg_subject)
    stc = apply_inverse(evoked, inv, **inv_params)

    # run each analysis within subject
    for analysis in analyses:
        # source transforming should be applied as early as possible,
        # but here I'm struggling on memory
        coefs = list()
        n_chunk = 20
        for time in np.array_split(epochs.times, n_chunk):
            stcs = apply_inverse_epochs(epochs.copy().crop(time[0], time[-1]),
                                        inv, **inv_params)
            stcs_data = np.array([ii.data for ii in stcs])

            # then we applied the same function as for the sensor analysis
            # FIXME this nested_analysis is here an overkill since we only
            # 1 level analysis
            coef, sub = nested_analysis(
                stcs_data, events, analysis['condition'],
                function=analysis.get('erf_function', None),
                query=analysis.get('query', None),
                single_trial=analysis.get('single_trial', False),
                y=analysis.get('y', None),
                n_jobs=-1)
            coefs.append(coef)
        stc._data = np.hstack(coefs)

        # Save all_evokeds
        save([stc, sub, analysis], 'evoked_source', subject=meg_subject,
             analysis=analysis['name'], overwrite=True)

        # Clean memory
        for ii in stcs:
            ii._data = None
        del stcs, stcs_data, coef, sub
    epochs._data = None
    del epochs
