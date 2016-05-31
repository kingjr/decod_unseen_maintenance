""""Search light across sensors"""
import numpy as np
from itertools import product
from mne import pick_types
from jr.gat import SensorDecoding
from .config import paths, subjects, tois, load
from .conditions import analyses

all_scores = list()
for s, subject in enumerate(subjects):
    # Read data
    epochs = load('epochs', subject=subject)
    events = load('behavior', subject=subject)

    # decode on combination of gradiometers
    grads = pick_types(epochs.info, meg='grad')
    ch_groups = np.vstack((grads[::2], grads[1::2])).T

    # preprocess data for memory issue
    scores_ = dict()
    for toi, analysis in product(tois, analyses):
        print subject, toi, analysis
        epochs_ = epochs.crop(toi[0], toi[1], copy=True)

        # subselect trials
        query, condition = analysis['query'], analysis['condition']
        sel = range(len(events)) if query is None \
            else events.query(query).index
        sel = [ii for ii in sel if ~np.isnan(events[condition][sel][ii])]
        y = np.array(events[condition], dtype=np.float32)

        score = None
        if len(sel) != 0:
            sd = SensorDecoding(clf=analysis['clf'], ch_groups=ch_groups,
                                cv=analysis['cv'],
                                scorer=analysis['scorer'],
                                n_jobs=-1)
            sd.fit(epochs_[sel], y=y[sel])
            score = sd.score(epochs_[sel], y=y[sel])
        scores_['%s_%.2f-%.2f' % (analysis['name'], toi[0], toi[1])] = score
    all_scores.append(scores_)

import pickle
fname = paths('score', analysis='all_sensor_decoding')
with open(fname, 'wb') as f:
    pickle.dump(all_scores, f)

# plot
import matplotlib.pyplot as plt
from jr.plot import share_clim
from .config import report
from .base import stats
from mne.channels import read_ch_connectivity
connectivity, ch_names = read_ch_connectivity('neuromag306mag')
evoked = epochs.average()
evoked.pick_types(meg='mag', copy=False)
for analysis in analyses:
    fig, axes = plt.subplots(1, len(tois))
    m, M = np.inf, -np.inf
    for toi, ax in zip(tois, axes):
        # gather data
        scores = [score_['%s_%.2f-%.2f' % (analysis['name'], toi[0], toi[1])]
                  for score_ in all_scores]
        # compute stats
        p_val = stats(np.array(scores)[:, :, None] - analysis['chance'],
                      connectivity)
        scores = np.mean(scores, axis=0)
        m = np.min(scores) if np.min(scores) < m else m
        M = np.max(scores) if np.max(scores) > M else M
        # plot
        evoked.data[:, :] = scores[:, np.newaxis]
        evoked.plot_topomap(
            times=[evoked.times[1]], axes=[ax], show=False, cmap='RdBu_r',
            scale=dict(mag=1), colorbar=False, contours=False, sensors=False,
            mask=np.tile(p_val < .05, [len(evoked.times), 1]).T,
            mask_params=dict(marker='*', markerfacecolor='k',
                             markeredgecolor='k', linewidth=0, markersize=6))
        ax.set_title(str(toi))
    M = M - analysis['chance']
    m = m - analysis['chance']
    m = -M if -m < M else m
    M = -m if -M > m else M
    share_clim(axes, [m + analysis['chance'], M + analysis['chance']])
    report.add_figs_to_section([fig], [analysis['name']], 'decod')
report.save()
