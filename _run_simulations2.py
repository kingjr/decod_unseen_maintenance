"""
These simulations aims at showing how different types of underlying sources may
reproduce the decoding results we observe: i.e.
1) A significant decoding of the target across different visibility levels,
2) A significant reduction of decoding accuracy after ~ 250 ms,
3) A significant correlation between decoding accuracy and the subjective,
   visibility of the stimulus after the stimulus onset,
4) A significant decoding of subjective visiblity (which actually necessarily
derives from 3.)
"""

import numpy as np
from jr.plot import pretty_decod, pretty_plot
from jr.gat import scorer_auc, scorer_spearman
from mne.decoding import GeneralizationAcrossTime
from mne import create_info, EpochsArray
import matplotlib.pyplot as plt
from conditions import analyses
from config import report

analysis = [ana for ana in analyses if ana['name'] == 'target_present'][0]

n_source = 5
n_chan = 32
n_time = 75
sfreq = 256.
tmin = -.050
snr = .5

times = np.arange(n_time) / sfreq + tmin


def simulate_model(sources, mixin, background, snr=.5, n_trial=100):
    """Run simulations :
    1. Takes source activations in two visibility conditions:
        dict(high=(n_sources * n_times), low=(n_sources * n_times))
    2. Target presence/absence is coded in y vector and corresponds to the
       reverse activation in source space.
    3. Takes a mixin matrix that project the data from source space to sensor
        space
    4. Generates multiple low and high visibility trials.
    5. Fit target presence (y) across all trials (both high and low visiblity),
    6. Score target presence separately for high and low visibility trials
    7. Fit and score target visibility (for simplicity reasons, we only have 2
       visibility conditions. Consequently, we will fit a logistic regression
       and not a ridge like the one used for in empirical part of the paper.)
    """
    n_source, n_chan = mixin.shape

    # add information
    X, y, visibility = list(), list(), list()
    for vis, source in sources.iteritems():
        n_source, n_time = source.shape
        # define present and absent in source space
        present = np.stack([source + background] * (n_trial // 2))
        absent = np.stack([background] * (n_trial // 2))
        source = np.vstack((present, absent))
        y_ = np.hstack((np.ones(n_trial // 2), -1 * np.ones(n_trial // 2)))

        # transform in sensor space
        sensor = np.dot(mixin.T, np.hstack((source)))
        sensor = np.reshape(sensor, [n_chan, -1, n_time]).transpose(1, 0, 2)

        # add sensor specific  noise
        sensor += np.random.randn(n_trial, n_chan, n_time) / snr
        X.append(sensor)
        y.append(y_)
        visibility.append(int(vis == 'high') * np.ones(n_trial))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    visibility = np.concatenate(visibility, axis=0)

    # shuffle trials
    idx = range(n_trial * 2)
    np.random.shuffle(idx)
    X, y, visibility = X[idx], y[idx], visibility[idx]

    # format to MNE epochs
    epochs = EpochsArray(X, create_info(n_chan, sfreq, 'mag'), tmin=times[0],
                         proj=False, baseline=None)

    gat = GeneralizationAcrossTime(clf=analysis['clf'], cv=8,
                                   scorer=scorer_auc, n_jobs=-1,
                                   score_mode='mean-sample-wise')

    gat.fit(epochs, y=y)
    y_pred = gat.predict(epochs)
    y_pred = y_pred[:, :, :, 0].transpose(2, 0, 1)

    score = list()
    for vis in range(2):
        # select all absent trials + present at a given visibility
        sel = np.unique(np.hstack((np.where(y == -1)[0],
                        np.where(visibility == vis)[0])))
        score_ = scorer_auc(y[sel], y_pred[sel], n_jobs=-1)
        score.append(score_)

    # correlation with visibility
    sel = np.where(y == 1)[0]
    corr_vis = scorer_spearman(visibility[sel], y_pred[sel], n_jobs=-1)

    # decode visibility
    sel = np.where(y == 1)[0]  # present trials only
    gat.fit(epochs[sel], y=visibility[sel])
    score_vis = gat.score(epochs[sel], y=visibility[sel])
    return np.array(score), np.squeeze(score_vis), np.squeeze(corr_vis)

# Architectures
t0 = 10
encoding = dict(high=np.zeros((n_source, n_time)),
                low=np.zeros((n_source, n_time)))
two_steps = dict(high=np.zeros((n_source, n_time)),
                 low=np.zeros((n_source, n_time)))
reactivation = dict(high=np.zeros((n_source, n_time)),
                    low=np.zeros((n_source, n_time)))
multisteps = dict(high=np.zeros((n_source, n_time)),
                  low=np.zeros((n_source, n_time)))
silent = dict(high=np.zeros((n_source, n_time)),
              low=np.zeros((n_source, n_time)))

# 1. visibility could be due to an encoding problem: the higher the response,
# the longer its maintenance
encoding['low'][0, t0:(t0+50)] = np.logspace(1.5, 0, 50) / 100. * 3
encoding['high'][0, t0:(t0+50)] = np.logspace(1, 0, 50) / 10. * 3

# 2. visibility could be due to a late stage maintenance
for vis in ['low', 'high']:
    two_steps[vis][0, t0:(t0+10)] = 1
two_steps['low'][1, t0+10:(t0+50)] = .25
two_steps['high'][1, t0+10:(t0+50)] = .5

# 3: two steps encoding but reactivation = aware
for vis in ['low', 'high']:
    reactivation[vis][0, t0:(t0+10)] = 1
    reactivation[vis][1, t0+10:(t0+20)] = .5
    reactivation[vis][2, t0+20:(t0+50)] = .25
reactivation['high'][0, t0+20:(t0+50)] = .25

# 4. Purely dynamical
for vis in ['low', 'high']:
    multisteps[vis][0, t0:(t0+10)] = 1
for ii in range(1, 5):
    multisteps['low'][ii, (t0 + 10*ii):(t0+((ii + 1)*10))] = .25
    multisteps['high'][ii, (t0 + 10*ii):(t0+((ii + 1)*10))] = .5


# 5. Silent
silent['low'][0, t0:(t0+10)] = 1
silent['low'][0, (t0+40):(t0+50)] = .5
silent['high'][0, t0:(t0+10)] = 1
silent['high'][0, (t0+40):(t0+50)] = 1

# empirical scenario: multisteps plus maintenance
sources = dict(encoding=encoding,
               two_steps=two_steps,
               reactivation=reactivation,
               multisteps=multisteps,
               silent=silent)

n_subject = 20
all_scores, all_scores_vis, all_corr_vis = dict(), dict(), dict()
for name, source in sources.iteritems():
    all_scores[name] = np.zeros((n_subject, 2, n_time, n_time))
    all_scores_vis[name] = np.zeros((n_subject, n_time, n_time))
    all_corr_vis[name] = np.zeros((n_subject, n_time, n_time))
    for subject in range(n_subject):
        # Define a subject specific background activity to mimmic mask, etc
        background = np.random.randn(n_source, n_time) / 10.
        mixin = np.random.randn(n_source, n_chan)
        score, score_vis, corr_vis = simulate_model(source, mixin, background)
        all_scores[name][subject] = score
        all_scores_vis[name][subject] = score_vis
        all_corr_vis[name][subject] = corr_vis

fig = plt.figure(figsize=[6, 4])
cmap_contrast = plt.get_cmap('hot_r')

fig_dec, axes_dec = plt.subplots(len(sources), 1, sharex=True, sharey=True)
fig_gat, axes_gat = plt.subplots(2, len(sources), figsize=[12, 4])
fig_src, axes_src = plt.subplots(2, len(sources), figsize=[12, 4])
fig_vis, axes_vis = plt.subplots(1, len(sources), figsize=[12, 2.5])
fig_vis_corr, axes_corr = plt.subplots(1, len(sources), figsize=[12, 2.5])
for ii, ((model, scores), scores_vis, corr_vis) in enumerate(zip(
        all_scores.iteritems(), all_scores_vis.itervalues(),
        all_corr_vis.itervalues())):
    for jj, score in enumerate(scores.transpose([1, 0, 2, 3])[[1, 0]]):
        # Source
        vis = 'high' if jj == 0 else 'low'
        linestyle = '--' if vis == 'low' else '-'
        cmap = plt.get_cmap('rainbow')
        n_source_ = sum(np.sum(sources[model][vis], axis=1) > 1)
        colors = cmap(np.linspace(0, 1., n_source_))
        for src, color in zip(sources[model][vis], colors):
            pretty_decod(src, ax=axes_src[jj, ii], fill=True, chance=0.,
                         sig=np.ones_like(src), color=color[:3], alpha=.5)
        axes_src[jj, ii].set_xlabel('')
        axes_src[jj, ii].set_ylabel('')
        axes_src[jj, ii].set_xticks([])
        axes_src[jj, ii].set_yticks([])

        # Gat
        axes_gat[jj, ii].matshow(score.mean(0), origin='lower',
                                 vmin=0, vmax=1., cmap='RdBu_r')
        axes_gat[jj, ii].set_xticks([])
        axes_gat[jj, ii].set_yticks([])
        pretty_plot(axes_gat[jj, ii])

        # Diagonal score
        diag = np.array([np.diag(s) for s in score])
        # sig = stats(diag - .5) < .05
        color = 'r' if vis == 'high' else 'b'
        pretty_decod(diag, chance=.5, fill=True, color=color,
                     ax=axes_dec[ii], alpha=.1)
        axes_dec[ii].set_xticks([])
        axes_dec[ii].set_yticks([])
        axes_dec[ii].set_xlabel('')

    # Gat visibility decoding
    axes_vis[ii].matshow(scores_vis.mean(0), origin='lower',
                         vmin=0, vmax=1., cmap='RdBu_r')
    axes_vis[ii].set_xticks([])
    axes_vis[ii].set_yticks([])
    pretty_plot(axes_vis[ii])

    # Gat visibility subtraction
    axes_corr[ii].matshow(np.mean(scores[:, 1] - scores[:, 0], axis=0),
                          origin='lower', vmin=-.25, vmax=.25, cmap='RdBu_r')
    axes_corr[ii].set_xticks([])
    axes_corr[ii].set_yticks([])
    pretty_plot(axes_corr[ii])


report.add_figs_to_section([fig_gat, fig_dec, fig_src, fig_vis, fig_vis_corr],
                           ['gat', 'decod', 'source', 'vis', 'corr_vis'],
                           'simulation')
report.save()
