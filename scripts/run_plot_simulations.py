# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# Licence: BSD 3-clause

"""
These simulations aim at showing how different types of underlying sources may
reproduce the 3 main decoding results we observed: i.e.
1) A significant decoding of the target across different visibility levels,
2) A significant reduction of decoding accuracy after ~ 250 ms,
3) A significant correlation between decoding accuracy and the subjective,
   visibility of the stimulus after the stimulus onset)

Used to generate Figure 5.
"""
from collections import OrderedDict as dict
import numpy as np
from jr.plot import pretty_plot
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

    # Temporal generalization pipeline
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
t0 = 10  # onset of the first evoked response
init = lambda: dict(high=np.zeros((n_source, n_time)),
                    low=np.zeros((n_source, n_time)))


# single stage:
# visibility could be due to an single problem: the late_maintain the
# response, the longer its maintain
single = init()
for vis in ['low', 'high']:
    single[vis][0, t0:(t0+10)] = 1.
single['low'][0, (t0+10):60] = .25
single['high'][0, (t0+10):60] = .5

# Early maintain:
early_maintain = init()
# visibility depends on the amplitude of the late stage
for vis in ['low', 'high']:
    early_maintain[vis][0, t0:(t0+10)] = 1.
    early_maintain[vis][0, (t0+10):60] = .25
early_maintain['low'][1, (t0+10):(t0+10)] = 0.
early_maintain['high'][1, (t0+10):60] = .5

# Re-entry:
# visibility depends on the amplitude a single early stage
reentry = init()
for vis in ['low', 'high']:
    reentry[vis][0, t0:(t0+10)] = 1.
    reentry[vis][1, (t0+10):60] = .25
reentry['low'][0, (t0+10):60] = 0.
reentry['high'][0, (t0+10):60] = .5

# Late-maintain
# visibility depends on the amplitude of a single late stage
late_maintain = init()
for vis in ['low', 'high']:
    late_maintain[vis][0, t0:(t0+10)] = 1.
late_maintain['low'][1, (t0+10):(t0+50)] = .25
late_maintain['high'][1, (t0+10):(t0+50)] = .5

# Dynamic
# visibility depends on the amplitude of the many late stages
dynamic_amplitude = init()
for vis in ['low', 'high']:
    dynamic_amplitude[vis][0, t0:(t0+10)] = 1.
for stage in range(1, 5):
    start = t0 + 10 * stage
    dynamic_amplitude['low'][stage, start:(start+10)] = .25
    dynamic_amplitude['high'][stage, start:(start+10)] = .5

# Dynamic maintenance
# visibility depends on the duration of the many late stages
dynamic_maintain = init()
for vis in ['low', 'high']:
    dynamic_maintain[vis][0, t0:(t0+10)] = 1.
for stage in range(1, 5):
    start = t0 + 10 * stage
    # low visibility = transient response
    dynamic_maintain['low'][stage, start:(start + 10)] = .42
    # high visibility = sustained response
    dynamic_maintain['high'][stage, start:60] = .42

# Hybrid model illustrate MEG results
hybrid = init()
for vis in ['low', 'high']:
    hybrid[vis][0, t0:(t0+5)] = 1.
    # Partial early reversal independent of visibility
    hybrid[vis][0, (t0+5):(t0+10)] = -.5
    # Reactivation independent of visibility
    hybrid[vis][0, (t0+15):(t0+25)] = .25
for ii in range(1, 5):
    start = (t0 - 4 + 9*ii)
    # meta stable in low visibility
    hybrid['low'][ii, start:(start + 20)] = .25
    # longer metastability and stronger amplitude in high visibility
    hybrid['high'][ii, start:(start + 30)] = .42
# Remove signal after 60 to get similar decoding
for vis in ['low', 'high']:
    hybrid[vis][:, 60:] = 0.

# empirical scenario: dynamic_amplitude plus maintain
sources = (
    ('single', single),
    ('reentry', reentry),
    ('late_maintain', late_maintain),
    ('early_maintain', early_maintain),
    ('dynamic_amplitude', dynamic_amplitude),
    ('dynamic_maintain', dynamic_maintain),
    ('hybrid', hybrid))

n_subject = 20
all_scores, all_scores_vis, all_corr_vis = dict(), dict(), dict()
for name, source in sources:
    # init results
    all_scores[name] = np.zeros((n_subject, 2, n_time, n_time))
    all_scores_vis[name] = np.zeros((n_subject, n_time, n_time))
    all_corr_vis[name] = np.zeros((n_subject, n_time, n_time))
    for subject in range(n_subject):
        # Define a subject specific background activity to mimmic mask, etc
        background = np.random.randn(n_source, n_time) / 10.
        # Define subject specific foward model
        mixin = np.random.randn(n_source, n_chan)
        # simulate
        score, score_vis, corr_vis = simulate_model(source, mixin, background)
        # store
        all_scores[name][subject] = score
        all_scores_vis[name][subject] = score_vis
        all_corr_vis[name][subject] = corr_vis

# Plot
report._setup_provenance()
for ii, ((model, scores), scores_vis, corr_vis) in enumerate(zip(
        all_scores.iteritems(), all_scores_vis.itervalues(),
        all_corr_vis.itervalues())):
    fig, axes = plt.subplots(1, 5, figsize=[20, 3])
    axes = dict(source=axes[0], decod=axes[1], high=axes[2], low=axes[3],
                vis=axes[4])
    # order scores by visibility and present seen first
    scores_ = dict(high=scores[:, 1], low=scores[:, 0])
    colors = dict(high='r', low='b')

    # get model
    source = dict(sources)[model]
    for vis in ('low', 'high'):
        score = scores_[vis]
        # plot sources
        n_source_ = sum(np.sum(source[vis], axis=1) > 1)
        y_shift = .2 * float(vis == 'high') - 1.
        for ii, src in enumerate(source[vis]):
            if sum(src) == 0:
                continue
            y_shift += 1.4
            zorder = - y_shift + .5 * float(vis == 'low')
            times = np.linspace(0, 1, len(src)) + .05 * int(vis == 'low')
            axes['source'].fill_between(
                times, y_shift, y_shift + src, color=colors[vis],
                alpha=1., linewidth=0., zorder=zorder)
            axes['source'].plot(
                times, y_shift + src, color='k',
                zorder=zorder)

        axes['source'].set_xticks([])
        axes['source'].set_yticks([])
        axes['source'].set_ylim(-1, y_shift + 1.5)
        for side in ('left', 'right', 'top', 'bottom'):
            axes['source'].spines[side].set_visible(False)

        # Diagonal score
        diag = np.array([np.diag(s) for s in score])
        zorder = float(vis == 'low')
        y_shift = .1 * float(vis == 'high')
        times = np.linspace(0, 2., len(diag.mean(0))) + .1 * int(vis == 'low')
        axes['decod'].fill_between(
            times, .5 + y_shift, diag.mean(0) + y_shift, color=colors[vis],
            alpha=1., linewidth=0., zorder=zorder)
        axes['decod'].plot(times, diag.mean(0) + y_shift, color='k',
                           zorder=zorder)
        axes['decod'].set_xticks([])
        axes['decod'].set_yticks([])
        axes['decod'].set_aspect('equal')
        for side in ('left', 'right', 'top', 'bottom'):
            axes['decod'].spines[side].set_visible(False)
        axes['decod'].set_ylim(-.1, 1.1)

        # Temporal generalization scores
        axes[vis].matshow(score.mean(0), origin='lower',
                          vmin=0, vmax=1., cmap='RdBu_r')
        axes[vis].set_xticks([])
        axes[vis].set_yticks([])
        pretty_plot(axes[vis])

    # TG subtraction: high vis - low vis
    axes['vis'].matshow(np.mean(scores[:, 1] - scores[:, 0], axis=0),
                        origin='lower', vmin=-.25, vmax=.25, cmap='RdBu_r')
    axes['vis'].set_xticks([])
    axes['vis'].set_yticks([])
    pretty_plot(axes['vis'])
    report.add_figs_to_section([fig], [model], model)
report.save()
