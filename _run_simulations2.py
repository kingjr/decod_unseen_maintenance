import numpy as np
from jr.plot import pretty_decod, pretty_plot
from jr.gat import subscore
from mne.decoding import GeneralizationAcrossTime
from mne import create_info, EpochsArray
import matplotlib.pyplot as plt
from base import stats
from conditions import analyses
from config import report

analysis = [ana for ana in analyses if ana['name'] == 'target_present'][0]

n_time = 150
n_source = 10
sfreq = 512.
tmin = -.050
times = np.arange(n_time) / sfreq + tmin
decim = 50

n_chan = 32


def simulate_model(sources, mixin):
    snr = .2
    n_trial = 100
    n_source, n_chan = mixin.shape

    # add information
    X, y, visibility = list(), list(), list()
    for vis, source in sources.iteritems():
        source = np.stack([source] * n_trial)
        n_trial, n_source, n_time = source.shape
        y_ = np.random.randint(0, 2, n_trial)
        source[y_ == 0, ...] *= -1

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

    # format to MNE epochs
    epochs = EpochsArray(X, create_info(n_chan, sfreq, 'mag'), tmin=times[0])
    gat = GeneralizationAcrossTime(clf=analysis['clf'], cv=8,
                                   scorer=analysis['scorer'], n_jobs=-1,
                                   score_mode='mean-sample-wise')

    gat.fit(epochs, y=y)
    gat.predict(epochs)
    score = list()
    for vis in range(2):
        sel = np.where(visibility == vis)[0]
        score.append(subscore(gat, y=y[sel], sel=sel))
    return np.array(score)


# WM architecture
t0 = 10
# 1: one step encoding and maintenance
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
encoding['low'][0, t0:(t0+100)] = np.logspace(2., 0, 100) / 100. * 3
encoding['high'][0, t0:(t0+100)] = np.logspace(1, 0, 100) / 10. * 3

# 2. visibility could be due to a late stage maintenance
two_steps['low'][0, t0:(t0+20)] = 1
two_steps['low'][1, t0+20:(t0+100)] = .5
two_steps['high'][0, t0:(t0+20)] = 1
two_steps['high'][1, t0+20:(t0+100)] = 1

# 3: two steps encoding but reactivation = aware
reactivation['low'][0, t0:(t0+20)] = 1
reactivation['low'][1, t0+20:(t0+40)] = .75
reactivation['low'][2, t0+40:(t0+100)] = .5
reactivation['high'][0, t0:(t0+20)] = 1
reactivation['high'][1, t0+20:(t0+40)] = .5
reactivation['high'][2, t0+40:(t0+100)] = .5
reactivation['high'][0, t0+40:(t0+100)] = .5

# 4. Purely dynamical
for ii in range(5):
    multisteps['low'][ii, (t0 + 20*ii):(t0+((ii + 1)*20))] = .5
    multisteps['high'][ii, (t0 + 20*ii):(t0+((ii + 1)*20))] = 1

# 5. Silent
silent['low'][0, t0:(t0+20)] = 1
silent['low'][0, (t0+80):(t0+100)] = .5
silent['high'][0, t0:(t0+20)] = 1
silent['high'][0, (t0+80):(t0+100)] = 1

# empirical scenario: multisteps plus maintenance
sources = dict(encoding=encoding,
               two_steps=two_steps,
               reactivation=reactivation,
               multisteps=multisteps,
               silent=silent)

n_subject = 20
all_scores = dict()
for name, source in sources.iteritems():
    all_scores[name] = dict()
    all_scores[name] = np.zeros((n_subject, 2, n_time, n_time))
    for subject in range(n_subject):
        mixin = np.random.randn(n_source, n_chan)
        all_scores[name][subject] = simulate_model(source, mixin)

fig = plt.figure(figsize=[6, 4])
cmap_contrast = plt.get_cmap('hot_r')

fig_dec, axes_dec = plt.subplots(len(sources), 1, sharex=True, sharey=True)
fig_gat, axes_gat = plt.subplots(2, len(sources), figsize=[12, 4])
fig_src, axes_src = plt.subplots(2, len(sources), figsize=[12, 4])
for ii, (model, scores) in enumerate(all_scores.iteritems()):
    for jj, (vis, score) in enumerate(scores.iteritems()):
        # Source
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
        # diagonal score
        diag = np.array([np.diag(s) for s in score])
        sig = stats(diag - .5) < .05
        color = 'r' if vis == 'high' else 'b'
        pretty_decod(diag, chance=.5, sig=sig, fill=True, color=color,
                     ax=axes_dec[ii])
        axes_dec[ii].set_xticks([])
        axes_dec[ii].set_yticks([])
        axes_dec[ii].set_xlabel('')

report.add_figs_to_section([fig_gat, fig_dec, fig_src],
                           ['gat', 'decod', 'source'], 'simulation')
report.save()
