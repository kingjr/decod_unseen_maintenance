import pickle
import numpy as np
import matplotlib.pyplot as plt
from jr.plot import plot_tuning, plot_sem, bar_sem, pretty_decod
from jr.stats import circ_tuning, circ_mean, repeated_spearman
from mne.stats import spatio_temporal_cluster_1samp_test as stats
from scripts.config import paths, subjects, subscores, report, analyses
analyses = [analysis for analysis in analyses if analysis['name'] in
            ['target_circAngle', 'probe_circAngle']]

tois = [(-.100, 0.050), (.100, .250), (.250, .800), (.900, 1.050)]


def get_predict(gat, sel=None, toi=None, mean=True):
    from jr.gat import get_diagonal_ypred
    # select diagonal
    y_pred = np.squeeze(get_diagonal_ypred(gat)).T % (2 * np.pi)
    # Select trials
    sel = range(len(y_pred)) if sel is None else sel
    y_pred = y_pred[sel, :]
    # select TOI
    times = np.array(gat.train_times_['times'])
    toi = times[[0, -1]] if toi is None else toi
    toi_ = np.where((times >= toi[0]) & (times < toi[1]))[0]
    y_pred = y_pred[:, toi_]
    # mean across time point
    if mean:
        y_pred = circ_mean(y_pred, axis=1)[:, None]
    return y_pred


def get_predict_error(gat, sel=None, toi=None, mean=True):
    y_pred = get_predict(gat, sel=sel, toi=toi, mean=mean)
    # error is diff modulo pi centered on 0
    y_true = np.tile(gat.y_true_, [np.shape(y_pred)[1], 1]).T
    y_error = (y_pred - y_true + np.pi) % (2 * np.pi) - np.pi
    return y_error

# Gather data
for analysis in ['target_circAngle', 'probe_circAngle']:
    results = dict(diagonal=list(), angle_pred=list(), toi=list(),
                   subscore=list())
    for s, subject in enumerate(subjects):
        print s
        fname = paths('decod', subject=subject, analysis=analysis)
        with open(fname, 'rb') as f:
            gat, _, events_sel, events = pickle.load(f)
        times = gat.train_times_['times']
        n_bins = 24

        def mean_acc(y_error, axis=None):
            # range between -pi and pi just in case not done already
            y_error = y_error % (2 * np.pi)
            y_error = (y_error + np.pi) % (2 * np.pi) - np.pi
            # random error = np.pi/2, thus:
            return np.pi / 2 - np.mean(np.abs(y_error), axis=axis)

        # Mean error across trial on the diagonal
        y_error = mean_acc(get_predict_error(gat, mean=False), axis=0)
        results['diagonal'].append(y_error)

        # Mean prediction for each angle
        results_ = list()
        for angle in np.unique(gat.y_true_):
            y_pred = get_predict(gat, sel=np.where(gat.y_true_ == angle)[0])
            probas, bins = circ_tuning(y_pred, n=n_bins)
            results_.append(probas)
        results['angle_pred'].append(results_)

        # Mean tuning error per toi
        results_ = list()
        for toi in tois:
            probas, bins = circ_tuning(get_predict_error(gat, toi=toi),
                                       n=n_bins)
            results_.append(probas)
        results['toi'].append(results_)

        # Mean y_error per toi per vis
        results_ = dict()
        y_error = get_predict_error(gat, mean=False)
        for subanalysis in subscores:
            results_[subanalysis[0] + '_toi'] = list()
            # subselect events (e.g. seen vs unseen)
            subevents = events.iloc[events_sel].reset_index()
            subsel = subevents.query(subanalysis[1]).index
            # add nan if no trial matches subconditions
            if len(subsel) == 0:
                results_[subanalysis[0]] = np.nan * np.zeros(y_error.shape[1])
                for toi in tois:
                    results_[subanalysis[0] + '_toi'].append(np.nan)
                continue
            # dynamics of mean error
            results_[subanalysis[0]] = mean_acc(y_error[subsel, :], axis=0)
            # mean error per toi
            for toi in tois:
                # mean error across time
                toi_ = np.where((times >= toi[0]) & (times < toi[1]))[0]
                y_error_toi = circ_mean(y_error[:, toi_], axis=1)
                y_error_toi = mean_acc(y_error_toi[subsel])
                results_[subanalysis[0] + '_toi'].append(y_error_toi)
        results['subscore'].append(results_)
    results['times'] = times
    results['bins'] = bins
    fname = paths('score', subject='fsaverage', analysis=analysis + '-tuning')
    with open(fname, 'wb') as f:
        pickle.dump(results, f)


# Plot
def plot_tuning_(data, ax, color, polar=True):
    shift = None if polar is True else np.pi
    plot_tuning(data, ax=ax, color=color, polar=polar, half=polar, shift=shift)
    plot_tuning(data, ax=ax, color='k', polar=polar, half=polar, alpha=0,
                shift=shift)
    if polar:
        ax.set_xticks([0, np.pi])
        ax.set_xticklabels([0, '$\pi$'])
    else:
        ax.set_xticks([-np.pi, np.pi])
        ax.set_xticklabels(['$-\pi/2$', '$\pi/2$'])


for analysis in analyses:
    fname = paths('score', subject='fsaverage',
                  analysis=analysis['name'] + '-tuning')
    with open(fname, 'rb') as f:
        results = pickle.load(f)
    times = results['times']

    # Plot the prediction per angle
    t = -1 if analysis['name'] == 'probe_circAngle' else 0
    fig, axes = plt.subplots(1, 6, subplot_kw=dict(polar=True),
                             figsize=[16, 2.5])
    cmap = plt.get_cmap('hsv')
    colors = cmap(np.linspace(0, 1., 6 + 1))
    data = np.array(results['angle_pred'])
    ylim = [func(np.mean(data, axis=0)) for func in [np.min, np.max]]
    for angle, ax, color in zip(range(6), axes, colors):
        plot_tuning_(data[:, angle, :], ax, color)
        ax.set_ylim(ylim)
    report.add_figs_to_section(fig, 'predictions', analysis['name'])

    # plot tuning error at each TOI
    fig, axes = plt.subplots(1, len(tois),
                             figsize=[5 * len(tois), 3], sharey=True)
    ylim = np.mean(np.array(results['toi']), axis=0)
    ylim = np.min(ylim), np.max(ylim)

    for t, (toi, ax) in enumerate(zip(tois, axes)):
        # Across all trials
        data = np.array(results['toi'])[:, t, :]
        plot_tuning_(data, ax, analysis['color'], False)
        ax.set_title('%i - %i ms.' % (toi[0] * 1000, toi[1] * 1000))
        ax.set_ylim(ylim)
    report.add_figs_to_section(fig, 'toi', analysis['name'])

    # seen versus unseen

    def get_sub(key):
        return np.array([subject[key] for subject in results['subscore']])

    fig, ax = plt.subplots(1)
    plot_sem(times[1:], get_sub('seen'), color='r', ax=ax)
    plot_sem(times[1:], get_sub('unseen'), color='b', ax=ax)
    report.add_figs_to_section(fig, 'seen_unseen', analysis['name'])

    seen_toi = get_sub('seen_toi').T
    unseen_toi = get_sub('unseen_toi').T
    fig, axes = plt.subplots(1, len(tois), sharey=True, figsize=[13, 2])
    for ax, unseen, seen, toi in zip(axes, unseen_toi, seen_toi, tois):
        bar_sem(range(3), np.vstack((unseen, seen, seen - unseen)).T, ax=ax,
                color=['b', 'r', 'k'])
        ax.set_xticks([])
        ax.set_title('%i - %i ms.' % (toi[0] * 1000, toi[1] * 1000))
    ax.set_ylim(-np.pi/8, np.pi/8)
    fig.tight_layout()
    report.add_figs_to_section(fig, 'seen_unseen_toi', analysis['name'])

    # contrasts
    cmap = plt.get_cmap('gray_r')
    fig, ax = plt.subplots(1)
    for contrast in [1., .75, .5]:
        data = np.array([subject['contrast' + str(contrast) + '']
                         for subject in results['subscore']])
        plot_sem(times[1:], data, color=cmap(contrast), ax=ax)
    report.add_figs_to_section(fig, 'contrast', analysis['name'])
    data_contrast = list()
    for contrast in [.5, .75, 1.]:
        data_contrast.append(get_sub('contrast' + str(contrast) + '_toi'))
    data_contrast = np.transpose(data_contrast, [2, 1, 0])
    fig, axes = plt.subplots(1, len(tois), sharey=True, figsize=[13, 2])
    cmap = plt.get_cmap('gray_r')
    for ax, data, toi in zip(axes, data_contrast, tois):
        bar_sem(range(3), data, ax=ax, color=cmap([.5, .75, 1.]))
        ax.set_xticks([])
        ax.set_title('%i - %i ms.' % (toi[0] * 1000, toi[1] * 1000))
    fig.tight_layout()
    report.add_figs_to_section(fig, 'contrast_toi', analysis['name'])

    # XXX WIP ANOVA
    # main effect of pas
    data = list()
    for pas in range(4):
        data_ = list()
        for contrast in [.5, .75, 1.]:
            data_.append(get_sub('pas%s-contrast%s' % (pas, contrast)))
        data.append(np.nanmean(data_, axis=0))
    data = np.array(data)
    R = list()
    for subject in np.transpose(data, [1, 0, 2]):
        r, _ = repeated_spearman(subject)
        R.append(r)
    T_obs, h, pval, clusters = stats(R)
    pretty_decod(R, times=times, sig=pval < .05)

    # main effect of contrast
    # TODO absent trials control reactivation

report.save()
