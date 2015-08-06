import pickle
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from scripts.config import paths, subjects, analyses, subscores
from base import plot_sem
from meeg_preprocessing.utils import setup_provenance

report, run_id, _, logger = setup_provenance(
    script=__file__, results_dir=paths('report'))


def pretty_plot(ax):
    ax.tick_params(colors='dimgray')
    ax.xaxis.label.set_color('dimgray')
    ax.yaxis.label.set_color('dimgray')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_color('dimgray')
    ax.spines['bottom'].set_color('dimgray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def pretty_polar_plot(ax):
    ax.set_rticks([])
    ax.set_xticks(2 * np.pi * (15. + 30. * np.arange(6)) / 360.)
    ax.set_xticklabels(np.array(15. + 30. * np.arange(6), dtype=int))
    ax.tick_params(colors='dimgray')
    ax.xaxis.label.set_color('dimgray')
    ax.yaxis.label.set_color('dimgray')
    ax.spines['polar'].set_color('dimgray')


def get_tuning(y_pred):
    h, _ = np.histogram((y_pred + np.pi) % (2 * np.pi), bins)
    error = 1. * h / np.sum(h)
    return error

tois = [(.100, .250), (.250, .800), (.900, 1.050)]
n_bins = 99
bins = np.linspace(0, 2 * np.pi, n_bins + 1)
cmap = plt.get_cmap('gist_rainbow')
_colors = cmap(np.linspace(0, 1., len(analyses) + 1))
analysis_colors = [_colors[3], _colors[4]]
for analysis, analysis_color in zip(['target_circAngle', 'probe_circAngle'],
                                    analysis_colors):
    score_fname = paths('score', subject='fsaverage',
                        analysis=analysis + '-tuning')
    if not op.exists(score_fname):
        tuning_error = np.zeros((len(subjects), n_bins, len(tois)))
        tuning_predict = np.zeros((len(subjects), n_bins, 6, len(tois)))
        tuning_error_diag = np.zeros((len(subjects), 154, n_bins))
        tuning_subscores = dict()
        for subanalysis in subscores:
            tuning_subscores[subanalysis[0]] = dict()
            tuning_subscores[subanalysis[0]]['diag'] = np.zeros_like(tuning_error_diag)
            tuning_subscores[subanalysis[0]]['toi'] = np.zeros_like(tuning_error)
        for s, subject in enumerate(subjects):
            print subject
            # define path to file to be loaded
            fname = paths('decod', subject=subject, analysis=analysis)
            with open(fname, 'rb') as f:
                gat, _, sel, events = pickle.load(f)
            times = gat.train_times_['times']

            # get diagonal
            y_train = gat.y_train_
            y_pred_diag = np.zeros((len(gat.y_pred_[0][0]), len(times)))
            for t in range(len(times)):
                y_pred_diag[:, t] = gat.y_pred_[t][t][:, 0]

            # tuning curve for prediction, and prediction errors
            def tunings(y_pred, y_train):
                tuning_error = np.zeros((n_bins, len(tois)))
                tuning_predict = np.zeros((n_bins, 6, len(tois)))
                tuning_error_diag = np.zeros((len(times), n_bins))

                # diagonal
                for t in range(len(times)):
                    tuning_error_diag[t, :] = get_tuning(y_pred[:, t] - y_train)
                # aggregated times of interest
                for t, toi in enumerate(tois):
                    # large window
                    toi_ = np.where((times >= toi[0]) & (times <= toi[1]))[0]
                    y_pred_toi = y_pred[:, toi_]
                    # merge time sample
                    y_train_toi = np.tile(y_train, len(toi_))
                    tuning_error[:, t] = get_tuning(y_pred_toi.T.flatten() - y_train_toi)
                    # store prediction for each angle separately
                    for a, angle in enumerate(np.unique(y_train)):
                        sel = np.where(y_train == angle)[0]
                        predict = get_tuning(y_pred_toi[sel, :] - angle)
                        align = (range(a * (n_bins / 6), n_bins) +
                                 range(0, a * (n_bins / 6)))
                        tuning_predict[align, a, t] = predict
                return tuning_error_diag, tuning_error, tuning_predict
            # across all trials
            tuning_error_diag[s, ...], tuning_error[s, ...], tuning_predict[s, ...] = tunings(y_pred_diag, y_train)
            # on subset of of trials
            for subanalysis in subscores:
                # subselect events
                subevents = events.iloc[sel].reset_index()
                subsel = subevents.query(subanalysis[1]).index
                # subscore
                error_diag, error, _ = tunings(y_pred_diag[subsel, :], y_train[subsel])
                tuning_subscores[subanalysis[0]]['diag'][s, ...] = error_diag
                tuning_subscores[subanalysis[0]]['toi'][s, ...] = error

        with open(score_fname, 'wb') as f:
            pickle.dump(dict(tuning_error=tuning_error, tois=tois, times=times,
                             bins=bins, tuning_predict=tuning_predict,
                             tuning_error_diag=tuning_error_diag,
                             tuning_subscores=tuning_subscores), f)
        tuning_predict_toi = tuning_predict
        tuning_error_toi = tuning_error
    else:
        with open(score_fname, 'rb') as f:
            out = pickle.load(f)
            tuning_error_toi = out['tuning_error']
            tois = out['tois']
            bins = out['bins']
            tuning_predict_toi = out['tuning_predict']
            times = out['times']
            tuning_error_diag = out['tuning_error_diag']
            tuning_subscores = out['tuning_subscores']

    # bin center (true angle in experiment started at 15 degrees)
    bins_center = ((bins[1:] - np.ptp(bins[:2]) / 2) +
                   15 / 360 * np.pi - np.pi / 2)
    closedrange = range(n_bins) + [0]
    bins_c = np.hstack((bins_center / 2, np.pi + bins_center[closedrange] / 2))
    bins_c[-1] = bins_c[0]
    def circular_tuning(tuning_error):
        tuning_error_c = np.concatenate((tuning_error,
                                         tuning_error[:, closedrange]),
                                        axis=1)
        return tuning_error_c
    def circular_predict(tuning_predict):
        tuning_predict_c = np.concatenate((tuning_predict,
                                           tuning_predict[:, closedrange, :]),
                                          axis=1)
        return tuning_predict_c

    def plot_sem_polar(angles, radius, ax, color='b'):
        sem = np.std(radius, axis=0) / np.sqrt(len(subjects))
        m = np.mean(radius, axis=0)
        ax.plot(angles, m, color='k', linewidth=2)
        ax.fill_between(np.hstack((angles, angles[::-1])),
                        np.hstack((m, m[::-1])) + np.hstack((sem, -sem[::1])),
                        facecolor=color, edgecolor='none', alpha=.5)
        ax.fill_between(angles, m, facecolor=color, edgecolor='none', alpha=.5)

    # plot each prediction

    def plot_predict(tuning_predict_c):
        fig_predict = plt.figure(figsize=[20, 3])
        cmap = plt.get_cmap('hsv')
        colors = cmap(np.linspace(0, 1., 6 + 1))
        rmax = tuning_predict_c.mean(0).max()
        for a, (tuning, angle, color) in enumerate(
                zip(tuning_predict_c.transpose([2, 0, 1]),
                    np.arange(6) * 30 + 15, colors)):
            ax = plt.subplot(160+a+1, polar=True)
            plot_sem_polar(bins_c - 30. / 360. * 2 * np.pi, tuning, ax, color=color)
            pretty_polar_plot(ax)
            ax.set_ylim(0, rmax)
            ax.set_xticks([angle / 360. * 2 * np.pi])
            ax.set_xticklabels([int(angle)])
            ax.set_title(str(int(angle)) + '$^\circ$')
        return fig_predict

    # plot prediction error

    def plot_error_polar(tuning_error_c, ax):
        plot_sem_polar(np.linspace(-np.pi, np.pi, 2 * n_bins + 1), tuning_error_c,
                       ax, color=analysis_color)
        ax.plot(np.linspace(-np.pi, np.pi, 2 * n_bins + 1), tuning_error_c.mean(0),
                color='k', linewidth=3.)
        pretty_polar_plot(ax)
        ax.set_xticks([0, np.pi/2, -np.pi/2, np.pi])
        ax.set_xticklabels(['90', '0', ''])
        ax.set_title('Angle Error')

    def plot_error(tuning_error, ax, color='k', n_bins_=n_bins, alpha=.5, ylim='chance'):
        # ylim = ax.get_ylim()
        tuning_error = tuning_error[np.sum(tuning_error, axis=1) > 0, :]
        if n_bins_ != n_bins:
            binsize = n_bins / n_bins_
            tuning_error = np.transpose([
                np.sum(tuning_error[:, b * binsize + np.arange(binsize)], axis=1)
                for b in range(n_bins_)])
        facecolor = color if color != 'k' else analysis_color
        hpi = np.pi / 2
        chance = 1. / (n_bins_ + 1)
        ax.fill_between(
            np.concatenate(([-hpi], np.linspace(-hpi, hpi, n_bins_), [hpi, -hpi])),
            np.concatenate(([chance], tuning_error.mean(0), [chance] * 2)),
            facecolor=facecolor, edgecolor='none', alpha=alpha)
        plot_sem(np.linspace(-np.pi/2, np.pi/2, n_bins_), tuning_error, ax=ax,
                 color=color)
        ax.plot(np.linspace(-np.pi/2, np.pi/2, n_bins_), tuning_error.mean(0),
                color=color, linewidth=3)
        ax.axhline(chance, linestyle='dotted', color='k')
        ax.set_xlabel('Angle Error')
        ax.set_xlim(-np.pi/2, np.pi/2)
        ax.set_xticks([-np.pi / 2, 0, np.pi / 2])
        ax.set_xticklabels(['-$\pi$ / 2', '0',  '$\pi$ / 2'])
        ax.set_ylabel('Trials Proportion (%)')
        if ylim == 'chance':
            ylim = chance + np.array([-1, 1]) * chance / 2.
        else:
            ylim = np.array([0.9, 1.1]) * [
                tuning_error.mean(0).min(), tuning_error.mean(0).max()]
        print ylim
        ax.set_ylim(ylim)
        ax.set_yticks([ylim[0], chance, ylim[1]])
        ax.set_yticklabels(['%.2f' % ylim[0], 'chance', '%.2f' % ylim[1]])
        pretty_plot(ax)

    # plot each prediction at probe respose
    t = -1 if analysis == 'probe_circAngle' else 0
    tuning_predict_c = circular_tuning(tuning_predict_toi[:, :, :, t])
    fig_predict = plot_predict(tuning_predict_c)
    report.add_figs_to_section(fig_predict, analysis + str(tois[t]), analysis)

    # plot tuning error at each TOI
    fig_error, axes_err = plt.subplots(
        1, len(tois), figsize=[5 * len(tois), 3], sharey=True)
    fig_vis, axes_vis = plt.subplots(
        1, len(tois), figsize=[5 * len(tois), 3], sharey=True)
    fig_contrast, axes_contrast = plt.subplots(
        1, len(tois), figsize=[5 * len(tois), 3], sharey=True)
    for t, (toi, ax_err, ax_vis, ax_contrast) in enumerate(zip(
            tois, axes_err, axes_vis, axes_contrast)):
        # Across all trials
        tuning_predict_c = circular_tuning(tuning_predict_toi[:, :, :, t])
        tuning_error = tuning_error_toi[:, :, t]
        plot_error(tuning_error, ax_err)
        ax_err.set_title('%i - %i ms.' % (toi[0] * 1000, toi[1] * 1000))
        # visibility
        seen = tuning_subscores['seen']['toi'][:, :, t]
        unseen = tuning_subscores['unseen']['toi'][:, :, t]
        plot_error(unseen, ax_vis, color='b', n_bins_=10, alpha=0., ylim=None)
        plot_error(seen, ax_vis, color='r', n_bins_=10, alpha=0., ylim=None)
        ax_vis.set_title('%i - %i ms.' % (toi[0] * 1000, toi[1] * 1000))
        # contrasts
        for contrast in [.5, .75, 1.]:
            error = tuning_subscores['contrast%s' % contrast]['toi'][:, :, t]
            plot_error(error, ax_contrast, color=[1 - contrast] * 3, ylim=None, alpha=0.)
        ax_contrast.set_title('%i - %i ms.' % (toi[0] * 1000, toi[1] * 1000))

        # contrasts x visibility XXX TODO
    report.add_figs_to_section([fig_error, fig_vis, fig_contrast],
                               [analysis] * 3, analysis)

    # plot diagonal
    fig_diag, ax = plt.subplots(1, figsize=[10, 3])
    ymin, ymax = np.percentile(tuning_error_diag.mean(0), [2, 98])
    im = ax.matshow(tuning_error_diag.mean(0).T, aspect='auto', cmap='RdBu_r',
                    vmin=ymin, vmax=ymax,
                    extent=[min(times), max(times), -np.pi/2, np.pi/2])
    cb = plt.colorbar(im, ax=ax, ticks=[ymin, 1. / n_bins, ymax])
    cb.ax.set_yticklabels(['%i' % ymin, 'Chance', '%i' % ymax],
                          color='dimgray')
    cb.ax.xaxis.label.set_color('dimgray')
    cb.ax.yaxis.label.set_color('dimgray')
    cb.ax.spines['left'].set_color('dimgray')
    cb.ax.spines['right'].set_color('dimgray')
    ax.set_xlabel('Times (ms.)')
    ax.set_ylabel('Angle Error')
    ax.set_yticks([-np.pi / 2, 0, np.pi / 2])
    ax.set_yticklabels(['-$\pi$ / 2', '0',  '$\pi$ / 2'])
    pretty_plot(ax)
    report.add_figs_to_section(fig_diag, 'diagonal', analysis)

report.save()
