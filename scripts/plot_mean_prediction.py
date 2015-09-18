import pickle
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from scripts.config import paths, subjects, analyses, subscores, report
from jr.plot import pretty_plot, plot_tuning


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
            tuning_subscores[subanalysis[0]]['diag'] = np.zeros_like(
                tuning_error_diag)
            tuning_subscores[subanalysis[0]]['toi'] = np.zeros_like(
                tuning_error)
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
                    tuning_error_diag[t, :] = get_tuning(y_pred[:, t] -
                                                         y_train)
                # aggregated times of interest
                for t, toi in enumerate(tois):
                    # large window
                    toi_ = np.where((times >= toi[0]) & (times <= toi[1]))[0]
                    y_pred_toi = y_pred[:, toi_]
                    # merge time sample
                    y_train_toi = np.tile(y_train, len(toi_))
                    tuning_error[:, t] = get_tuning(y_pred_toi.T.flatten() -
                                                    y_train_toi)
                    # store prediction for each angle separately
                    for a, angle in enumerate(np.unique(y_train)):
                        sel = np.where(y_train == angle)[0]
                        predict = get_tuning(y_pred_toi[sel, :] - angle)
                        align = (range(a * (n_bins / 6), n_bins) +
                                 range(0, a * (n_bins / 6)))
                        tuning_predict[align, a, t] = predict
                return tuning_error_diag, tuning_error, tuning_predict
            # across all trials
            (tuning_error_diag[s, ...], tuning_error[s, ...],
             tuning_predict[s, ...]) = tunings(y_pred_diag, y_train)
            # on subset of of trials
            for subanalysis in subscores:
                # subselect events
                subevents = events.iloc[sel].reset_index()
                subsel = subevents.query(subanalysis[1]).index
                # subscore
                error_diag, error, _ = tunings(y_pred_diag[subsel, :],
                                               y_train[subsel])
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

    def plot_tuning_(data, ax, color, polar=True):
        plot_tuning(data, ax=ax, color=color, polar=polar, half=polar)
        plot_tuning(data, ax=ax, color='k', polar=polar, half=polar, alpha=0)

    # plot each prediction at probe response
    t = -1 if analysis == 'probe_circAngle' else 0
    fig, axes = plt.subplots(1, 6, subplot_kw=dict(polar=True),
                             figsize=[16, 2.5])
    cmap = plt.get_cmap('hsv')
    colors = cmap(np.linspace(0, 1., 6 + 1))
    ylim = [func(np.mean(tuning_predict_toi[:, :, :, t], axis=0))
            for func in [np.min, np.max]]

    for angle, ax, color in zip(range(6), axes, colors):
        plot_tuning_(tuning_predict_toi[:, :, angle, t], ax, color)
        # FIXME: this is likely to be incorrect: you have to recompute
        # prediction properly
        ax.set_ylim(ylim)
    # report.add_figs_to_section(fig, 'predictions', analysis)

    # plot tuning error at each TOI
    fig, axes = plt.subplots(1, len(tois),
                             figsize=[5 * len(tois), 3], sharey=True)
    for t, (toi, ax) in enumerate(zip(tois, axes)):
        # Across all trials
        plot_tuning_(tuning_error_toi[:, :, t], ax, color, False)
        ax.set_title('%i - %i ms.' % (toi[0] * 1000, toi[1] * 1000))
    report.add_figs_to_section(fig, 'error TOI', analysis)

    # contrasts x visibility XXX TODO

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
