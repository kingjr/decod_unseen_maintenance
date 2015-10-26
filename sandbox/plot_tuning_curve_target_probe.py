import pickle
import os.path as op
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from jr.plot import plot_sem, pretty_plot
from scripts.config import paths, subjects, report

# tuning curve as a function of visibility
visibilities = ['unseen', 'pas1.0', 'pas2.0', 'pas3.0']
contrasts = ['contrast0.5', 'contrast0.75', 'contrast1.0']
analyses = [
    dict(name='probe_visibility_tuning',
         subanalyses=['probe_circAngle-tuning-%s' % s for s in visibilities],
         subtitles=['visibility=%i' % v for v in range(4)]),
    dict(name='target_visibility_tuning',
         subanalyses=['target_circAngle-tuning-%s' % s for s in visibilities],
         subtitles=['visibility=%i' % v for v in range(4)]),
    dict(name='target_contrast_tuning',
         subanalyses=['target_circAngle-tuning-%s' % c for c in contrasts],
         subtitles=['contrast=%i' % v for v in [.5, .75, 1.]]),
]


for analysis in analyses:
    pkl_fname = paths('score', subject='fsaverage', analysis=analysis['name'])
    if not op.exists(pkl_fname):
        # GATHER DATA
        all_scores = list()
        all_ypreds = list()
        for subanalysis in analysis['subanalyses']:
            print analysis
            scores = list()
            y_preds = list()
            for subject in subjects:
                print subject
                # define path to file to be loaded
                score_fname = paths('score', subject=subject,
                                    analysis=subanalysis)
                with open(score_fname, 'rb') as f:
                    out = pickle.load(f)
                    gat = out[0]

                scores.append(gat.scores_)
                y_preds.append(gat.y_pred_)
            all_scores.append(scores)
            all_ypreds.append(y_preds)
        # Save
        with open(pkl_fname, 'wb') as f:
            gat.estimators = list()
            gat.y_pred_ = list()
            pickle.dump([gat, all_scores, all_ypreds], f)
    else:
        with open(pkl_fname, 'rb') as f:
            gat, all_scores, all_ypreds = pickle.load(f)

    # TUNING DIAGONAL
    # Try to render it better...
    all_ypreds = np.array(all_ypreds)
    # all_ypreds_6 = np.zeros((4, len(subjects), 154, 154, 3))
    # for a in range(3):
    #     all_ypreds_6[..., a] = np.sum(all_ypreds[..., (6*a):(6*(a+1))], axis=4)
    times = gat.train_times_['times'] * 1000
    chance = 1. / (all_ypreds.shape[-1] - 1)
    vmin = 1.5 * chance
    vmax = 0.5 * chance
    fig_tune, axes_tune = plt.subplots(len(all_ypreds), 1,
                                       figsize=[8, 7. / 4. * len(all_ypreds)])
    for y_preds, ax, title in zip(all_ypreds, axes_tune, analysis['subtitles']):
        tuning_diag = [np.diag(np.squeeze(p))
                       for p in np.mean(y_preds, axis=0).transpose(2, 0, 1)]
        # XXX
        spread = np.percentile(np.abs(chance - np.array(tuning_diag)), 99.5)
        vmin_ = chance - spread
        vmax_ = chance + spread
        im = ax.matshow(tuning_diag, aspect='auto', vmin=vmin_, vmax=vmax_,
                        extent=[min(times), max(times), -np.pi/2, np.pi/2],
                        cmap='RdBu_r')
        ax.axvline(0, color='k')
        ax.axvline(800, color='k')
        cb = plt.colorbar(im, ax=ax, ticks=[vmin_, chance, vmax_])
        cb.ax.set_yticklabels(['%.2f' % vmin_, 'Chance', '%.2f' % vmax_],
                              color='dimgray')
        cb.ax.xaxis.label.set_color('dimgray')
        cb.ax.yaxis.label.set_color('dimgray')
        cb.ax.spines['left'].set_color('dimgray')
        cb.ax.spines['right'].set_color('dimgray')
        pretty_plot(ax)
        if ax != axes_tune[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Times (s.)')
        ax.set_yticks([-1.52, 1.52])
        ax.set_yticklabels(['$-\pi/2$', '$-\pi/2$'])
        ax.set_ylabel('Angle Error')
        ax.set_title(title, color='dimgray', size=13, va='center')

    # SCORE DIAGONAL
    fig_diag, ax = plt.subplots(1, figsize=[8, 3])
    chance = 0.
    cmap = mpl.colors.LinearSegmentedColormap.from_list('RdPuBu', ['b', 'r'])
    colors = cmap([0., .33, .66, 1.])
    for ii, (scores, color, title) in enumerate(
            zip(all_scores, colors, analysis['subtitles'])):
        # only plot extremes for readibility
        if ii in [1, 2] and 'visibility' in analysis['name']:
            continue
        scores_diag = np.array([np.diag(score) for score in scores])
        plot_sem(times, scores_diag, color=color, line_args=dict(label=title))
    ax.axhline(chance, linestyle='dotted', color='k', label=None)
    ax.axvline(0, color='k')
    ax.axvline(800, color='k')
    ax.set_yticks([0., .35])
    ax.set_yticklabels(['chance', '0.35'])
    ax.set_xlim(times[0], times[-1])
    ax.text(0, .35, 'Target',  backgroundcolor='w', ha='center', va='top')
    ax.text(800, .35, 'Probe', backgroundcolor='w', ha='center', va='top')
    legend = ax.legend(loc=[.15, .70], ncol=2, fontsize=12,
                       markerscale=.05, frameon=False)
    for color, text in zip(colors, legend.get_texts()):
        text.set_color(color)
    ax.set_xlabel('Times (s.)')
    ax.set_ylabel('Mean Angle Error')
    pretty_plot(ax)

    # GAT
    fig_gat, axes = plt.subplots(1, 4, figsize=[20, 6])
    ymin = -.17
    ymax = .17
    for ax, scores, title in zip(axes, all_scores, analysis['subtitles']):
        ax.matshow(np.nanmean(scores, axis=0), vmin=ymin, vmax=ymax,
                   extent=[min(times), max(times)] * 2, cmap='RdBu_r',
                   origin='lower')
        ax.axhline(0, color='k')
        ax.axhline(800, color='k')
        ax.axvline(0, color='k')
        ax.axvline(800, color='k')
        pretty_plot(ax)
        if ax != axes[0]:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Train Times (s.)')
        ax.set_xlabel('Test Times (s.)')
        ax.set_title(title)

    # Plot GAT slices
    # XXX WIP: too noisy to show?
    ymin = -.20
    ymax = .20
    plot_times = [180, 300] #np.linspace(100, 400, 20)
    fig_offdiag, axes = plt.subplots(len(plot_times), 1, figsize=[5, 6])
    for sel_time, ax in zip(plot_times, reversed(axes)):
        for ii, (scores, color) in enumerate(zip(all_scores, colors)):
            # only plot extremes for readibility
            if ii in [1, 2] and 'visibility' in analysis['name']:
                continue
            scores = np.array(scores)
            # plot shading between two lines if diag is sig. != from off diag
            idx = np.argmin(abs(times-sel_time))
            scores_off = np.array(scores)[:, idx, :]
            # change line width where sig != from change
            plot_sem(times, scores_off, color=color, ax=ax, alpha=.2,
                     line_args=dict(linewidth=2))
        ax.axhline(chance, linestyle='dotted', color='k')
        ax.axvline(0, color='k')
        ax.axvline(800, color='k')
        ax.plot([sel_time] * 2, [ymin, scores_off.mean(0)[idx]], color='k')
        ax.set_xlim(np.min(times), np.max(times))
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([ymin, ymax])
        ax.text(sel_time, ymin + np.ptp([ymin, ymax])/20, '%i ms.' % sel_time,
                color='k', backgroundcolor='w', ha='center')
        ax.text(0, ymax, 'Target', color='k', backgroundcolor='w',
                ha='center', va='top')
        ax.text(800, ymax, 'Probe', color='k', backgroundcolor='w',
                ha='center', va='top')
        pretty_plot(ax)
        if ax != axes[-1]:
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.spines['bottom'].set_visible(False)
    # around diagonal

    def diag_alignment(scores):
        scores = np.array(scores)
        nt = len(times)
        for t in range(len(times)):
            time_slice = np.array(range(t, nt) + range(0, t))
            scores[t, :] = scores[t, (nt / 2 + time_slice) % nt]
        return scores

    tois = [[50, 200], [200, 800]]
    fig_duration, axes = plt.subplots(2, 1, figsize=[4, 8])
    ymin, ymax = -.06, .10
    for toi, ax in zip(tois, axes):
        for ii, (scores, color) in enumerate(zip(all_scores, colors)):
            # only plot extremes for readibility
            if ii in [1, 2]:
                continue
            scores_aligned = np.array([diag_alignment(score) for score in scores])
            toi_idx = np.where((times > toi[0]) & (times < toi[1]))[0]
            score_duration = np.mean(scores_aligned[:, toi_idx, :], axis=1)
            score_duration = score_duration
            plot_sem(times - np.ptp(times) / 2 - times[0], score_duration,
                     color=color, line_args=dict(linewidth=2), ax=ax)
        ax.set_xlim(-401, 401)
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([ymin, 0, ymax])
        ax.set_xticks(range(-400, 400, 200))
        pretty_plot(ax)
        ax.axhline(0, linestyle='dotted')
        ax.set_ylabel('Score')
        ax.set_title('Trained between %i & %i ms.' % (toi[0], toi[1]),
                     color='dimgray')
        if ax == axes[1]:
            ax.set_xlabel('Time from train (ms.)')
            ax.text(-100, .07, 'max visibility', color='red')
            ax.text(0, -.035, 'unseen', color='blue', ha='center')
        else:
            ax.text(100, .07, 'max visibility', color='red')
            ax.text(-200, .07, 'unseen', color='blue', ha='center')
            ax.set_xticklabels([])
    # XXX duration as a function of time
    report.add_figs_to_section(
        [fig_tune, fig_diag, fig_gat, fig_duration, fig_offdiag],
        [analysis['name']] * 5, analysis['name'])

report.save()
