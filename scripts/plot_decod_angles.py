"""Plot analyses related to the decoding of target and probe angles"""
import pickle
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from jr.plot import plot_tuning, bar_sem, pretty_decod
from jr.utils import table2html
from jr.stats import repeated_spearman
from scripts.config import paths, report, analyses, tois
from scripts.base import stats
analyses = [analysis for analysis in analyses if analysis['name'] in
            ['target_circAngle', 'probe_circAngle']]


def resample1D(x):
    factor = 5.
    x = x[:, None].T if x.ndim == 1 else x
    x = x[:, range(int(np.floor(x.shape[1] / factor) * factor))]
    x = x.reshape([x.shape[0], x.shape[1] / factor, factor])
    x = np.nanmedian(x, axis=2)
    return x


def plot_tuning_(data, ax, color, polar=True):
    '''facilitate the tuning'''
    shift = None if polar is True else np.pi
    plot_tuning(np.mean(data, axis=0), ax=ax, color=color, polar=polar,
                half=polar, shift=shift, alpha=.75)
    plot_tuning(data, ax=ax, color='k', polar=polar, half=polar, shift=shift,
                chance=None, alpha=.75)
    if polar:
        ax.set_xticks([0, np.pi])
        ax.set_xticklabels([0, '$\pi$'])
    else:
        ax.set_xticks([-np.pi, np.pi])
        ax.set_xticklabels(['$-\pi/2$', '$\pi/2$'])


def quick_stats(x, ax=None):
    """returns wilcoxon stats and plot stars where p<.05"""
    pvals = [wilcoxon(ii)[1] for ii in x.T]
    sig = [['', '*'][p < .05] for p in pvals]
    m = np.mean(x, axis=0)
    s = np.std(x, axis=0)
    print(m, s, pvals, sig)
    if ax is not None:
        for x_, (y_, sig_) in enumerate(zip(m / 2., sig)):
            ax.text(x_ + .5, y_, sig_, color='w', weight='bold', size=20,
                    ha='center', va='center')


def get_ylim(data):
    """get the y-axis limits"""
    m = np.mean(np.array(data), axis=0)
    s = np.std(np.array(data), axis=0) / np.sqrt(len(data))
    return np.min(m-s*1.1), np.max(m+s*1.1)

ylim = [.01, .08]
for analysis in analyses:
    # load data
    fname = paths('score', subject='fsaverage',
                  analysis=analysis['name'] + '-tuning')
    with open(fname, 'rb') as f:
        results = pickle.load(f)
    times = results['times']
    n_bins = len(results['bins'])

    def get_sub(key):
        return np.array([subject[key] for subject in results['subscore']])

    # Plot the prediction per angle
    t = -1 if analysis['name'] == 'probe_circAngle' else 0
    fig, axes = plt.subplots(1, 6, subplot_kw=dict(polar=True),
                             figsize=[16, 2.5])
    data = np.array(results['angle_pred'])
    for angle, ax in zip(range(6), axes):
        plot_tuning_(data[:, angle, :], ax, analysis['color'])
        ax.set_ylim(get_ylim(data))
    report.add_figs_to_section(fig, 'predictions', analysis['name'])

    # plot tuning error at each TOI
    fig, axes = plt.subplots(1, len(tois), figsize=[9, 2.5])
    for t, (toi, ax) in enumerate(zip(tois, axes)):
        # Across all trials
        data = np.array(results['toi'])[:, t, :]
        plot_tuning_(data, ax, analysis['color'], False)
        ax.set_title('%i $-$ %i ms' % (toi[0] * 1000, toi[1] * 1000))
        # ylim = get_ylim(results['toi'])
        ax.set_ylim(ylim)
        ax.set_yticks([ylim[0], 1./n_bins, ylim[1]])
        ax.set_yticklabels(['', '', ''])
        ax.set_xlabel('Angle Error', labelpad=-10)
        if ax == axes[0]:
            ax.set_yticklabels(['%.2f' % ylim[0], '', '%.2f' % ylim[1]])
            ax.set_ylabel('P ( Trials )', labelpad=-15)
    fig.tight_layout()
    report.add_figs_to_section(fig, 'toi', analysis['name'])

    # seen versus unseen (/ 2 because circle => orientation)
    seen_toi = get_sub('seen_toi').T / 2.
    unseen_toi = get_sub('unseen_toi').T / 2.
    fig, axes = plt.subplots(1, len(tois), figsize=[6, 1.5])
    for ax, unseen, seen, toi in zip(axes, unseen_toi, seen_toi, tois):
        print(toi)
        bar_sem(range(2), np.vstack((unseen, seen)).T, ax=ax, color=['b', 'r'])
        quick_stats(np.vstack((unseen, seen)).T, ax=ax)
        ax.set_xticks([])
        ax.set_title('%i $-$ %i ms' % (toi[0] * 1000, toi[1] * 1000),
                     fontsize=10)
        # ylim_ = get_ylim(seen_toi)
        ylim_ = [-.05, .34]
        ax.set_ylim(ylim_)
        ax.set_yticks(ylim_)
        ax.set_yticklabels(['', ''])
        if ax == axes[0]:
            ax.set_yticklabels(['%.2f' % ylim_[0], '%.2f' % ylim_[1]])
            ax.set_ylabel('rad.', labelpad=-15)
    fig.tight_layout()
    fig.subplots_adjust(wspace=.2)
    report.add_figs_to_section(fig, 'seen_unseen_toi', analysis['name'])

    # contrasts
    data_contrast = list()
    for contrast in [.5, .75, 1.]:
        data_contrast.append(get_sub('contrast' + str(contrast) + '_toi'))
    data_contrast = np.transpose(data_contrast, [2, 1, 0]) / 2.
    fig, axes = plt.subplots(1, len(tois), figsize=[6, 1.5])
    cmap = plt.get_cmap('gray_r')
    cmap = plt.get_cmap('afmhot_r')
    for ax, data, toi in zip(axes, data_contrast, tois):
        bar_sem(range(3), data, ax=ax, color=cmap([.5, .75, 1.]))
        quick_stats(data, ax=ax)
        ax.set_xticks([])
        ax.set_title('%i $-$ %i ms' % (toi[0] * 1000, toi[1] * 1000),
                     fontsize=10)
        # ylim_ = get_ylim(data_contrast)
        ylim_ = [-.05, .34]
        ax.set_ylim(ylim_)
        ax.set_yticks(ylim_)
        ax.set_yticklabels(['', ''])
        if ax == axes[0]:
            ax.set_yticklabels(['%.2f' % ylim_[0], '%.2f' % ylim_[1]])
            ax.set_ylabel('rad.', labelpad=-15)
    fig.tight_layout()
    fig.subplots_adjust(wspace=.25)
    report.add_figs_to_section(fig, 'contrast_toi', analysis['name'])

    # dynamics
    seen, unseen = get_sub('seen') / 2., get_sub('unseen') / 2.
    seen = resample1D(get_sub('seen'))
    unseen = resample1D(get_sub('unseen'))
    times_r = np.squeeze(resample1D(times[1:]))
    p_seen = stats(seen[:, :, None])
    p_unseen = stats(unseen[:, :, None])
    fig, ax = plt.subplots(1, figsize=[6, 2.])
    opts = dict(chance=0, ax=ax, alpha=1., width=1, times=times_r, fill=True)
    pretty_decod(seen, sig=p_seen < .05, color='r', **opts)
    pretty_decod(np.mean(seen, axis=0), sig=p_seen < .05, color='k',
                 times=times_r)
    pretty_decod(unseen, sig=p_unseen < .05, color='b', **opts)
    pretty_decod(np.mean(unseen, axis=0), sig=p_unseen < .05, color='k',
                 times=times_r)
    ax.axvline(.800, color='k')
    ax.set_ylim([-.1, .21])
    ax.set_yticks([ax.get_ylim()[1]])
    ax.set_yticklabels(['%.1f' % ax.get_ylim()[1]])
    ax.set_ylabel('rad.', labelpad=-15.)
    xticks = np.arange(-.100, 1.101, .100)
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(1e3 * x) if x in [0., 1.] else '' for x in xticks])
    ax.set_xlabel('Times', labelpad=-10.)
    ax.text(.200, -.05, 'Unseen', color='b', weight='bold', ha='center')
    ax.text(.200, .18, 'Seen', color='r', weight='bold', ha='center')
    ax.text(0, .2, 'Target',  backgroundcolor='w', ha='center', va='top')
    ax.text(.800, .2, 'Probe', backgroundcolor='w', ha='center', va='top')
    fig.tight_layout()
    report.add_figs_to_section(fig, 'seen_unseen', analysis['name'])

    # Duration early late
    data = np.array(results['align_on_diag']) / 2.
    data = [data[:, 1, :, :], data[:, 2, :, :]]
    freq = np.ptp(times) / len(times)
    times_align = times - times.min()
    # fig, axes = plt.subplots(1, len(data), figsize=[2, 4])
    fig = plt.figure(figsize=[7.8, 5.5])
    axes = list()
    for ii in range(4):
        axes.append([plt.subplot2grid((4, 3), (ii, 0), colspan=1),
                     plt.subplot2grid((4, 3), (ii, 1), colspan=2)])
    cmap = plt.get_cmap('bwr_r')
    for jj, result, toi, t in zip(range(2), data, tois[1:], [.300, .600]):
        for ii, col in enumerate(cmap(np.linspace(0, 1, 4.))):
            ax = axes[ii][jj]
            toi_align = np.where((times - times.min()) <= t)[0]
            sig = stats(result[:, 3-ii, toi_align-len(toi_align)/2]) < .05
            pretty_decod(result[:, 3, toi_align-len(toi_align)/2], ax=ax,
                         times=times_align[toi_align] - t/2, color='r',
                         chance=0.)
            pretty_decod(result[:, 3-ii, toi_align-len(toi_align)/2],
                         color=col, ax=ax, fill=True,
                         times=times_align[toi_align] - t/2., alpha=1.,
                         sig=sig, chance=0.)
            ax.set_yticks([-.03, .07])
            ax.set_yticklabels([-.03, .07])
            ax.set_ylabel('$\Delta angle$', labelpad=-15)
            if jj != 0:
                ax.set_yticklabels(['', ''])
                ax.set_ylabel('')
            ax.set_ylim([-.03, .07])
            xticks = np.arange(-t/2., t/2.+.01, .100)
            ax.set_xticks(xticks)
            ax.set_xticklabels([''] * len(xticks))
            ax.set_aspect('auto')
            ax.set_xlim(-t/2.-.01, t/2.+.01)
            ax.set_xlabel('')
            if ii == 0:
                ax.set_title('%i $-$ %i ms' % (1e3 * toi[0], 1e3 * toi[1]))
            if ii == 3:
                ax.set_xlabel('Duration', labelpad=-10)
                ax.set_xticklabels(
                    [int(x) if np.round(x) in [-t/2 * 1e3, t/2 * 1e3]
                     else '' for x in np.round(1e3 * xticks)])
    fig.tight_layout()
    report.add_figs_to_section(fig, 'duration', analysis['name'])

    # early maintain
    fig, ax = plt.subplots(1)
    for ii, col in enumerate(['r', 'b']):
        score = np.array(results['early_maintain'])[:, ii, :] / 2.
        p_val = stats(score)
        pretty_decod(score, sig=p_val < .05, times=times, ax=ax, color=col,
                     fill=True)
    # plt.show()

    # Report Stats Table
    table = np.empty((6, len(tois)), dtype=object)
    # Score seen unseen
    for ii, score in enumerate([get_sub('seen_toi') / 2.,
                                get_sub('unseen_toi') / 2.]):
        for jj, toi in enumerate(tois):
            score_ = score[:, jj]
            p_val = wilcoxon(score_)[1]
            table[ii, jj] = '[%.3f+/-%.3f, p=%.4f]' % (
                np.nanmean(score_), np.nanstd(score_) / np.sqrt(len(score_)),
                p_val)

    # Difference seen unseen:
    score = get_sub('seen_toi') / 2. - get_sub('unseen_toi') / 2.
    for jj, toi in enumerate(tois):
        score_ = score[:, jj]
        p_val = wilcoxon(score_)[1]
        table[2, jj] = '[%.3f+/-%.3f, p=%.4f]' % (
            np.nanmean(score_), np.nanstd(score_) / np.sqrt(len(score_)),
            p_val)

    # Contrast effect:
    score = list()
    for contrast in [.5, .75, 1.]:
        score.append(get_sub('contrast' + str(contrast) + '_toi'))
    score = np.transpose(score, [2, 1, 0]) / 2.
    for jj, toi in enumerate(tois):
        R = repeated_spearman(np.transpose(score[jj, :, :]),
                              np.array([.5, .75, 1.]))
        p_val = wilcoxon(R)[1]
        table[3, jj] = '[%.3f+/-%.3f, p=%.4f]' % (
            np.nanmean(R), np.nanstd(R) / np.sqrt(len(R)),
            p_val)

    # Across trials
    # R contrast
    score = np.squeeze(results['R_contrast_toi'])
    for jj, toi in enumerate(tois):
        score_ = score[:, jj]
        p_val = wilcoxon(score_)[1]
        table[4, jj] = '[%.3f+/-%.3f, p=%.4f]' % (
            np.nanmean(score_), np.nanstd(score_) / np.sqrt(len(score_)),
            p_val)
    print('contrast:late-early:%.4f' % wilcoxon(score[:, 2] - score[:, 1])[1])

    # R vis
    score = np.squeeze(results['R_vis_toi'])
    for jj, toi in enumerate(tois):
        score_ = score[:, jj]
        p_val = wilcoxon(score_)[1]
        table[5, jj] = '[%.3f+/-%.3f, p=%.4f]' % (
            np.nanmean(score_), np.nanstd(score_) / np.sqrt(len(score_)),
            p_val)
    table = np.vstack(([str(t) for t in tois], table))
    table = np.hstack((np.array(['', 'seen', 'unseen', 's-u', 'contrast',
                                 'R contrast', 'R vis'])[:, None], table))

    report.add_htmls_to_section(table2html(table), analysis['name'], 'table')


report.save()
