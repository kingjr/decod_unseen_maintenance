import pickle
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from jr.plot import plot_tuning, bar_sem, pretty_decod
from scripts.config import paths, report, analyses
from base import stats
analyses = [analysis for analysis in analyses if analysis['name'] in
            ['target_circAngle', 'probe_circAngle']]

tois = [(-.100, 0.050), (.100, .250), (.300, .800), (.900, 1.050)]


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

    # seen versus unseen
    seen_toi = get_sub('seen_toi').T
    unseen_toi = get_sub('unseen_toi').T

    fig, axes = plt.subplots(1, len(tois), figsize=[9, 2])
    for ax, unseen, seen, toi in zip(axes, unseen_toi, seen_toi, tois):
        print(toi)
        bar_sem(range(2), np.vstack((unseen, seen)).T, ax=ax, color=['b', 'r'])
        quick_stats(np.vstack((unseen, seen)).T, ax=ax)
        ax.set_xticks([])
        ax.set_title('%i $-$ %i ms' % (toi[0] * 1000, toi[1] * 1000))
        ylim_ = get_ylim(seen_toi)
        ax.set_ylim(ylim_)
        ax.set_yticks(ylim_)
        ax.set_yticklabels(['', ''])
        if ax == axes[0]:
            ax.set_yticklabels(['%.2f' % ylim_[0], '%.2f' % ylim_[1]])
    fig.tight_layout()
    fig.subplots_adjust(wspace=.2)
    report.add_figs_to_section(fig, 'seen_unseen_toi', analysis['name'])

    # contrasts
    data_contrast = list()
    for contrast in [.5, .75, 1.]:
        data_contrast.append(get_sub('contrast' + str(contrast) + '_toi'))
    data_contrast = np.transpose(data_contrast, [2, 1, 0])
    fig, axes = plt.subplots(1, len(tois), figsize=[9, 2])
    cmap = plt.get_cmap('gray_r')
    cmap = plt.get_cmap('afmhot_r')
    for ax, data, toi in zip(axes, data_contrast, tois):
        bar_sem(range(3), data, ax=ax, color=cmap([.5, .75, 1.]))
        quick_stats(data, ax=ax)
        ax.set_xticks([])
        ax.set_title('%i $-$ %i ms' % (toi[0] * 1000, toi[1] * 1000))
        ylim_ = get_ylim(data_contrast)
        ax.set_ylim(ylim_)
        ax.set_yticks(ylim_)
        ax.set_yticklabels(['', ''])
        if ax == axes[0]:
            ax.set_yticklabels(['%.2f' % ylim_[0], '%.2f' % ylim_[1]])
    fig.tight_layout()
    fig.subplots_adjust(wspace=.25)
    report.add_figs_to_section(fig, 'contrast_toi', analysis['name'])

    # dynamics
    seen, unseen = get_sub('seen'), get_sub('unseen')
    seen = resample1D(get_sub('seen'))
    unseen = resample1D(get_sub('unseen'))
    times_r = np.squeeze(resample1D(times[1:]))
    p_seen = stats(seen[:, :, None])
    p_unseen = stats(unseen[:, :, None])
    fig, ax = plt.subplots(1, figsize=[5, 2])
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
    ax.set_ylabel('R', labelpad=-15.)
    ax.text(.200, -.05, 'Unseen', color='b', weight='bold', ha='center')
    ax.text(.200, .18, 'Seen', color='r', weight='bold', ha='center')
    report.add_figs_to_section(fig, 'seen_unseen', analysis['name'])

    # Duration early late
    data = results['align_on_diag']
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
            pretty_decod(result[:, 3-ii, toi_align-len(toi_align)/2],
                         color=col, ax=ax, fill=True,
                         times=times_align[toi_align] - t/2., alpha=1.,
                         sig=sig, chance=0.)
            ax.set_yticks([-.07, .15])
            ax.set_yticklabels([-.07, .15])
            ax.set_ylabel('$\Delta angle$', labelpad=-15)
            if ax != axes[0]:
                ax.set_yticklabels(['', ''])
                ax.set_ylabel('')
            ax.set_ylim([-.07, .15])
            xticks = np.arange(-t/2., t/2.+.01, .100)
            ax.set_xticks(xticks)
            ax.set_xticklabels([''] * len(xticks))
            ax.set_aspect('auto')
            ax.set_xlim(-t/2., t/2.)
            if ii == 0:
                ax.set_title('%i $-$ %i ms' % (1e3 * toi[0], 1e3 * toi[1]))
            if ii == 3:
                ax.set_xlabel('Duration', labelpad=-10)
                ax.set_xticklabels([int(x * 1e3) if x in [0., 1.] else ''
                                    for x in xticks])
    # fig.tight_layout()
    report.add_figs_to_section(fig, 'duration', analysis['name'])

    # early maintain
    fig, ax = plt.subplots(1)
    for ii, col in enumerate(['r', 'b']):
        score = np.array(results['early_maintain'])[:, ii, :]
        p_val = stats(score)
        pretty_decod(score, sig=p_val < .05, times=times, ax=ax, color=col,
                     fill=True)
    plt.show()

report.save()
