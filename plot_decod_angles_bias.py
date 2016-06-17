"""Plot analyses related to the control analyses of target and probe angles"""

import numpy as np
import matplotlib.pyplot as plt
from jr.plot import (pretty_gat, plot_tuning, pretty_axes, pretty_decod,
                     pretty_colorbar, bar_sem)
from jr.utils import table2html
from config import load, report
from scipy.stats import wilcoxon

cmap = plt.get_cmap('bwr')
colors_vis = cmap(np.linspace(0, 1, 4))

# Load data
results = load('score', subject='fsaverage', analysis='target_probe')
times = results['times']
tois = results['tois']

# Plot tuning curve at probe time for each estimator, alignment and tilt
cmap = plt.get_cmap('BrBG')
colors = cmap(np.linspace(0.2, .8, 3))
fig, axes = plt.subplots(2, 2, figsize=[5, 3.8])
for ii in range(2):  # Estimator: target or probe orientation?
    for jj in range(2):  # Angle error with regard to: target or probe ?
        for tilt, color in enumerate(colors):  # clockwise or counter clockwise
            if tilt == 1:
                continue  # do not plot absent case
            plot_tuning(results['tuning'][:, ii, jj, :, tilt],
                        ax=axes[ii, jj], shift=np.pi, color=color)
            plot_tuning(results['tuning'][:, ii, jj, :, tilt],
                        ax=axes[ii, jj], shift=np.pi, color='k', alpha=0.)
        axes[ii, jj].axvline((jj * 2 - 1) * np.pi / 3, color=colors[0])
        axes[ii, jj].axvline(-(jj * 2 - 1) * np.pi / 3, color=colors[2])

pretty_axes(axes, xticklabels=['$-\pi/2$', '', '$\pi/2$'],
            xlabel='Angle Error',
            ylim=[0.014, .08],
            yticks=[0.014, 1./len(results['bins']), .08],
            yticklabels=[1.4, '', 8.], ylabel='Probability')
fig.tight_layout()
report.add_figs_to_section(fig, 'target_probe', 'cross_generalization')

# Plot bias GAT
fig, axes = plt.subplots(2, 2, figsize=[6.15, 5.6])
fig.subplots_adjust(right=0.85, hspace=0.05, wspace=0.05)
for ii in range(2):
    for jj in range(2):
        scores = -np.array(results['bias'][:, ii, jj, ...])
        p_val = results['bias_pval'][ii, jj, :, :]
        pretty_gat(scores.mean(0), times=times, ax=axes[ii, jj],
                   colorbar=False, clim=[-.1, .1], sig=p_val < .05)
        axes[ii, jj].axvline(.800, color='k')
        axes[ii, jj].axhline(.800, color='k')
pretty_axes(axes, ylabelpad=-15, xticks=np.linspace(-.100, 1.400, 13),
            xticklabels=['', 0] + [''] * 13 + [1400, ''])
pretty_colorbar(cax=fig.add_axes([.88, .25, .02, .55]), ax=axes[0, 0])
report.add_figs_to_section(fig, 'gat', 'bias')

# plot bias diagonal
fig, ax = plt.subplots(1, figsize=[7., 2.])
scores = np.array([np.diag(s) for s in results['bias'][:, 0, 1, ...]])
p_val = np.diag(results['bias_pval'][0, 1, :, :])
color = cmap(1.)
pretty_decod(-scores, ax=ax, times=times, color=color, sig=p_val < .05,
             fill=True)
ax.axvline(.800, color='k')
ax.set_xlabel('Times', labelpad=-10)
fig.tight_layout()
report.add_figs_to_section(fig, 'diagonal', 'bias')


def quick_stats(x, ax=None):
    """Test significant bias in each toi for unseen and seen"""
    pvals = [wilcoxon(ii)[1] for ii in x.T]
    sig = [['', '*'][p < .05] for p in pvals]
    m = np.nanmean(x, axis=0)
    s = np.nanstd(x, axis=0)
    print(m, s, pvals)
    if ax is not None:
        for x_, (y_, sig_) in enumerate(zip(m / 2., sig)):
            ax.text(x_ + .5, y_, sig_, color='w', weight='bold', size=20,
                    ha='center', va='center')

# Test whether angle error to probe varies with tilt
# (first with clf_target, then with clf_probe)
for TP, stimuli in enumerate(['target', 'probe']):
    fig, axes = plt.subplots(1, len(tois), figsize=[8, 2])
    table = np.empty((5, len(tois)), dtype=object)
    for t, (toi, ax) in enumerate(zip(tois, axes)):
        score = list()
        for vis in range(4):
            score.append(-results['bias_vis_toi'][:, TP, 1, vis, t])
        bar_sem(np.vstack(score).T, color=colors_vis, ax=ax)
        quick_stats(np.vstack(score).T, ax=ax)
        diff = score[-1] - score[0]  # max vis - min vis
        # print 'diff', wilcoxon(diff[~np.isnan(diff)])
        for ii, score in enumerate(score + [diff]):
            p_val = wilcoxon(score)[1]
            m = np.nanmean(score)
            sem = np.nanstd(score) / np.sqrt(sum(~np.isnan(score)))
            table[ii, t] = '[%.3f+/-%.3f, p=%.4f]' % (m, sem, p_val)
        ax.set_title('%i $-$ %i ms' % (toi[0] * 1e3, toi[1] * 1e3))
    pretty_axes(axes, xticks=[], xticklabels='', ylim=[-.1, .25],
                yticks=[-.1, 0, .25], yticklabels=[-.1, '', .25])
    fig.tight_layout()
    fig.subplots_adjust(wspace=.1)
    report.add_figs_to_section(fig, 'visibility_%s' % stimuli, 'bias')

    table = np.vstack(([str(t) for t in tois], table))
    table = np.hstack((
        np.array(['', 'vis0', 'vis1', 'vis2', 'vis3', 'diff'])[:, None],
        table))
    report.add_htmls_to_section(table2html(table), stimuli, 'table')

# report assymetry of target probe bias to prove independence
diff = results['bias_toi'][:, 1, 1, 3] - results['bias_toi'][:, 0, 1, 3]
p_val = wilcoxon(diff)[1]
m = np.nanmean(diff)
sem = np.nanstd(diff) / np.sqrt(sum(~np.isnan(diff)))
print('[%.3f+/-%.3f, p=%.4f]' % (m, sem, p_val))

report.save()
