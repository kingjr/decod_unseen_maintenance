import pickle
import numpy as np
import matplotlib.pyplot as plt
from jr.plot import (pretty_gat, plot_tuning, pretty_axes, pretty_decod,
                     pretty_colorbar, bar_sem)
from scripts.config import paths, report
from scipy.stats import wilcoxon

fname = paths('score', subject='fsaverage', analysis='target_probe')
with open(fname, 'rb') as f:
    results = pickle.load(f)
times = results['times']
tois = results['tois']

# Plot tuning probe time: train test target probe
cmap = plt.get_cmap('BrBG')
colors = cmap(np.linspace(0.2, .8, 3))
fig, axes = plt.subplots(2, 2)
for ii in range(2):
    for jj in range(2):
        for tilt, color in enumerate(colors):
            if tilt == 1:
                continue  # do not plot absent case
            plot_tuning(results['tuning'][:, ii, jj, :, tilt],
                        ax=axes[ii, jj], shift=np.pi, color=color)
            plot_tuning(results['tuning'][:, ii, jj, :, tilt],
                        ax=axes[ii, jj], shift=np.pi, color='k', alpha=0.)
            axes[ii, jj].axvline(-np.pi / 3, color='k')
            axes[ii, jj].axvline(np.pi / 3, color='k')
            axes[ii, jj].axvline(0, color='k')
pretty_axes(axes, xticklabels=['$-\pi/2$', '', '$\pi/2$'],
            xlabel='Angle Error',
            yticks=[0.014, 1./len(results['bins']), 0.08],
            yticklabels=[1.4, '', 8.], ylabel='Probability')
fig.tight_layout()
report.add_figs_to_sections(fig, 'target_probe', 'cross_generalization')

# Plot bias GAT
fig, axes = plt.subplots(2, 2, figsize=[6.15, 6.])
for ii in range(2):
    for jj in range(2):
        scores = np.array(results['bias'][:, ii, jj, ...])
        p_val = results['bias_pval'][ii, jj, :, :].T  # XXX ? why T
        pretty_gat(scores.mean(0), times=times, ax=axes[ii, jj],
                   colorbar=False, clim=[-.1, .1], sig=p_val < .05)
        axes[ii, jj].axvline(.800, color='k')
        axes[ii, jj].axhline(.800, color='k')
pretty_axes(axes)
pretty_colorbar(cax=fig.add_axes([.92, .2, .025, .55]), ax=axes[0, 0])
report.add_figs_to_sections(fig, 'gat', 'bias')

# plot bias diagonal
fig, ax = plt.subplots(1, figsize=[7., 2.])
scores = np.array([np.diag(s) for s in results['bias'][:, 0, 1, ...]])
p_val = np.diag(results['bias_pval'][0, 1, :, :])
color = cmap(1.)
pretty_decod(-scores, ax=ax, times=times, color=color, sig=p_val < .05,
             fill=True)
ax.axvline(.800, color='k')
report.add_figs_to_sections(fig, 'diagonal', 'bias')

# Test significant bias in each toi for unseen and seen


def quick_stats(x, ax=None):
    pvals = [wilcoxon(ii[~np.isnan(ii)])[1] for ii in x.T]
    sig = [['', '*'][p < .05] for p in pvals]
    m = np.nanmean(x, axis=0)
    s = np.nanstd(x, axis=0)
    print(m, s, pvals)
    if ax is not None:
        for x_, (y_, sig_) in enumerate(zip(m / 2., sig)):
            ax.text(x_ + .5, y_, sig_, color='w', weight='bold', size=20,
                    ha='center', va='center')

fig, axes = plt.subplots(1, len(tois), figsize=[8, 2])
for t, (toi, ax) in enumerate(zip(tois, axes)):
    absent = -results['target_absent_bias_toi'][:, t]
    seen = -results['bias_vis_toi'][:, 0, 1, 3, t]
    unseen = -results['bias_vis_toi'][:, 0, 1, 0, t]
    bar_sem(np.vstack((absent, unseen, seen)).T, color=['k', 'b', 'r'], ax=ax)
    quick_stats(np.vstack((absent, unseen, seen)).T, ax=ax)
    diff = seen - unseen
    print wilcoxon(diff[~np.isnan(diff)])
    ax.set_title('%i $-$ %i ms' % (toi[0] * 1e3, toi[1] * 1e3))
pretty_axes(axes, xticks=[], xticklabels='', ylim=[-.1, .25],
            yticks=[-.1, 0, .25], yticklabels=[-.1, '', .25])
fig.tight_layout()
fig.subplots_adjust(wspace=.1)
report.add_figs_to_sections(fig, 'visibility', 'bias')

report.save()
