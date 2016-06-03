import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from jr.plot import pretty_plot, pretty_colorbar
from base import stats
from conditions import analyses
from config import load, subjects, report

fig, axes = plt.subplots(5, 2, figsize=[8, 15])
axes = axes.ravel()
for analysis, ax in zip(analyses, axes):
    scores = list()
    for subject in subjects:
        score, times, freqs = load('score_tfr', subject=subject,
                                   analysis=analysis['name'])
        scores.append(score)
    scores = np.array(scores)
    if 'circAngle' in analysis['name']:
        scores /= 2
    # compute stats
    p_val = stats(scores - analysis['chance'])
    sig = p_val < .05

    # plot effect size
    scores = np.mean(scores, axis=0)
    cmap = LinearSegmentedColormap.from_list(
        'RdBu', ['w', analysis['color'], 'k'])
    im = ax.matshow(scores, aspect='auto', origin='lower',
                    extent=[times[0], times[-1], 0, len(freqs)],
                    vmin=analysis['chance'], vmax=np.max(scores), cmap=cmap)

    # plot stats
    xx, yy = np.meshgrid(times, range(len(freqs)), copy=False, indexing='xy')
    ax.contour(xx, yy, sig, colors='black', levels=[0],
               linestyles='dotted')

    # pretty plot
    pretty_plot(ax)
    ticks = []
    for ii in np.arange(10, 71, 10):
        ticks.append(np.where(np.round(freqs) >= ii)[0][0])
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    if ax in axes[::2]:
        ax.set_yticklabels([10, '', 30, '', '', '', 70])
    ax.set_xticks([0., .800])
    ax.set_xticklabels(['', ''])
    if ax in axes[-2:]:
        ax.set_xticklabels([0., .800])
    ax.axvline([0.], color='k')
    ax.axvline([.800], color='k')
    ax.set_title(analysis['title'])
    pretty_colorbar(im, ax=ax, ticks=[analysis['chance'], np.max(scores)])

fig.tight_layout()
report.add_figs_to_section([fig], ['all freqs'], 'all')
