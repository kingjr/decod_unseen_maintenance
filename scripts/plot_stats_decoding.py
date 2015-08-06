import sys
sys.path.insert(0, './')
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle
import numpy as np
from meeg_preprocessing.utils import setup_provenance
from base import plot_sem, plot_widths

from scripts.config import (
    paths,
    open_browser,
    analyses
)


report, run_id, _, logger = setup_provenance(
    script='scripts/_plot_stats_decoding.py', results_dir=paths('report'))


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


def minmax(data, chance=None, born=99):
    if chance is None:
        spread = np.round(np.percentile(
            data, [100 - born, born]) * 1e2) / 1e2
        m = np.mean(spread)
        spread = np.ptp(spread)
    else:
        spread = 2 * np.round(np.percentile(
            np.abs(data - chance), born) * 1e2) / 1e2
        m = chance
    ymin, ymax = m + spread * np.array([-.6, .6])
    return ymin, ymax


def plot_gat(ax, scores, p_values, chance, times, alpha=.05):
    scores = np.mean(scores, axis=0)
    ymin, ymax = minmax(scores, chance)
    im = ax.imshow(scores,
                   extent=[min(times), max(times)] * 2,
                   cmap='RdBu_r', origin='lower', vmin=ymin, vmax=ymax)
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    ax.contour(xx, yy, p_values < alpha, colors='black', levels=[0],
               linestyles='dotted')
    ax.axhline(0, color='k')
    ax.axvline(0, color='k')
    ax.axhline(800, color='k')
    ax.axvline(800, color='k')
    ax.set_xlabel('Test times (ms.)')
    ax.set_ylabel('Train times (ms.)')
    cb = plt.colorbar(im, ax=ax, ticks=[ymin, chance, ymax])
    cb.ax.set_yticklabels(['%.2f' % ymin, 'Chance', '%.2f' % ymax],
                          color='dimgray')
    cb.ax.xaxis.label.set_color('dimgray')
    cb.ax.yaxis.label.set_color('dimgray')
    cb.ax.spines['left'].set_color('dimgray')
    cb.ax.spines['right'].set_color('dimgray')
    pretty_plot(ax)


def plot_diag(ax, scores, p_values, chance, times, width=3., alpha=.05,
              color='k', fill=False):
    sem = scores.std(0) / np.sqrt(len(scores))
    ymin, ymax = minmax([scores.mean(0) + sem, scores.mean(0) - sem])
    widths = width * (p_values < alpha)
    plot_sem(times, scores, color=color, ax=ax)
    ax.set_ylim(ymin, ymax)
    plot_widths(times, scores.mean(0), widths, ax=ax, color=color)
    if fill:
        scores_sig = (scores.mean(0) * (p_values > .05) +
                      chance * (p_values < .05))
        ax.fill_between(times, scores.mean(0), scores_sig, color=color,
                        alpha=.75, linewidth=0)

    ax.axhline(chance, linestyle='dotted', color='k', zorder=-3)
    ax.axvline(0, color='k', zorder=-3)
    ax.axvline(800, color='k', zorder=-3)
    ax.set_xlabel('Times (ms.)')
    ax.set_xlim(np.min(times), np.max(times))
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([ymin, chance, ymax])
    ax.set_yticklabels(['%.2f' % ymin, 'Chance', '%.2f' % ymax])
    pretty_plot(ax)

fig_alldiag, axes_alldiag = plt.subplots(len(analyses), 1, figsize=[8, 12])
cmap = plt.get_cmap('gist_rainbow')
colors = cmap(np.linspace(0, 1., len(analyses) + 1))

for analysis, ax_alldiag, color in zip(analyses, axes_alldiag, colors):
    print analysis['name']
    # Load
    stats_fname = paths('score', subject='fsaverage', data_type='erf',
                        analysis=('stats_' + analysis['name']))
    with open(stats_fname, 'rb') as f:
        out = pickle.load(f)
        scores = out['scores']
        p_values = out['p_values']
        p_values_off = out['p_values_off']
        p_values_diag = np.squeeze(out['p_values_diag'])
        times = out['times']

    # Parameters
    chance = analysis['chance']
    alpha = 0.05
    scores_diag = np.array([np.diag(score) for score in scores])
    diag_offdiag = scores - np.tile([np.diag(score) for score in scores],
                                    [len(times), 1, 1]).transpose(1, 0, 2)

    # Plots
    fig_gat, ax = plt.subplots(1, figsize=[7, 5.5])
    plot_gat(ax, scores, p_values, chance, times, alpha=alpha)

    fig_diag, ax = plt.subplots(1, figsize=[5, 2.5])
    plot_diag(ax, scores_diag, p_values_diag, chance, times, width=3,
              color=color, fill=True)
    plot_diag(ax, scores_diag, p_values_diag, chance, times, width=3,
              color='k')
    ax.text(0, ax.get_ylim()[1], 'Target',  backgroundcolor='w',
            ha='center', va='top')
    ax.text(800, ax.get_ylim()[1], 'Probe', backgroundcolor='w',
            ha='center', va='top')

    ax = ax_alldiag
    plot_diag(ax, scores_diag, p_values_diag, chance, times, width=3, color=color, fill=True)
    plot_diag(ax, scores_diag, p_values_diag, chance, times, width=3, color='k', fill=False)
    if ax != axes_alldiag[-1]:
        ax.set_xlabel('')
        ax.set_xticklabels([])
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_yticks([chance, ylim[1]])
    ax.set_yticklabels(['chance', '%.2f' % ylim[1]])
    if chance == .5:
        ax.set_ylabel('AUC')
    else:
        ax.set_ylabel('R')
    if ax == axes_alldiag[0]:
        ax.text(0, ylim[1], 'Target', backgroundcolor='w', ha='center', va='top')
        ax.text(800, ylim[1], 'Probe', backgroundcolor='w', ha='center', va='top')
    # FIXME
    names = dict(target_present='Target Present',
                 target_contrast_pst='Target Contrast',
                 target_spatialFreq='Target Spatial Frequency',
                 target_circAngle='Target Angle',
                 detect_button_pst='Visibility Response',
                 probe_circAngle='Probe Angle',
                 probe_tilt='Target - Probe Tilt',
                 discrim_button='Tilt Discrimination Response')
    name = names[analysis['name']]
    txt = ax.text(xlim[0] + .5 * np.ptp(xlim), ylim[0] + .75 * np.ptp(ylim),
                  name, color=.75 * color, ha='center', weight='bold')

    # ------ Plot times slices score
    plot_times = np.arange(125, 1125, 200)
    fig_offdiag, axs = plt.subplots(len(plot_times), 1, figsize=[5, 6])
    ylim = minmax(np.mean(scores, axis=0), chance=chance, born=100)
    for sel_time, ax in zip(plot_times, reversed(axs)):
        idx = np.argmin(abs(times-sel_time))
        scores_off = np.array(scores)[:, idx, :]
        p_values_off = p_values[idx, :]
        plot_diag(ax, scores_off, p_values_off, chance, times, width=3, color='b')
        plot_diag(ax, scores_diag, p_values_diag, chance, times, width=3, color='k')
        scores_sig = (scores_diag.mean(0) * (p_values_off[idx] > .05) +
                      scores_off.mean(0) * (p_values_off[idx] < .05))
        ax.fill_between(times, scores_diag.mean(0), scores_sig, color='yellow',
                        alpha=.5, linewidth=0)
        ax.set_ylim(ylim)
        ax.set_yticks([ylim[0], chance, ylim[1]])
        ax.set_yticklabels(['%.2f' % ylim[0], '', '%.2f' % ylim[1]])
        ymin, ymax = ax.get_ylim()
        ax.plot([sel_time] * 2, [ymin, scores_off.mean(0)[idx]], color='b',
                zorder=-2)
        ax.text(sel_time, ymin, '%i ms.' % sel_time,
                color='b', backgroundcolor='w', ha='center', zorder=-1)
        pretty_plot(ax)
        if ax != axs[-1]:
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.spines['bottom'].set_visible(False)

    # Plot mean prediction # XXX
    # Report
    report.add_figs_to_section([fig_diag, fig_gat, fig_offdiag],
                               [analysis['name']] * 3, analysis['name'])
report.add_figs_to_section(fig_alldiag, 'all', 'all')
report.save(open_browser=open_browser)
# upload_report(report)
