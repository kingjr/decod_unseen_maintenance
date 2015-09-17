import matplotlib.pyplot as plt
import pickle
import numpy as np
from jr.plot import pretty_gat, pretty_decod, pretty_slices
from scripts.config import paths, analyses, report

fig_alldiag, axes_alldiag = plt.subplots(len(analyses), 1, figsize=[6, 9])
cmap = plt.get_cmap('gist_rainbow')
colors = cmap(np.linspace(0, 1., len(analyses) + 1))

for analysis, ax, color in zip(analyses, axes_alldiag, colors):
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
        times = out['times'] / 1e3  # FIXME

    # Parameters
    chance = analysis['chance']
    alpha = 0.05
    scores_diag = np.array([np.diag(score) for score in scores])
    diag_offdiag = scores - np.tile([np.diag(score) for score in scores],
                                    [len(times), 1, 1]).transpose(1, 0, 2)

    # GAT
    clim = np.percentile(np.diag(np.mean(scores, axis=0)), 97)
    clim = [chance-(clim-chance), clim]
    fig_gat, ax_gat = plt.subplots(1, figsize=[7, 5.5])
    pretty_gat(np.mean(scores, axis=0), times=times, sig=p_values < .05,
               chance=chance, ax=ax_gat, clim=clim)
    report.add_figs_to_section(fig_gat, 'gat', analysis['name'])

    # Decod
    pretty_decod(scores_diag, times=times, sig=p_values_diag < .05,
                 chance=chance, color=color, fill=True, ax=ax)
    pretty_decod(scores_diag, times=times, sig=p_values_diag < .05,
                 chance=chance, color='k', fill=False, ax=ax)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.axvline(.800, color='k')
    if ax != axes_alldiag[-1]:
        ax.set_xlabel('')
        ax.set_xticklabels([])
    elif ax == axes_alldiag[-1]:
        ax.text(0, ylim[1], 'Target',  backgroundcolor='w', ha='center',
                va='top')
        ax.text(.800, ylim[1], 'Probe', backgroundcolor='w', ha='center',
                va='top')
    ax.set_yticks([chance, ylim[1]])
    ax.set_yticklabels(['', '%.2f' % ylim[1]])
    if chance == .5:
        ax.set_ylabel('AUC', labelpad=-10)
    else:
        ax.set_ylabel('R', labelpad=-10)
    txt = ax.text(xlim[0] + .5 * np.ptp(xlim), ylim[0] + .75 * np.ptp(ylim),
                  analysis['title'], color=.75 * color, ha='center',
                  weight='bold')

    # ------ Plot times slices score
    tois = np.arange(.150, 1.150, .200)
    fig_offdiag, axs = plt.subplots(len(tois), 1, figsize=[5, 6])
    pretty_slices(scores, times=times, chance=chance, axes=axs, tois=tois,
                  sig_diagoff=p_values_off < .05)
    report.add_figs_to_section(fig_offdiag, 'slices', analysis['name'])
fig_alldiag.tight_layout()
report.add_figs_to_section(fig_alldiag, 'diagonal', 'all')
report.save()
