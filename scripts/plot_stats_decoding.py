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
    scores_diag = np.array([np.diag(score) for score in scores])
    diag_offdiag = scores - np.tile([np.diag(score) for score in scores],
                                    [len(times), 1, 1]).transpose(1, 0, 2)
    tois = np.arange(.100, 1.100, .200)

    # GAT
    clim = np.percentile(np.diag(np.mean(scores, axis=0)), 97)
    clim = [chance-(clim-chance), clim]
    fig_gat, ax_gat = plt.subplots(1, figsize=[7, 5.5])
    pretty_gat(np.mean(scores, axis=0), times=times, sig=p_values < .05,
               chance=chance, ax=ax_gat, clim=clim)
    # for toi in tois:
    #     ax_gat.axhline(toi, color='b')
    ax_gat.axvline(.800, color='k')
    ax_gat.axhline(.800, color='k')
    ax_gat.set_xlabel('Test Times', labelpad=-10)
    ax_gat.set_ylabel('Train Times', labelpad=-15)
    report.add_figs_to_section(fig_gat, 'gat', analysis['name'])

    # Small GAT
    clim = np.percentile(np.diag(np.mean(scores, axis=0)), 97)
    clim = [chance-(clim-chance), clim]
    fig_gat_small, ax_gat = plt.subplots(1, figsize=[3.5, 2.5])
    pretty_gat(np.mean(scores, axis=0), times=times, sig=p_values < .05,
               chance=chance, ax=ax_gat, clim=clim)
    ax_gat.axvline(.800, color='k')
    ax_gat.axhline(.800, color='k')
    ax_gat.set_xlabel('Test Times', labelpad=-10)
    ax_gat.set_ylabel('Train Times', labelpad=-15)
    ax_gat.set_xlim(-.050, .350)
    ax_gat.set_ylim(-.050, .350)
    ax_gat.set_yticks(np.arange(0, .301, .100))
    ax_gat.set_yticklabels([0, '', '', 300])
    ax_gat.set_xticks(np.arange(0, .301, .100))
    ax_gat.set_xticklabels([0, '', '', 300])
    if 'probe' in analysis['name']:
        ax_gat.set_xlim(-.050 + .800, .350 + .800)
        ax_gat.set_ylim(-.050 + .800, .350 + .800)
        ax_gat.set_yticks(np.arange(.800, 1.101, .100))
        ax_gat.set_yticklabels([800, '', '', 1100])
        ax_gat.set_xticks(np.arange(.800, 1.101, .100))
        ax_gat.set_xticklabels([800, '', '', 1100])
    fig_gat_small.tight_layout()
    report.add_figs_to_section(fig_gat_small, 'gat_small', analysis['name'])

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
    elif ax == axes_alldiag[0]:
        ax.text(0, ylim[1], 'Target',  backgroundcolor='w', ha='center',
                va='top')
        ax.text(.800, ylim[1], 'Probe', backgroundcolor='w', ha='center',
                va='top')
    ax.set_yticks([chance, ylim[1]])
    ax.set_yticklabels(['', '%.2f' % ylim[1]])
    if chance == .5:
        ax.set_ylabel('AUC', labelpad=-15)
    else:
        ax.set_ylabel('R', labelpad=-15)
    txt = ax.text(xlim[0] + .5 * np.ptp(xlim), ylim[0] + .75 * np.ptp(ylim),
                  analysis['title'], color=.75 * color, ha='center',
                  weight='bold')

    # ------ Plot times slices score
    fig_offdiag, axs = plt.subplots(len(tois), 1, figsize=[5, 6])
    pretty_slices(scores, times=times, chance=chance, axes=axs, tois=tois,
                  sig=p_values < .05, sig_diagoff=p_values_off < .05,
                  colors=[analysis['color'], 'b'],
                  fill_color=analysis['color'])
    for ax in axs:
        ax.axvline(.800, color='k')
        ax.set_ylabel('R', labelpad=-15)
        if chance == .5:
            ax.set_ylabel('AUC', labelpad=-15)
        ax.set_yticklabels(['', '', '%.2f' % ax.get_yticks()[2]])
    ax.set_xlabel('Times', labelpad=-10)
    report.add_figs_to_section(fig_offdiag, 'slices', analysis['name'])
fig_alldiag.tight_layout()
report.add_figs_to_section(fig_alldiag, 'diagonal', 'all')
report.save()
