import matplotlib.pyplot as plt
import pickle
import numpy as np
from jr.plot import pretty_gat, pretty_decod, pretty_slices
from jr.utils import table2html
from scipy.stats import wilcoxon
from scripts.config import paths, analyses, report, tois

fig_alldiag, axes_alldiag = plt.subplots(len(analyses), 1, figsize=[6, 9])

table_toi = np.empty((len(analyses), len(tois)), dtype=object)
table_reversal = np.empty((len(analyses), 2), dtype=object)
figs = list()
for ii, (analysis, ax_diag) in enumerate(zip(analyses, axes_alldiag)):
    print analysis['name']
    # Load
    stats_fname = paths('score', subject='fsaverage', data_type='erf',
                        analysis=('stats_' + analysis['name']))
    with open(stats_fname, 'rb') as f:
        out = pickle.load(f)
        scores = np.array(out['scores'])
        p_values = out['p_values']
        p_values_off = out['p_values_off']
        p_values_diag = np.squeeze(out['p_values_diag'])
        times = out['times'] / 1e3  # FIXME

    if 'Angle' in analysis['title']:
        scores /= 2.  # from circle to half circle

    # Parameters
    chance = analysis['chance']
    scores_diag = np.array([np.diag(score) for score in scores])
    diag_offdiag = scores - np.tile([np.diag(score) for score in scores],
                                    [len(times), 1, 1]).transpose(1, 0, 2)
    slices_tois = np.arange(.100, 1.100, .200)

    # GAT
    clim = np.percentile(np.diag(np.mean(scores, axis=0)), 97)
    clim = [chance-(clim-chance), clim]
    fig_gat, ax_gat = plt.subplots(1, figsize=[7, 5.5])
    pretty_gat(np.mean(scores, axis=0), times=times, sig=p_values < .05,
               chance=chance, ax=ax_gat, clim=clim)
    # for toi in slices_tois:
    #     ax_gat.axhline(toi, color='b')
    ax_gat.axvline(.800, color='k')
    ax_gat.axhline(.800, color='k')
    ax_gat.set_xlabel('Test Times', labelpad=-10)
    ax_gat.set_ylabel('Train Times', labelpad=-15)
    report.add_figs_to_section(fig_gat, 'gat', analysis['name'])

    fig_, ax = plt.subplots(1)
    ax.matshow(np.mean(scores, axis=0), origin='lower',
               extent=[np.min(times), np.max(times)] * 2)
    ax.set_xticks(np.arange(0., 1.200, .01))
    ax.set_yticks(np.arange(0., 1.200, .01))
    figs.append(fig_)

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

    # ------ Plot times slices score
    fig_offdiag, axs = plt.subplots(len(slices_tois), 1, figsize=[5, 6])
    pretty_slices(scores, times=times, chance=chance, axes=axs,
                  sig=p_values < .05, sig_diagoff=p_values_off < .05,
                  colors=[analysis['color'], 'b'], tois=slices_tois,
                  fill_color=analysis['color'])
    for ax in axs:
        ax.axvline(.800, color='k')
        if analysis['typ'] == 'regress':
            ax.set_ylabel('R', labelpad=-15)
        elif analysis['typ'] == 'categorize':
            ax.set_ylabel('AUC', labelpad=-15)
        else:
            ax.set_ylabel('rad.', labelpad=-15)
        ax.set_yticklabels(['', '', '%.2f' % ax.get_yticks()[2]])
    ax.set_xlabel('Times', labelpad=-10)
    report.add_figs_to_section(fig_offdiag, 'slices', analysis['name'])

    # Decod
    pretty_decod(scores_diag, times=times, sig=p_values_diag < .05,
                 chance=chance, color=analysis['color'], fill=True, ax=ax_diag)
    xlim, ylim = ax_diag.get_xlim(), np.array(ax_diag.get_ylim())
    ylim[1] = np.ceil(ylim[1] * 10) / 10.
    ax_diag.set_ylim(ylim)
    ax_diag.axvline(.800, color='k')
    if ax_diag != axes_alldiag[-1]:
        ax_diag.set_xlabel('')
        ax_diag.set_xticklabels([])
    elif ax_diag == axes_alldiag[0]:
        ax_diag.text(0, ylim[1], 'Target',  backgroundcolor='w', ha='center',
                     va='top')
        ax_diag.text(.800, ylim[1], 'Probe', backgroundcolor='w', ha='center',
                     va='top')
        ax_diag.set_xticklabels([int(x) if x in [0., 800.] else '' for x in
                                 np.round(1e3 * ax_diag.get_xticks())])
    ax_diag.set_yticks([chance, ylim[1]])
    ax_diag.set_yticklabels(['', '%.1f' % ylim[1]])
    if analysis['typ'] == 'regress':
        ax_diag.set_ylabel('R', labelpad=-15)
    elif analysis['typ'] == 'categorize':
        ax_diag.set_ylabel('AUC', labelpad=-15)
    else:
        ax_diag.set_ylabel('rad.', labelpad=-15)
    txt = ax_diag.text(xlim[0] + .5 * np.ptp(xlim),
                       ylim[0] + .75 * np.ptp(ylim),
                       analysis['title'], color=.75 * analysis['color'],
                       ha='center', weight='bold')

    # Add reversal score to table_toi
    toi_reversal = np.array([.12, .180])
    if 'Probe' in analysis['title']:
        toi_reversal += .817
    toi_ = [np.where(times >= toi_reversal[0])[0][0],
            np.where(times >= toi_reversal[1])[0][0]]
    # -- is there a reversal?
    score = (scores[:, toi_[0], toi_[1]] + scores[:, toi_[1], toi_[0]]) / 2.
    p_val = wilcoxon(score - analysis['chance'])[1]
    table_reversal[ii, 0] = '[%.3f+/-%.3f, p=%.4f]' % (
        np.nanmean(score), np.nanstd(score) / np.sqrt(len(score)), p_val)
    # -- is the reversal complete?
    score = (scores[:, toi_[0], toi_[1]] + scores[:, toi_[1], toi_[0]] -
             scores[:, toi_[0], toi_[0]] - scores[:, toi_[1], toi_[1]])
    p_val = wilcoxon(score - analysis['chance'])[1]
    table_reversal[ii, 1] = '[%.3f+/-%.3f, p=%.4f]' % (
        np.nanmean(score), np.nanstd(score) / np.sqrt(len(score)), p_val)

    # Add TOI diag score to table_toi
    for jj, toi in enumerate(tois):
        toi = np.where((times >= toi[0]) & (times < toi[1]))[0]
        score = np.nanmean(scores_diag[:, toi], axis=1)
        table_toi[ii, jj] = '[%.3f+/-%.3f, p=%.4f]' % (
            np.nanmean(score), np.nanstd(score) / np.sqrt(len(score)),
            np.median(p_values_diag[toi]))

fig_alldiag.tight_layout()
report.add_figs_to_section(fig_alldiag, 'diagonal', 'all')

table_toi = table2html(table_toi, head_line=tois,
                       head_column=[a['title'] for a in analyses])
report.add_htmls_to_section(table_toi, 'table_toi', 'all')

table_reversal = table2html(table_reversal, head_line=toi_reversal,
                            head_column=[a['title'] for a in analyses])
report.add_htmls_to_section(table_reversal, 'table_reversal', 'all')

report.save()
