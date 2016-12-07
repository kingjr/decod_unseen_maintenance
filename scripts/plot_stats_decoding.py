"""Plot decoding and Temporal Generalization (TG) results

Used to generate Figures 3 & S6.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from jr.plot import pretty_gat, pretty_decod, pretty_slices
from jr.utils import table2html
from scipy.stats import wilcoxon
from config import load, report
from base import stats
from conditions import analyses, tois

fig_alldiag = plt.figure(figsize=[6.5, 11])
axes_alldiag = gridspec.GridSpec(len(analyses), 1, hspace=0.1)

table_toi = np.empty((len(analyses), len(tois)), dtype=object)
table_reversal = np.empty((len(analyses), 2), dtype=object)
figs = list()
alpha = 0.05  # statistical threshold for line delimitation

# Plot diagonal decoding and temporal generalization for each analysis
for ii, (analysis, ax_diag) in enumerate(zip(analyses, axes_alldiag)):
    print analysis['name']
    # Load
    out = load('score', subject='fsaverage',
               analysis=('stats_' + analysis['name']))
    scores = np.array(out['scores'])
    p_values = out['p_values']
    p_values_off = out['p_values_off']
    p_values_diag = np.squeeze(out['p_values_diag'])
    times = out['times']

    if 'Angle' in analysis['title']:
        scores /= 2.  # from circle to half circle

    # Parameters
    chance = analysis['chance']
    scores_diag = np.array([np.diag(score) for score in scores])
    diag_offdiag = scores - np.tile([np.diag(score) for score in scores],
                                    [len(times), 1, 1]).transpose(1, 0, 2)
    slices_tois = np.arange(.100, 1.100, .200)

    # Temporal generalization matrices
    clim = np.percentile(np.diag(np.mean(scores, axis=0)), 97)
    clim = [chance-(clim-chance), clim]
    fig_gat, ax_gat = plt.subplots(1, figsize=[7, 5.5])
    pretty_gat(np.mean(scores, axis=0), times=times, sig=p_values < alpha,
               chance=chance, ax=ax_gat, clim=clim)
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

    # ------ Plot times slices score
    fig_offdiag, axs = plt.subplots(len(slices_tois), 1, figsize=[5, 6])
    pretty_slices(scores, times=times, chance=chance, axes=axs,
                  sig=p_values < alpha, sig_diagoff=p_values_off < alpha,
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
    ax_diag = fig_alldiag.add_subplot(ax_diag)
    pretty_decod(scores_diag, times=times, sig=p_values_diag < alpha,
                 chance=chance, color=analysis['color'], fill=True, ax=ax_diag)
    xlim, ylim = ax_diag.get_xlim(), np.array(ax_diag.get_ylim())
    # ylim[1] = np.ceil(ylim[1] * 10) / 10.
    sem = scores_diag.std(0) / np.sqrt(len(scores_diag))
    ylim = [np.min(scores_diag.mean(0) - sem),
            np.max(scores_diag.mean(0) + sem)]
    ax_diag.set_ylim(ylim)
    ax_diag.axvline(.800, color='k')
    ax_diag.set_xticklabels([int(x) if x in np.linspace(0., 1000., 11) else ''
                             for x in np.round(1e3 * ax_diag.get_xticks())])
    if ax_diag != axes_alldiag[-1]:
        # bottom suplot
        ax_diag.set_xlabel('')
        ax_diag.set_xticklabels([])
        ax_diag.xaxis.set_visible(False)
    elif ax_diag == axes_alldiag[0]:
        # top subplot
        ax_diag.text(0, ylim[1], 'Target',  backgroundcolor='w', ha='center',
                     va='top')
        ax_diag.text(.800, ylim[1], 'Probe', backgroundcolor='w', ha='center',
                     va='top')
    ax_diag.set_yticks([chance, ylim[1]])
    ax_diag.set_yticklabels(['', '%.2f' % ylim[1]])
    if analysis['typ'] == 'regress':
        ax_diag.set_ylabel('R', labelpad=-15)
    elif analysis['typ'] == 'categorize':
        ax_diag.set_ylabel('AUC', labelpad=-15)
    else:
        ax_diag.set_ylabel('rad.', labelpad=-15)
    txt = ax_diag.text(xlim[0] + .5 * np.ptp(xlim),
                       ylim[0] + .75 * np.ptp(ylim),
                       analysis['title'], color=[.2, .2, .2],
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

report.add_figs_to_section(fig_alldiag, 'diagonal', 'all')

table_toi = table2html(table_toi.T, head_line=[str(t) for t in tois],
                       head_column=[a['title'] for a in analyses])
report.add_htmls_to_section(table_toi, 'table_toi', 'all')

table_reversal = table2html(table_reversal.T, head_line=toi_reversal,
                            head_column=[a['title'] for a in analyses])
report.add_htmls_to_section(table_reversal, 'table_reversal', 'all')

# Report main effect of task relevance
score_relevant, score_irrelevant = list(), list()
relevant = ['target_circAngle', 'target_present', 'detect_button_pst']
irrelevant = ['target_contrast_pst', 'target_spatialFreq', 'target_phase']
for ii, (analysis, ax_diag) in enumerate(zip(analyses, axes_alldiag)):
    print analysis['name']
    # Load
    if analysis['name'] not in relevant + irrelevant:
        continue
    out = load('score', subject='fsaverage',
               analysis=('stats_' + analysis['name']))
    # we'll be looking at the sign of the effect per subjects as compared
    # to chance level
    scores = np.array(out['scores']) - analysis['chance']
    # XXX you need to relaunch one of the analyses that only has 153 and
    # not 154 time points
    scores = np.array([np.diag(subject) for subject in scores])
    if analysis['name'] in relevant:
        score_relevant.append(scores)
    elif analysis['name'] in irrelevant:
        score_irrelevant.append(scores)

fig, ax = plt.subplots(1, figsize=[6, 2])
# non parametric necessitate to take the mean of the sign of the effects
# because each decoding score uses different metrics
scores_interaction = np.mean(np.sign(score_relevant) -
                             np.sign(score_irrelevant), axis=0)
scores_relevant = np.mean(np.sign(score_relevant), axis=0)
scores_irrelevant = np.mean(np.sign(score_irrelevant), axis=0)

sig = stats(scores_interaction) < alpha
pretty_decod(scores_relevant, times=times, ax=ax, color='y', sig=sig,
             fill=True, width=0.)
sig = stats(scores_relevant) < alpha
pretty_decod(scores_relevant, times=times, ax=ax, color='r', sig=sig)
sig = stats(scores_irrelevant) < alpha
pretty_decod(scores_irrelevant, times=times, ax=ax, color='w',
             sig=np.ones_like(times), fill=True, width=0.)
pretty_decod(scores_irrelevant, times=times, ax=ax, color='k', sig=sig)
ax.set_ylim(-1., 1.)
ax.set_yticks([-1., 1])
ax.set_yticklabels([-1, 1])
xticks = np.arange(-.100, 1.101, .100)
ax.set_xticks(xticks)
ax.set_xticklabels([int(1e3*ii) if ii in np.linspace(-0.1, 1., 12.)
                    else '' for ii in xticks])
ax.axvline(.800, color='k')
report.add_figs_to_section(fig, 'interaction relevance', 'all')

report.save()
