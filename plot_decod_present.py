"""Plot control analyses related to the present vs absent decoding"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import wilcoxon
from jr.plot import pretty_decod, pretty_gat, pretty_axes, pretty_colorbar
from jr.utils import table2html
from config import load, report, tois
from conditions import analyses
from base import stats, table_duration

# Test whether decoding of presence varies as a function of contrast and
# visibility

# Gather data
contrast_list = [.5, .75, 1.]  # XXX to config
pas_list = np.arange(4.)

# Plot
results = load('score', analysis='present_anova')
times = results['times']
color_vis = [ana['color'] for ana in analyses
             if ana['title'] == 'Visibility Decision'][0]
color_contrast = [ana['color'] for ana in analyses
                  if ana['title'] == 'Target Contrast'][0]

# Visibility effect
fig, ax = plt.subplots(1, figsize=[6, 2])
pretty_decod([np.diag(ii) for ii in results['R_vis']], times=times,
             sig=np.diag(results['p_vis']) < .05,
             color=color_vis, chance=0., fill=True,)
ax.axvline(.800, color='k')
ax.set_xlabel('Times', labelpad=-10)
ylim = ax.get_ylim()
ax.set_yticklabels(['', '', '%.2f' % ylim[1]])
ax.text(.370, 1.1 * ylim[1], 'Visibility Effect', color=color_vis,
        weight='bold')
ax.set_ylim(ylim[0], 1.3 * ylim[1])
ax.set_ylabel('R', labelpad=-10)
fig.tight_layout()
report.add_figs_to_section(fig, 'visibility', 'R')

# Contrast effect
fig, ax = plt.subplots(1, figsize=[6, 2])
pretty_decod([np.diag(ii) for ii in results['R_contrast']], times=times,
             sig=np.diag(results['p_contrast']) < .05,
             color=color_contrast, chance=0., fill=True,)
ax.axvline(.800, color='k')
ylim = ax.get_ylim()
ax.set_yticklabels(['', '', '%.2f' % ylim[1]])
ax.text(.370, .8 * ylim[1], 'Contrast Effect', color=color_contrast,
        weight='bold')
fig.tight_layout()
ax.set_ylabel('R', labelpad=-10)
ax.set_xlabel('Times', labelpad=-10)
report.add_figs_to_section(fig, 'contrast', 'R')

# AUC for each visibility level: GAT
gats = [('Unseen', results['AUC_pas'][0, :, :, :], .5, [0, 1], 'AUC'),
        ('Seen', results['AUC_pas'][-1, :, :, :], .5, [0, 1], 'AUC'),
        ('Contrast', results['R_contrast'], 0, [-.3, .3], 'R'),
        ('Visibility', results['R_vis'], 0, [-.3, .3], 'R')]
fig, axes = plt.subplots(2, 2, figsize=[8.3, 8.])
axes = np.reshape(axes, -1)
for ii, (name, gat, chance, clim, metric) in enumerate(gats):
    p_val = stats(gat - chance)
    pretty_gat(np.nanmean(gat, axis=0), times=times, chance=chance,
               sig=p_val < .05, ax=axes[ii], clim=clim, colorbar=False)
    axes[ii].axvline(.800, color='k')
    axes[ii].axhline(.800, color='k')
    axes[ii].set_title(name)
ticks = np.arange(-.100, 1.101, .100)
ticklabels = [int(1e3 * ii) if ii in [0, .800] else '' for ii in ticks]
pretty_axes(axes.reshape(2, 2), xlabel='Test Times', ylabel='Train Times',
            xticks=ticks, yticks=ticks,
            xticklabels=ticklabels, yticklabels=ticklabels)
# add colorbars
for ii in [1, 3]:
    pos = axes[ii].get_position()
    clim, metric = gats[ii][-2], gats[ii][-1]
    cax = fig.add_axes([pos.x0 + pos.width + .01, pos.y0, .02, pos.height])
    ticks = [clim[0], np.mean(clim), clim[1]]
    labels = [clim[0], metric, clim[1]]
    pretty_colorbar(ax=axes[ii], cax=cax, ticks=ticks, ticklabels=labels)

report.add_figs_to_section(fig, 'visibility GATs', 'AUC')

# AUC for each visibility level: decoding
fig, ax = plt.subplots(1, figsize=[6, 2])
cmap = mpl.colors.LinearSegmentedColormap.from_list('RdPuBu', ['b', 'r'])
colors = cmap(np.linspace(0., 1., 4))
for ii, (auc, color) in enumerate(zip(results['AUC_pas'][-1::-1, :, :],
                                      colors[-1::-1, :])):
    if ii not in [0, 3]:
        continue
    p_val = stats(auc - .5)
    pretty_decod([np.diag(ii) for ii in auc], times=times, ax=ax, width=1.,
                 alpha=1., chance=.5, color=color, fill=True,
                 sig=np.diag(p_val) < .05)
ax.set_ylim([.45, 1.])
ax.set_yticks([1.])
ax.set_yticklabels([1.])
ax.axvline(.800, color='k')
ax.set_ylabel('AUC', labelpad=-10)
# ax.text(.450, .53, 'Unseen', color='w', weight='bold')
# ax.text(.450, .83, 'Seen', color='r', weight='bold')
ylim = ax.get_ylim()
# ax.text(0, ylim[1], 'Target',  backgroundcolor='w', ha='center', va='top')
# ax.text(.800, ylim[1], 'Probe', backgroundcolor='w', ha='center', va='top')
fig.tight_layout()
report.add_figs_to_section(fig, 'visibility decod', 'AUC')

# Duration analyses for each visibility and each TOI.
# Q: How long do the presence estimators generalize on average?
toi_sel = slice(1, 3)  # only evaluates early and maintenance time windows
results['AUC_pas_duration']  # TOI x PAS x subjects x aligned_test_times
data = results['AUC_pas_duration'][toi_sel, ...]
# ---- define time axis as t0 = train times
times_align = times - times.min()
# ---- setup figure
fig = plt.figure(figsize=[7.8, 5.5])
fig_time_ranges = [.300, .600]  # shorter range for early than late
axes = list()
for ii in range(4):  # one subplot for each visibility rating (PAS)
    axes.append([plt.subplot2grid((4, 3), (ii, 0), colspan=1),
                 plt.subplot2grid((4, 3), (ii, 1), colspan=2)])
cmap = plt.get_cmap('bwr_r')
for jj, (result, toi, time_range) in enumerate(zip(
        data, tois[toi_sel], fig_time_ranges)):
    for ii, col in enumerate(cmap(np.linspace(0, 1, 4.))):
        ax = axes[ii][jj]
        toi_align = np.where((times - times.min()) <= time_range)[0]
        p_val = stats(result[3-ii, :, toi_align-len(toi_align)/2].T - .5)
        pretty_decod(result[3, :, toi_align-len(toi_align)/2].T, ax=ax,
                     times=times_align[toi_align] - time_range/2, color='r',
                     chance=.5)
        pretty_decod(result[3-ii, :, toi_align-len(toi_align)/2].T,
                     color=col, ax=ax, chance=.5,
                     times=times_align[toi_align] - time_range/2, alpha=1.,
                     fill=True, sig=p_val < .05)
        ax.set_yticks([.25, 1.])
        ax.set_yticklabels([.25, 1.])
        ax.set_ylabel('AUC', labelpad=-15)
        ax.set_ylim([.25, 1.])
        xticks = np.arange(-time_range/2., time_range/2.+.01, .100)
        ax.set_xticks(xticks)
        ax.set_xlim(-time_range/2., time_range/2)
        ax.set_xticklabels([''] * len(xticks))
        if jj != 0:
            ax.set_yticklabels(['', ''])
            ax.set_ylabel('')
        if ii == 0:
            ax.set_title('%i $-$ %i ms' % (1e3 * toi[0], 1e3 * toi[1]))

        ax.set_xlabel('', labelpad=-10)
        ax.set_xticklabels([''])
        if ii == 3:
            ax.set_xlabel('Duration', labelpad=-10)
            ax.set_xticklabels(
                [int(x) if np.round(x) in
                 [-time_range/2 * 1e3, time_range/2 * 1e3]
                 else '' for x in np.round(1e3 * xticks)])
fig.tight_layout()
report.add_figs_to_section(fig, 'duration', 'duration')

# --- Repeat this duration figure for max seen and min unseen only
times_align = times - times.min()
fig = plt.figure(figsize=[7.8, 2])
early = dict(unseen=results['AUC_pas_duration'][1, 0, ...],
             seen=results['AUC_pas_duration'][1, -1, ...], toi=tois[1],
             tlim=.300, ax=plt.subplot2grid((1, 3), (0, 0), colspan=1))
late = dict(unseen=results['AUC_pas_duration'][2, 0, ...],
            seen=results['AUC_pas_duration'][2, -1, ...], toi=tois[2],
            tlim=.600, ax=plt.subplot2grid((1, 3), (0, 1), colspan=2))
for data_toi in [early, late]:
    toi_align = np.where((times - times.min()) <= data_toi['tlim'])[0]
    chance = .5
    ax = data_toi['ax']
    # seen
    duration = data_toi['seen'][:, toi_align-len(toi_align)/2]
    p_val = stats(duration - chance)
    pretty_decod(duration, color='r', ax=ax, chance=chance,
                 times=times_align[toi_align] - data_toi['tlim']/2,
                 alpha=1., fill=True, sig=p_val < .05)
    # unseen
    duration = data_toi['unseen'][:, toi_align-len(toi_align)/2]
    p_val = stats(duration - chance)
    pretty_decod(duration, color='b', ax=ax, chance=chance,
                 times=times_align[toi_align] - data_toi['tlim']/2,
                 alpha=1., fill=True, sig=p_val < .05)
    # plotting
    ax.set_yticks([.25, 1.])
    ax.set_yticklabels([.25, 1.])
    ax.set_ylabel('AUC', labelpad=-15)
    ax.set_ylim([.25, 1.])
    ax.set_xticks(np.arange(-.600, .600, .100))
    ax.set_xticklabels([ii if ii != 0 else ''
                        for ii in np.arange(-600, 600, 100)])
    ax.set_xlim(-data_toi['tlim']/2, data_toi['tlim']/2)
    ax.set_title('%i $-$ %i ms' % (1e3 * data_toi['toi'][0],
                 1e3 * data_toi['toi'][1]))
    ax.set_xlabel('Duration', labelpad=-10)
fig.tight_layout()
report.add_figs_to_section(fig, 'duration small', 'duration')

# add report to Table: Duration
# AUC_pas_duration: shape(TOI x PAS x subjects x aligned_test_times)
data_ = results['AUC_pas_duration'][toi_sel, :, :, :]
# table_duration() computes the mean duration of early and late estiamtors
# across subjects and outputs an HTML table
table = table_duration(data=data_, tois=tois[toi_sel], times=times, chance=.5)
report.add_htmls_to_section(table, 'duration', 'table')

# Table report: AUC
table = np.empty((4, len(tois)), dtype=object)
for pas in range(4):
    score = results['AUC_pas'][pas, :, :]
    p_val = stats(score - .5)
    for jj, toi in enumerate(tois):
        toi_ = np.where((times >= toi[0]) & (times < toi[1]))[0]
        score_ = np.mean(score[:, toi_], axis=1)
        table[pas, jj] = '[%.3f+/-%.3f, p=%.4f]' % (
            np.nanmean(score_), np.nanstd(score_) / np.sqrt(len(score_)),
            np.min(p_val[toi_]))
table = table2html(table, head_column=tois,
                   head_line=['pas%i' % pas for pas in range(4)])
report.add_htmls_to_section(table, 'AUC', 'table')

# Table report: R: modulation of present score by contrast and visibility
table = np.empty((3, len(tois)), dtype=object)
for ii, key in enumerate(['contrast', 'vis']):
    R = results['R_%s' % key]
    p_val = results['p_%s' % key]
    for jj, toi in enumerate(tois):
        toi_ = np.where((times >= toi[0]) & (times < toi[1]))[0]
        R_ = np.mean(R[:, toi_], axis=1)
        table[ii, jj] = '[%.3f+/-%.3f, p=%.4f]' % (
            np.nanmean(R_), np.nanstd(R_) / np.sqrt(len(R_)),
            np.min(p_val[toi_]))

# add interaction visibility & contrast modulation of decoding score
R = results['R_contrast'] - results['R_vis']
p_val = stats(R)
for jj, toi in enumerate(tois):
    toi_ = np.where((times >= toi[0]) & (times < toi[1]))[0]
    R_ = np.mean(R[:, toi_], axis=1)
    table[2, jj] = '[%.3f+/-%.3f, p=%.4f]' % (
        np.nanmean(R_), np.nanstd(R_) / np.sqrt(len(R_)),
        np.min(p_val[toi_]))
table = table2html(table, head_column=tois,
                   head_line=['R_contrast', 'R_vis', 'diff'])
report.add_htmls_to_section(table, 'R', 'table')

# Is the modulation of contrast different between early versus delay TOI?
t_baseline, t_early, t_delay, t_probe = [
    np.where((times >= toi[0]) & (times < toi[1]))[0] for toi in tois]
R_early = np.mean([np.diag(G)[t_early] for G in results['R_contrast']], axis=1)
R_delay = np.mean([np.diag(G)[t_delay] for G in results['R_contrast']], axis=1)
wilcoxon(R_early - R_delay)

report.save()
