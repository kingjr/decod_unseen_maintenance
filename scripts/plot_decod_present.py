"""Plot control analyses related to the present vs absent decoding"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import wilcoxon
from jr.plot import pretty_decod
from jr.utils import table2html
from scripts.config import paths, subjects, report, analyses, tois
from scripts.base import stats

# Test whether decoding of presence varies as a function of contrast and
# visibility

# Gather data
n_times = 154  # XXX
contrast_list = [.5, .75, 1.]
pas_list = np.arange(4.)
results = dict(
    data=np.nan * np.zeros((len(subjects), n_times, 4, 3)),
    R_vis=np.nan * np.zeros((len(subjects), n_times)),
    R_contrast=np.nan * np.zeros((len(subjects), n_times)),
    AUC_pas=np.nan * np.zeros((4, len(subjects), n_times)),
    AUC_pas_duration=np.nan * np.zeros((len(tois), len(pas_list),
                                        len(subjects), n_times))
)

# Plot
fname = paths('score', analysis='present_anova')
with open(fname, 'rb') as f:
    results = pickle.load(f)
times = results['times']
color_vis = [ana['color'] for ana in analyses
             if ana['title'] == 'Visibility Response'][0]
color_contrast = [ana['color'] for ana in analyses
                  if ana['title'] == 'Target Contrast'][0]

# Visibility effect
fig, ax = plt.subplots(1, figsize=[6, 2])
pretty_decod(results['R_vis'], times=times, sig=results['p_vis'] < .05,
             color=color_vis, chance=0., fill=True)
ax.axvline(.800, color='k')
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
pretty_decod(results['R_contrast'], times=times, fill=True,
             sig=results['p_contrast'] < .05, color=color_contrast, chance=0.)
ax.axvline(.800, color='k')
ylim = ax.get_ylim()
ax.set_yticklabels(['', '', '%.2f' % ylim[1]])
ax.text(.370, .8 * ylim[1], 'Contrast Effect', color=color_contrast,
        weight='bold')
fig.tight_layout()
ax.set_ylabel('R', labelpad=-10)
ax.set_xlabel('Times', labelpad=-10)
report.add_figs_to_section(fig, 'contrast', 'R')

# AUC for each visibility level
fig, ax = plt.subplots(1, figsize=[6, 2])
cmap = mpl.colors.LinearSegmentedColormap.from_list('RdPuBu', ['b', 'r'])
colors = cmap(np.linspace(0., 1., 4))
for ii, (auc, color) in enumerate(zip(results['AUC_pas'][-1::-1, :, :],
                                      colors[-1::-1, :])):
    if ii not in [0, 3]:
        continue
    p_val = stats(auc[:, :, None] - .5)
    pretty_decod(auc, times=times, ax=ax, width=1., alpha=1.,
                 chance=.5, color=color, fill=True, sig=p_val < .05)
ax.set_ylim([.45, 1.])
ax.set_yticks([1.])
ax.set_yticklabels([1.])
ax.axvline(.800, color='k')
ax.set_ylabel('AUC', labelpad=-10)
ax.text(.450, .53, 'Unseen', color='w', weight='bold')
ax.text(.450, .83, 'Seen', color='r', weight='bold')
ylim = ax.get_ylim()
ax.text(0, ylim[1], 'Target',  backgroundcolor='w', ha='center', va='top')
ax.text(.800, ylim[1], 'Probe', backgroundcolor='w', ha='center', va='top')
fig.tight_layout()
report.add_figs_to_section(fig, 'visibility', 'AUC')

# Duration for each visibility and TOI
data = results['AUC_pas_duration'][1:-1, ...]  # activation & maintenance TOI
times_align = times - times.min()
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
        p_val = stats(result[3-ii, :, toi_align-len(toi_align)/2].T - .5)
        pretty_decod(result[3, :, toi_align-len(toi_align)/2].T, ax=ax,
                     times=times_align[toi_align] - t/2, color='r', chance=.5)
        pretty_decod(result[3-ii, :, toi_align-len(toi_align)/2].T,
                     color=col, ax=ax, chance=.5,
                     times=times_align[toi_align] - t/2, alpha=1., fill=True,
                     sig=p_val < .05)
        ax.set_yticks([.25, 1.])
        ax.set_yticklabels([.25, 1.])
        ax.set_ylabel('AUC', labelpad=-15)
        ax.set_ylim([.25, 1.])
        xticks = np.arange(-t/2., t/2.+.01, .100)
        ax.set_xticks(xticks)
        ax.set_xlim(-t/2., t/2)
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
                [int(x) if np.round(x) in [-t/2 * 1e3, t/2 * 1e3]
                 else '' for x in np.round(1e3 * xticks)])

fig.tight_layout()
report.add_figs_to_section(fig, 'duration', 'duration')

# add report to Table: Duration
table_data = np.zeros((8, 2, len(subjects)))
freq = len(times) / np.ptp(times)
for jj, result, toi, t in zip(range(2), data, tois[1:], [.300, .600]):
    for pas in range(4):
        score = results['AUC_pas_duration'][jj, pas, :, :len(times)/2]
        # we had .5 at the end, to ensure that we find a value for each subject
        # This could bias the distribution towards longer duration, so we'll
        # take the median across subjects.
        score = np.hstack((score, [[.5]] * len(subjects)))
        # for each subject, find the first time sample that is below chance
        table_data[pas, jj, :] = [np.where(s <= .5)[0][0]/freq for s in score]
    # mean across visibility to get overall estimate
    table_data[4, jj, :] = np.nanmean(table_data[:, jj, :], axis=0)
    # seen - unseen
    table_data[5, jj, :] = table_data[3, jj, :] - table_data[0, jj, :]
# interaction time
table_data[6, 0, :] = table_data[4, 1, :] - table_data[4, 0, :]
# interaction time x vis
table_data[7, 0, :] = table_data[5, 1, :] - table_data[5, 0, :]
table = np.empty((8, 2), dtype=object)
# Transfor this data into stats summary:
for ii in range(8):
    for jj in range(2):
        score = table_data[ii, jj, :]
        m = np.nanmean(score)
        sem = np.nanstd(score) / np.sqrt(sum(~np.isnan(score)))
        p_val = wilcoxon(score)[1] if sum(abs(score)) > 0. else 1.
        # the stats for each pas is not meaningful because there's no chance
        # level, we 'll thus skip it
        p_val = p_val if ii > 3 else np.inf
        table[ii, jj] = '[%.3f+/-%.3f, p=%.4f]' % (m, sem, p_val)
table = table2html(table, head_column=tois[1:3],
                   head_line=['pas%i' % pas for pas in range(4)] +
                             ['pst', 'seen-unseen', 'late-early',
                              '(late-early)*(seen-unseen)'])
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

R_early = np.mean(results['R_contrast'][:, t_early], axis=1)
R_delay = np.mean(results['R_contrast'][:, t_delay], axis=1)
wilcoxon(R_early - R_delay)

report.save()
