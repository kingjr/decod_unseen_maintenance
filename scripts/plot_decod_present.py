import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from jr.gat import get_diagonal_ypred, subscore
from jr.stats import repeated_spearman
from jr.plot import pretty_decod
from jr.utils import align_on_diag
from scripts.config import paths, subjects, report, analyses
from base import stats

# Test whether decoding of presence varies as a function of contrast and
# visibility

# Gather data
tois = [(-.100, 0.050), (.100, .250), (.300, .800), (.900, 1.050)]
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

# Gather data
if False:
    for s, subject in enumerate(subjects):
        print s
        fname = paths('decod', subject=subject, analysis='target_present')
        with open(fname, 'rb') as f:
            gat, _, events_sel, events = pickle.load(f)
        times = gat.train_times_['times']
        y_pred = np.transpose(np.squeeze(get_diagonal_ypred(gat)))
        y_error = y_pred - np.tile(gat.y_true_, [n_times, 1]).T
        subevents = events.iloc[events_sel].reset_index()

        # contrast effect
        r = list()
        for ii, pas in enumerate(pas_list):
            key = 'detect_button == %s and target_present == True' % pas
            subsel = subevents.query(key).index
            if len(subsel) > 0:
                r.append(repeated_spearman(y_error[subsel, :],
                         np.array(subevents.target_contrast)[subsel]))
        results['R_contrast'][s, :] = np.nanmean(r, axis=0)

        # visibility effect
        r = list()
        for ii, contrast in enumerate(contrast_list):
            key = 'target_contrast == %s' % contrast
            subsel = subevents.query(key).index
            if len(subsel) > 0:
                r.append(repeated_spearman(y_error[subsel, :],
                         np.array(subevents.detect_button)[subsel]))
        results['R_vis'][s, :] = np.nanmean(r, axis=0)

        # mean decoding seen unseen
        for ii, pas in enumerate(pas_list):
            key = 'detect_button == %s or target_present == False' % pas
            subsel = subevents.query(key).index
            if len(subsel) == 0:
                continue
            score = subscore(gat, subsel)
            results['AUC_pas'][ii, s, :] = np.diagonal(score)
            # duration effect
            score_align = align_on_diag(score)
            for jj, toi in enumerate(tois):
                results_pas = list()
                toi_ = np.where((times >= toi[0]) & (times <= toi[1]))[0]
                results['AUC_pas_duration'][jj, ii, s, :] = np.mean(
                    score_align[toi_, :], axis=0)

    results['times'] = times
    results['p_vis'] = stats(results['R_vis'][:, :, None])
    results['p_contrast'] = stats(results['R_contrast'][:, :, None])

    fname = paths('score', analysis='present_anova')
    with open(fname, 'wb') as f:
        pickle.dump(results, f)


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
freq = np.ptp(times) / len(times)
times_align = times - np.min(times) - np.ptp(times) / 2
toi_align = np.where((times_align > 0.) & (times_align < .304))[0]
fig, axes = plt.subplots(1, len(data), figsize=[4, 4])
cmap = plt.get_cmap('bwr_r')
for ax, result, toi in zip(axes, data, tois[1:]):
    for ii, col in enumerate(cmap(np.linspace(0, 1, 4.))):
        if ii in [1, 2]:
            continue
        pretty_decod(result[3-ii, :, toi_align].T, color=col, ax=ax, chance=.5,
                     times=times_align[toi_align], alpha=1., fill=True,
                     sig=stats(result[3-ii, :, toi_align].T - .5) < .05)
        ax.set_yticks([.25, 1.])
        ax.set_yticklabels([.25, 1.])
        ax.set_ylabel('AUC', labelpad=-15)
        if ax != axes[0]:
            ax.set_yticklabels(['', ''])
            ax.set_ylabel('')
        ax.set_ylim([.25, 1.])
        ax.set_xticks(np.arange(0, .301, .100))
        ax.set_title('%i $-$ %i ms' % (1e3 * toi[0], 1e3 * toi[1]))
        ax.set_xticks([0, .300])
        ax.set_xticklabels([0, 300])
        ax.set_xlabel('Duration', labelpad=-10)
fig.tight_layout()
report.add_figs_to_section(fig, 'duration', 'duration')

report.save()
