import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scripts.config import paths, subjects, report
from base import stats
from jr.gat import get_diagonal_ypred, subscore
from jr.stats import repeated_spearman
from jr.plot import pretty_decod
import scipy.io as sio
from scripts.config import analyses

# Test whether decoding of presence varies as a function of contrast and
# visibility

# Gather data
data = np.nan * np.zeros((20, 154, 4, 3))
R_vis = np.nan * np.zeros((20, 154))
R_contrast = np.nan * np.zeros((20, 154))
AUC_pas = np.nan * np.zeros((4, 20, 154))
contrast_list = [.5, .75, 1.]
pas_list = np.arange(4.)

for s, subject in enumerate(subjects):
    print s
    fname = paths('decod', subject=subject, analysis='target_present')
    with open(fname, 'rb') as f:
        gat, _, events_sel, events = pickle.load(f)
    times = gat.train_times_['times']
    y_pred = np.transpose(np.squeeze(get_diagonal_ypred(gat)))
    y_error = y_pred - np.tile(gat.y_true_, [154, 1]).T
    subevents = events.iloc[events_sel].reset_index()

    # For ANOVA: (better if need complex analysis, but parametric and effect
    # size across subjects)
    for ii, pas in enumerate(pas_list):
        for jj, contrast in enumerate(contrast_list):
            key = 'detect_button == %s and target_contrast == %s' % (pas,
                                                                     contrast)
            subsel = subevents.query(key).index
            data[s, :, ii, jj] = np.nanmean(y_error[subsel, :], axis=0)

    # Manual, and non parametric
    # contrast
    r = list()
    for ii, pas in enumerate(pas_list):
        key = 'detect_button == %s and target_present == True' % pas
        subsel = subevents.query(key).index
        if len(subsel) > 0:
            r.append(repeated_spearman(y_error[subsel, :],
                     np.array(subevents.target_contrast)[subsel]))
    R_contrast[s, :] = np.nanmean(r, axis=0)

    # vis
    r = list()
    for ii, contrast in enumerate(contrast_list):
        key = 'target_contrast == %s' % contrast
        subsel = subevents.query(key).index
        if len(subsel) > 0:
            r.append(repeated_spearman(y_error[subsel, :],
                     np.array(subevents.detect_button)[subsel]))
    R_vis[s, :] = np.nanmean(r, axis=0)

    # mean decoding seen unseen
    for ii, pas in enumerate(pas_list):
        key = 'detect_button == %s or target_present == False' % pas
        subsel = subevents.query(key).index
        if len(subsel) > 0:
            score = subscore(gat, subsel)
            AUC_pas[ii, s, :] = np.diagonal(score)

sio.savemat('test.mat', dict(data=data))
# anova in matlab
# # addpath('/media/DATA/Pro/Toolbox/JR_toolbox/')
# [n_subjects, n_time, n_pas, n_cont,] = size(data)
# P = zeros(3, n_time);
# for t = 1:n_time
#     [Y, GROUP] = prepare_anovan(squeeze(data(:,t,:,:)));
#     P(:, t) = anovan(Y, GROUP, 'random', 1,
#                      'model', 'linear', 'display', 'off');
# end
# plot(-log10(P(2:end,:))','linewidth',3)

fname = paths('score', analysis='present_anova')
with open(fname, 'wb') as f:
    pickle.dump([data, R_vis, R_contrast, AUC_pas], f)

p_vis = stats(R_vis[:, :, None])
p_contrast = stats(R_contrast[:, :, None])

color_vis = [analysis['color'] for analysis in analyses
             if analysis['title'] == 'Visibility Response'][0]
color_contrast = [analysis['color'] for analysis in analyses
                  if analysis['title'] == 'Target Contrast'][0]

fig, ax = plt.subplots(1, figsize=[6, 2])
pretty_decod(R_vis, times=times, sig=p_vis < .05, color=color_vis, chance=0.,
             fill=True)
pretty_decod(R_vis, times=times, sig=p_vis < .05, color='k', chance=0.)
ax.axvline(.800, color='k')
ylim = ax.get_ylim()
ax.set_yticklabels(['', '', '%.2f' % ylim[1]])
ax.text(.370, 1.1 * ylim[1], 'Visibility Effect', color=color_vis,
        weight='bold')
ax.set_ylim(ylim[0], 1.3 * ylim[1])
ax.set_ylabel('R', labelpad=-10)
fig.tight_layout()
report.add_figs_to_section(fig, 'visibility', 'R')

fig, ax = plt.subplots(1, figsize=[6, 2])
pretty_decod(R_contrast, times=times, sig=p_contrast < .05,
             color=color_contrast, chance=0., fill=True)
pretty_decod(R_contrast, times=times, sig=p_contrast < .05, color='k',
             chance=0.)
ax.axvline(.800, color='k')
ylim = ax.get_ylim()
ax.set_yticklabels(['', '', '%.2f' % ylim[1]])
ax.text(.370, .8 * ylim[1], 'Contrast Effect', color=color_contrast,
        weight='bold')
fig.tight_layout()
ax.set_ylabel('R', labelpad=-10)
ax.set_xlabel('Times', labelpad=-10)
report.add_figs_to_section(fig, 'contrast', 'R')

fig, ax = plt.subplots(1, figsize=[6, 2])
cmap = mpl.colors.LinearSegmentedColormap.from_list('RdPuBu', ['b', 'r'])
colors = cmap(np.linspace(0., 1., 4))
for ii, (auc, color) in enumerate(zip(AUC_pas[-1::-1, :, :],
                                      colors[-1::-1, :])):
    if ii not in [0, 3]:
        continue
    p_val = stats(auc[:, :, None] - .5)
    pretty_decod(auc, times=times, ax=ax, width=1., alpha=1.,
                 chance=.5, color=color, fill=True, sig=p_val < .05)
    pretty_decod(np.mean(auc, axis=0)[:, None].T, times=times, chance=.5,
                 color='k', sig=p_val < .05)
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

report.save()
