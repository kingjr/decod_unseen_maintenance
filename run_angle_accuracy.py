import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from jr.plot import pretty_decod
from jr.gat import get_diagonal_ypred, scorer_angle
from config import subjects, load, report, save, paths
from base import stats

if not op.exists(paths('score', analysis='accuracy')):
    scores = np.ones((20, 151, 2))
    all_y = list()
    for s, subject in enumerate(subjects):
        # Load data
        print s
        gat, _, events_sel, events = load('decod', subject=subject,
                                          analysis='target_circAngle')
        times = gat.train_times_['times']
        y_true = gat.y_train_
        y_correct = np.array(events['discrim_correct'])[events_sel]

        # get diagonal
        y_pred = np.transpose(get_diagonal_ypred(gat), [1, 0, 2])[..., 0]
        for correct in range(2):
            sel = np.where(y_correct == correct)[0]
            scores[s, :, correct] = scorer_angle(y_true[sel], y_pred[sel, :])
    save([scores, times], 'score', analysis='accuracy',
         upload=True, overwrite=True)

scores, times = load('score', analysis='accuracy')
# do stats
# pval_incorrect = stats(scores[:, :, 0])
# pval_correct = stats(scores[:, :, 1])
pval_diff = stats(scores[:, :, 1] - scores[:, :, 0])

# Plot
fig, ax = plt.subplots(1, figsize=[6.5, 3])
pretty_decod(scores[:, :, 1] / 2., times=times, sig=pval_diff < .05,
             chance=0., color='y', fill=True, ax=ax)
pretty_decod(scores[:, :, 1] / 2., times=times,  # sig=pval_correct < .05,
             chance=0., color='r', fill=False, ax=ax)
pretty_decod(scores[:, :, 0.] / 2., times=times,  # sig=pval_incorrect < .05,
             chance=0., color='b', fill=False, ax=ax)

xlim, ylim = ax.get_xlim(), np.array(ax.get_ylim())
sem = scores.std(0) / np.sqrt(len(scores[:, :, 1]))
ylim = [np.min(scores.mean(0) - sem), np.max(scores.mean(0) + sem)]
ax.set_ylim(ylim)
ax.axvline(.800, color='k')
ax.set_xticklabels([int(x) if x in np.linspace(0., 1000., 11) else ''
                    for x in np.round(1e3 * ax.get_xticks())])
ax.text(0, ylim[1], 'Target',  backgroundcolor='w', ha='center', va='top')
ax.text(.800, ylim[1], 'Probe', backgroundcolor='w', ha='center', va='top')
ax.set_yticks([0., ylim[1]])
ax.set_yticklabels(['', '%.2f' % ylim[1]])
ax.set_ylabel('rad.', labelpad=-15)
txt = ax.text(xlim[0] + .5 * np.ptp(xlim), ylim[0] + .75 * np.ptp(ylim),
              'Target Angle x Accuracy', color=[.2, .2, .2], ha='center',
              weight='bold')
report.add_figs_to_section([fig], ['accuracy'], 'accuracy')
