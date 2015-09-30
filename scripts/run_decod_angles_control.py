import pickle
import numpy as np
from jr.stats import circ_tuning
from scripts.config import paths, subjects
from base import stats, get_predict_error, mean_acc, mean_bias

# XXX => in config
tois = [(-.100, 0.050), (.100, .200), (.300, .800), (.900, 1.050)]
n_bins = 24
toi_probe = [.900, 1.050]

results = dict(accuracy=np.nan*np.zeros((20, 2, 2, 154, 154)),
               bias=np.nan*np.zeros((20, 2, 2, 154, 154)),
               bias_vis=np.nan*np.zeros((20, 2, 2, 4, 154, 154)),
               bias_vis_toi=np.nan*np.zeros((20, 2, 2, 4, len(tois))),
               tuning=np.nan*np.zeros((20, 2, 2, n_bins, 3)))

for ii, train_analysis in enumerate(['target_circAngle', 'probe_circAngle']):
    for s, subject in enumerate(subjects):
        print s
        fname = paths('decod', subject=subject, analysis=train_analysis)
        with open(fname, 'rb') as f:
            gat, _, events_sel, events = pickle.load(f)
        subevents = events.iloc[events_sel].reset_index()
        y_tilt = np.array(subevents['probe_tilt'])
        times = gat.train_times_['times']
        n_train, n_test = np.shape(gat.y_pred_)[:2]
        # Mean error across trial on the diagonal
        # when train on probe, some trial contain no target => sel
        test_analysis = ['target_circAngle', 'probe_circAngle']
        for jj, test in enumerate(test_analysis):
            y_true = np.array(subevents[test])
            sel = np.where(~np.isnan(y_true))[0]
            # compute angle error
            y_error = get_predict_error(gat, mean=False, typ='gat',
                                        y_true=y_true)

            # Accuracy train test target probe
            accuracy = mean_acc(y_error[sel, :, :], axis=0)
            results['accuracy'][s, ii, jj, :, :] = accuracy

            # Bias train test target probe
            sel = np.where(~np.isnan(subevents['target_circAngle']))[0]
            results['bias'][s, ii, jj, :, :] = mean_bias(
                y_error[sel, :, :], y_tilt[sel])

            # Tuning bias seen / unseen
            for pas in range(4):
                sel = np.where((~np.isnan(subevents['target_circAngle'])) &
                               (subevents.detect_button == pas))[0]
                if len(sel) < 10:
                    continue
                results['bias_vis'][s, ii, jj, pas, :, :] = mean_bias(
                    y_error[sel, :, :], y_tilt[sel])

                for t, toi in enumerate(tois):
                    y_error_toi = get_predict_error(gat, y_true=y_true[sel],
                                                    sel=sel, toi=toi,
                                                    typ='diagonal')
                    # same but after averaging predicted angle across time
                    results['bias_vis_toi'][s, ii, jj, pas, t] = mean_bias(
                        np.squeeze(y_error_toi), y_tilt[sel])

            # Tuning curve for probe 1 and probe 2
            tuning = list()
            for probe_tilt in [-1, np.nan, 1]:
                if np.isnan(probe_tilt):
                    sel = np.where(np.isnan(subevents.probe_tilt))[0]
                else:
                    sel = np.where(subevents.probe_tilt == probe_tilt)[0]
                if len(sel) == 0:
                    tuning.append(np.nan * np.zeros(n_bins))
                    continue
                y_error = get_predict_error(gat, toi=toi_probe, sel=sel,
                                            y_true=y_true[sel])
                probas, bins = circ_tuning(y_error, n=n_bins)
                tuning.append(probas)
            results['tuning'][s, ii, jj, :, :] = np.transpose(tuning)

results['bias_pval'] = np.zeros_like((results['bias'][0]))
for ii in range(2):
    for jj in range(2):
        scores = results['bias'][:, ii, jj, :, :]
        results['bias_pval'][ii, jj, :, :] = stats(scores)

# test significance of target versus probe train test
results['target_probe_pval'] = np.zeros((154, 154, 2, 2))
for ii in range(2):
    for jj in range(2):
        results['target_probe_pval'][:, :, ii, jj] = stats(
            results['accuracy'][:, ii, jj, :, :])

# load absent target prediction
results['target_absent'] = np.zeros((20, 154, 153))
results['target_absent_bias_toi'] = np.zeros((20, len(tois)))
for s, subject in enumerate(subjects):  # Loop across each subject
    print(subject)
    pkl_fname = paths('decod', subject=subject,
                      analysis='target_circAngle_absent')
    with open(pkl_fname, 'rb') as f:
        gat, analysis, events_sel, events = pickle.load(f)
    results['target_absent'][s, :, :] = gat.scores_
    # compute virtual bias to compare to unseen trials
    subevents = events.iloc[events_sel].reset_index()
    # virtual tilt
    y_tilt = (np.arange(len(events_sel)) % 2) * 2. - 1.
    y_true = subevents['probe_circAngle']
    for t, toi in enumerate(tois):
        y_error_toi = get_predict_error(gat, y_true=y_true, toi=toi,
                                        typ='diagonal')
        # same but after averaging predicted angle across time
        results['target_absent_bias_toi'][s, t] = mean_bias(
            np.squeeze(y_error_toi), y_tilt)
results['target_absent_pval'] = stats(results['target_absent'])

# save
results['times'] = gat.train_times_['times']
results['bins'] = bins
results['tois'] = tois
fname = paths('score', subject='fsaverage', analysis='target_probe')
with open(fname, 'wb') as f:
    pickle.dump(results, f)
