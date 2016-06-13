"""This set of analyses are perform to test whether the decoding of the target
angle after probe onset is bias by and/or solely due to the presence of a probe
whose orientation is correlated to the target's"""
import numpy as np
from jr.stats import circ_tuning
from config import load, save, subjects, tois
from base import stats, get_predict_error, angle_acc, angle_bias

n_bins = 24
toi_probe = tois[-1]

n_time = 151  # XXX should be identified automatically
# initialize results
results = dict(accuracy=np.nan*np.zeros((20, 2, 2, n_time, n_time)),
               bias=np.nan*np.zeros((20, 2, 2, n_time, n_time)),
               bias_toi=np.nan*np.zeros((20, 2, 2, len(tois))),
               bias_vis=np.nan*np.zeros((20, 2, 2, 4, n_time, n_time)),
               bias_vis_toi=np.nan*np.zeros((20, 2, 2, 4, len(tois))),
               tuning=np.nan*np.zeros((20, 2, 2, n_bins, 3)))


# This Analysis is performed twice: 1) for estimators fitted on the
# orientations of the target 2) for estimators fitted on the orientations of
# the probe.
for ii, train_analysis in enumerate(['target_circAngle', 'probe_circAngle']):
    for s, subject in enumerate(subjects):
        print s

        # Load decoding data
        gat, _, events_sel, events = load('decod', subject=subject,
                                          analysis=train_analysis)
        subevents = events.iloc[events_sel].reset_index()

        # Single trial Target - Probe tilt (-1 or 1)
        y_tilt = np.array(subevents['probe_tilt'])
        times = gat.train_times_['times']
        n_train, n_test = np.shape(gat.y_pred_)[:2]

        # Mean error across trial on the diagonal
        # when train on probe, some trial contain no target => sel
        test_analysis = ['target_circAngle', 'probe_circAngle']
        for jj, test in enumerate(test_analysis):
            y_true = np.array(subevents[test])
            sel = np.where(~np.isnan(y_true))[0]
            # compute angle error between estimated angle from MEG topography
            # and target angle (target_circAngle) OR probe angle.
            # This therefore results in a 2x2 results of
            # target/probe estimators X target/probe bias.
            y_error = get_predict_error(gat, mean=False, typ='gat',
                                        y_true=y_true)

            # Accuracy train test target probe: absolute values
            accuracy = angle_acc(y_error[sel, :, :], axis=0)
            results['accuracy'][s, ii, jj, :, :] = accuracy

            # Bias train test target probe: signed values
            sel = np.where(~np.isnan(subevents['target_circAngle']))[0]
            results['bias'][s, ii, jj, :, :] = angle_bias(
                y_error[sel, :, :], y_tilt[sel])

            # Tuning bias toi
            sel = np.where(~np.isnan(subevents['target_circAngle']))[0]
            for t, toi in enumerate(tois):
                y_error_toi = get_predict_error(gat, y_true=y_true[sel],
                                                sel=sel, toi=toi, mean=True,
                                                typ='diagonal')
                # same but after averaging predicted angle across time
                results['bias_toi'][s, ii, jj, t] = angle_bias(
                    np.squeeze(y_error_toi), y_tilt[sel])

            # Tuning bias seen / unseen
            for pas in range(4):
                sel = np.where((~np.isnan(subevents['target_circAngle'])) &
                               (subevents.detect_button == pas))[0]
                if len(sel) < 10:
                    continue
                results['bias_vis'][s, ii, jj, pas, :, :] = angle_bias(
                    y_error[sel, :, :], y_tilt[sel])

                for t, toi in enumerate(tois):
                    y_error_toi = get_predict_error(gat, y_true=y_true[sel],
                                                    sel=sel, toi=toi,
                                                    mean=True,
                                                    typ='diagonal')
                    # same but after averaging predicted angle across time
                    results['bias_vis_toi'][s, ii, jj, pas, t] = angle_bias(
                        np.squeeze(y_error_toi), y_tilt[sel])

            # Tuning curve for probe tilted to -1 and probe tilted to 1
            tuning = list()
            for probe_tilt in [-1, np.nan, 1]:
                if np.isnan(probe_tilt):
                    sel = np.where(np.isnan(subevents.probe_tilt))[0]
                else:
                    sel = np.where(subevents.probe_tilt == probe_tilt)[0]
                if len(sel) == 0 or np.isnan(y_true[sel]).any():
                    tuning.append(np.nan * np.zeros(n_bins))
                    continue
                y_error = get_predict_error(gat, toi=toi_probe, sel=sel,
                                            mean=True, y_true=y_true[sel])
                probas, bins = circ_tuning(y_error, n=n_bins)
                tuning.append(probas)
            results['tuning'][s, ii, jj, :, :] = np.transpose(tuning)

# test significance of target versus probe train test
# for biases (signed values)
results['bias_pval'] = np.zeros_like((results['bias'][0]))
for ii in range(2):
    for jj in range(2):
        scores = results['bias'][:, ii, jj, :, :]
        results['bias_pval'][ii, jj, :, :] = stats(scores)

# for accuracy (absolute values)
results['target_probe_pval'] = np.zeros((n_time, n_time, 2, 2))
for ii in range(2):
    for jj in range(2):
        results['target_probe_pval'][:, :, ii, jj] = stats(
            results['accuracy'][:, ii, jj, :, :])

# load absent target prediction to perform the control analysis of virtual
# biases
results['target_absent'] = np.zeros((20, n_time, n_time))
results['target_absent_bias_toi'] = np.zeros((20, len(tois)))
for s, subject in enumerate(subjects):  # Loop across each subject
    print(subject)
    gat, analysis, events_sel, events = load(
        'decod', subject=subject, analysis='target_circAngle_absent')
    results['target_absent'][s, :, :] = gat.scores_
    # compute virtual bias to compare to unseen trials
    subevents = events.iloc[events_sel].reset_index()
    # virtual tilt
    y_tilt = (np.arange(len(events_sel)) % 2) * 2. - 1.
    y_true = subevents['probe_circAngle']
    for t, toi in enumerate(tois):
        y_error_toi = get_predict_error(gat, y_true=y_true, toi=toi,
                                        typ='diagonal', mean=True)
        # same but after averaging predicted angle across time
        results['target_absent_bias_toi'][s, t] = angle_bias(
            np.squeeze(y_error_toi), y_tilt)
results['target_absent_pval'] = stats(results['target_absent'])

# Save results
results['times'] = gat.train_times_['times']
results['bins'] = bins
results['tois'] = tois
save(results, 'score', analysis='target_probe', overwrite=True, upload=True)
