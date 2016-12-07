# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# Licence: BSD 3-clause

"""This set of analyses are perform to test whether the decoding of the target
angle after probe onset is bias by and/or solely due to the presence of a probe
whose orientation is correlated to the target's"""
import numpy as np
from jr.stats import circ_tuning, circ_mean, corr_circular_linear
from config import load, save, subjects
from base import stats
from conditions import tois


# ad-hoc functions: XXX needs clean-up


def get_predict(gat, sel=None, toi=None, mean=True, typ='diagonal'):
    """Retrieve decoding prediction from a GeneralizationAcrossTime object"""
    from jr.gat import get_diagonal_ypred
    from jr.utils import align_on_diag
    # select data in the gat matrix
    if typ == 'diagonal':
        y_pred = np.transpose(get_diagonal_ypred(gat), [1, 0, 2])
    elif typ == 'align_on_diag':
        y_pred = np.squeeze(align_on_diag(gat.y_pred_)).transpose([2, 0, 1, 3])
    elif typ == 'gat':
        y_pred = np.squeeze(gat.y_pred_).transpose([2, 0, 1, 3])
    elif typ == 'slice':
        raise NotImplementedError('slice')
    y_pred = y_pred % (2 * np.pi)  # make sure data is in on circle
    # Select trials
    sel = range(len(y_pred)) if sel is None else sel
    y_pred = y_pred[sel, ...]
    # select TOI
    times = np.array(gat.train_times_['times'])
    toi = times[[0, -1]] if toi is None else toi
    toi_ = np.where((times >= toi[0]) & (times <= toi[1]))[0]
    y_pred = y_pred[:, toi_, ...]
    # mean across time point
    if mean:
        # weighted circular mean (dim = angle * radius)
        cos = np.mean(np.cos(y_pred[..., 0]) * y_pred[..., 1], axis=1)
        sin = np.mean(np.sin(y_pred[..., 0]) * y_pred[..., 1], axis=1)
        radius = np.median(y_pred[..., 1], axis=1)
        angle = np.arctan2(sin, cos)
        y_pred = lstack(angle, radius)
    return y_pred[:, None] if y_pred.ndim == 1 else y_pred


def lstack(x, y):
    """Stack x and y and transpose"""
    z = np.stack([x, y])
    return np.transpose(z, np.r_[range(1, z.ndim), 0])


def get_predict_error(gat, sel=None, toi=None, mean=True, typ='diagonal',
                      y_true=None):
    """Retrieve single trial error from a GeneralizationAcrossTime object"""
    y_pred = get_predict(gat, sel=sel, toi=toi, mean=mean, typ=typ)[..., 0]
    # error is diff modulo pi centered on 0
    sel = range(len(y_pred)) if sel is None else sel
    if y_true is None:
        y_true = gat.y_true_[sel]
    y_true = np.tile(y_true, np.hstack((np.shape(y_pred)[1:], 1)))
    y_true = np.transpose(y_true, [y_true.ndim - 1] + range(y_true.ndim - 1))
    y_error = (y_pred - y_true + np.pi) % (2 * np.pi) - np.pi
    return y_error


def angle_acc(y_error, axis=None):
    # range between -pi and pi just in case not done already
    y_error = y_error % (2 * np.pi)
    y_error = (y_error + np.pi) % (2 * np.pi) - np.pi
    # random error = np.pi/2, thus:
    return np.pi / 2 - np.mean(np.abs(y_error), axis=axis)


def angle_bias(y_error, y_tilt):
    # This is an ad hoc function to compute the systematic bias across angles
    # It consists in testing whether the angles are correlated with the tilt
    # [-1, 1] and multiplying the root square resulting R square value by the
    #  sign of the mean angle.
    # In this way, if there is a correlations, we can get a positive or
    # negative R value depending on the direction of the bias, and get 0 if
    # there's no correlation.
    n_train, n_test = 1, 1
    y_tilt_ = y_tilt
    if y_error.ndim == 3:
        n_train, n_test = np.shape(y_error)[1:]
        y_tilt_ = np.tile(y_tilt, [n_train, n_test, 1]).transpose([2, 0, 1])

    # compute mean angle
    alpha = circ_mean(y_error * y_tilt_, axis=0)
    alpha = ((alpha + np.pi) % (2 * np.pi)) - np.pi
    # compute correlation
    _, R2, _ = corr_circular_linear(y_error.reshape([len(y_error), -1]),
                                    y_tilt)
    R2 = R2.reshape([n_train, n_test])
    # set direction of the bias
    R = np.sqrt(R2) * np.sign(alpha)
    return R


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
results['target_absent'] = np.zeros((20, n_time, 181))
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
