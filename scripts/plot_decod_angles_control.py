import pickle
import numpy as np
import matplotlib.pyplot as plt
from jr.utils import align_on_diag
from jr.plot import pretty_gat, plot_tuning
from jr.stats import circ_tuning, circ_mean, corr_circular_linear

from scripts.config import paths, subjects, report
from base import stats

tois = [(-.100, 0.050), (.100, .200), (.300, .800), (.900, 1.050)]


def get_predict(gat, sel=None, toi=None, mean=True, typ='diagonal'):
    from jr.gat import get_diagonal_ypred
    # select data in the gat matrix
    if typ == 'diagonal':
        y_pred = np.squeeze(get_diagonal_ypred(gat)).T
    elif typ == 'align_on_diag':
        y_pred = np.squeeze(align_on_diag(gat.y_pred_)).transpose([2, 0, 1])
    elif typ == 'gat':
        y_pred = np.squeeze(gat.y_pred_).transpose([2, 0, 1])
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
        y_pred = circ_mean(y_pred, axis=1)
    return y_pred[:, None] if y_pred.ndim == 1 else y_pred


def get_predict_error(gat, sel=None, toi=None, mean=True, typ='diagonal',
                      y_true=None):
    y_pred = get_predict(gat, sel=sel, toi=toi, mean=mean, typ=typ)
    # error is diff modulo pi centered on 0
    sel = range(len(y_pred)) if sel is None else sel
    if y_true is None:
        y_true = gat.y_true_[sel]
    y_true = np.tile(y_true, np.hstack((np.shape(y_pred)[1:], 1)))
    y_true = np.transpose(y_true, [y_true.ndim - 1] + range(y_true.ndim - 1))
    y_error = (y_pred - y_true + np.pi) % (2 * np.pi) - np.pi
    return y_error


def mean_acc(y_error, axis=None):
    # range between -pi and pi just in case not done already
    y_error = y_error % (2 * np.pi)
    y_error = (y_error + np.pi) % (2 * np.pi) - np.pi
    # random error = np.pi/2, thus:
    return np.pi / 2 - np.mean(np.abs(y_error), axis=axis)


def mean_bias(y_error, y_tilt):
    # This is an ad hoc function to compute the systematic bias across angles
    # It consists in whether the angles are correlated with the tilt [-1, 1]
    # and multiplying the root square resulting R² value by the sign of the
    # mean angle.
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

# XXX => in config
n_bins = 24
toi_probe = [.900, 1.050]
# Gather data #################################################################
results = dict(accuracy=np.zeros((20, 2, 2, 154, 154)),
               bias=np.zeros((20, 2, 2, 154, 154)),
               bias_vis=np.zeros((20, 2, 2, 4, 154, 154)),
               bias_vis_toi=np.zeros((20, 2, 2, 4)),
               tuning=np.zeros((20, 2, 2, n_bins, 3)))
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

                y_error_toi = np.array([np.diag(y_) for y_ in y_error])
                toi_ = (times >= .900) & (times <= 1.050)
                y_error_toi = circ_mean(y_error_toi[:, toi_], axis=1)
                # same but after averaging predicted angle across time
                results['bias_vis_toi'][s, ii, jj, pas] = mean_bias(
                    y_error_toi[sel], y_tilt[sel])

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


# test significance of target versus probe train test
results['target_probe_pval'] = np.zeros((154, 154, 2, 2))
for ii in range(2):
    for jj in range(2):
        results['target_probe_pval'][:, :, ii, jj] = stats(
            results['accuracy'][:, ii, jj, :, :])

# load absent target prediction
results['target_absent'] = np.zeros((20, 154, 153))
for s, subject in enumerate(subjects):  # Loop across each subject
    print(subject)
    pkl_fname = paths('decod', subject=subject,
                      analysis='target_circAngle_absent')
    with open(pkl_fname, 'rb') as f:
        gat, analysis, sel, events = pickle.load(f)
    results['target_absent'][s, :, :] = gat.scores_
results['target_absent_pval'] = stats(results['target_absent'])

# save
results['times'] = gat.train_times_['times']
results['bins'] = bins
fname = paths('score', subject='fsaverage', analysis='target_probe')
with open(fname, 'wb') as f:
    pickle.dump(results, f)


# #####################################
fname = paths('score', subject='fsaverage', analysis='target_probe')
with open(fname, 'rb') as f:
    results = pickle.load(f)


# Plot tuning probe time: train test target probe
cmap = plt.get_cmap('BrBG')
colors = cmap(np.linspace(0.2, .8, 3))
fig, axes = plt.subplots(2, 2)
for ii in range(2):
    for jj in range(2):
        for tilt, color in enumerate(colors):
            if tilt == 1:
                continue  # do not plot absent case
            plot_tuning(results['tuning'][:, ii, jj, :, tilt],
                        ax=axes[ii, jj], shift=np.pi, color=color)
            plot_tuning(results['tuning'][:, ii, jj, :, tilt],
                        ax=axes[ii, jj], shift=np.pi, color='k', alpha=0.)
            if ii == 0:
                axes[ii, jj].set_title(['Target', 'Probe'][jj])
                axes[ii, jj].set_xlabel('')
                axes[ii, jj].set_xticklabels([])
            else:
                axes[ii, jj].set_xticklabels(['$-\pi/2$', '', '$\pi/2$'])
                axes[ii, jj].set_xlabel('Angle Error', labelpad=-10)
            axes[ii, jj].set_yticks([0.014, 0.08])
            axes[ii, jj].set_yticklabels([])
            if jj == 0:
                axes[ii, jj].set_yticklabels([1.4, 8.])
                axes[ii, jj].set_ylabel('Probability', labelpad=-10)
            axes[ii, jj].axvline(-np.pi / 3, color='k')
            axes[ii, jj].axvline(np.pi / 3, color='k')
            axes[ii, jj].axvline(0, color='k')
fig.tight_layout()

# Plot bias GAT
fig, axes = plt.subplots(2, 2)
for ii in range(2):
    for jj in range(2):
        score = np.mean(results['bias'][:, ii, jj, ...], axis=0)
        pretty_gat(score, times=times, ax=axes[ii, jj])

# Plot bias unseen: doesn't work
fig, axes = plt.subplots(2, 2)
score = results['bias_vis'][:, 0, 1, 1, :, :]
score = np.array([np.diagonal(s) for s in score])
score = np.mean(score[:, (times > .900) & (times < 1.050)], axis=1)

# From here: test whether bias unseen works when focusing on a TOI
