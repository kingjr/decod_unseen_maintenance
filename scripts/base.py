import numpy as np
from sklearn.svm import LinearSVR, SVC
from sklearn.linear_model import LogisticRegression
from jr.utils import tile_memory_free, pairwise, table2html
from jr.stats import (repeated_spearman, repeated_corr, corr_linear_circular,
                      circ_tuning, circ_mean, corr_circular_linear)
from jr.gat.scorers import scorer_auc, scorer_spearman
from mne.stats import spatio_temporal_cluster_1samp_test
from scipy.stats import wilcoxon


# STATS #######################################################################

def stat_fun(x, sigma=0, method='relative'):
    from mne.stats import ttest_1samp_no_p
    t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
    t_values[np.isnan(t_values)] = 0
    return t_values


def stats(X):
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask', stat_fun=stat_fun, n_permutations=2**12,
        n_jobs=-1)
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
    return np.squeeze(p_values_).T

# ANALYSES ####################################################################


def table_duration(data, tois, times, chance):
    n_toi, n_vis, n_subject, n_time = data.shape
    if n_toi != 2:
        raise RuntimeError("Can only process early versus late time windows." +
                           " Got %i tois instead" % n_toi)
    if n_vis != 4:
        raise RuntimeError("Can only process early 4 visibility ratings")

    if len(times) != n_time:
        raise RuntimeError("Inconsistent time dimension")
    freq = len(times) / np.ptp(times)

    table_data = np.zeros((8, n_toi, n_subject))  # initialize array
    # loop across tois
    for ii, (data, toi) in enumerate(zip(data, tois)):
        # loop across visibility ratings
        for pas in range(n_vis):
            score = data[pas, :, :len(times)/2]
            # we had a chance value at the end, to ensure that we find a value
            # for each subject. Note that this could bias the distribution
            # towards shorter durations depending on the noise level.
            score = np.hstack((score, [[chance]] * n_subject))
            # for each subject, find the first time sample that is below chance
            table_data[pas, ii, :] = [np.where(s <= chance)[0][0]/freq
                                      for s in score]
        # mean across visibility to get overall estimate
        table_data[4, ii, :] = np.nanmean(table_data[:4, ii, :], axis=0)
        # seen - unseen
        table_data[5, ii, :] = table_data[3, ii, :] - table_data[0, ii, :]
    # interaction time: is early duration different from late
    table_data[6, 0, :] = table_data[4, 1, :] - table_data[4, 0, :]
    # interaction time x vis: does the difference (seen- unseen) vary with TOI
    table_data[7, 0, :] = table_data[5, 1, :] - table_data[5, 0, :]
    table = np.empty((8, n_toi), dtype=object)

    # Transfor this data into stats summary:
    for ii in range(8):
        for jj in range(n_toi):
            score = table_data[ii, jj, :]
            m = np.nanmedian(score)
            sem = np.nanstd(score) / np.sqrt(sum(~np.isnan(score)))
            p_val = wilcoxon(score)[1] if sum(abs(score)) > 0. else 1.
            # the stats for each pas is not meaningful because there's no
            # chance level, we 'll thus skip it
            p_val = p_val if ii > 3 else np.inf
            table[ii, jj] = '[%.3f+/-%.3f, p=%.4f]' % (m, sem, p_val)
    # HTML export
    headlines = (['pas%i' % pas for pas in range(4)] +
                 ['pst', 'seen-unseen', 'late-early',
                  '(late-early)*(seen-unseen)'])
    return table2html(table, head_column=tois, head_line=headlines)


def nested_analysis(X, df, condition, function=None, query=None,
                    single_trial=False, y=None, n_jobs=-1):
    """ Apply a nested set of analyses.
    Parameters
    ----------
    X : np.array, shape(n_samples, ...)
        Data array.
    df : pandas.DataFrame
        Condition DataFrame
    condition : str | list
        If string, get the samples for each unique value of df[condition]
        If list, nested call nested_analysis.
    query : str | None, optional
        To select a subset of trial using pandas.DataFrame.query()
    function : function
        Computes across list of evoked. Must be of the form:
        function(X[:], y[:])
    y : np.array, shape(n_conditions)
    n_jobs : int
        Number of core to compute the function. Defaults to -1.

    Returns
    -------
    scores : np.array, shape(...)
        The results of the function
    sub : dict()
        Contains results of sub levels.
    """
    import numpy as np
    if isinstance(condition, str):
        # Subselect data using pandas.DataFrame queries
        sel = range(len(X)) if query is None else df.query(query).index
        X = X.take(sel, axis=0)
        y = np.array(df[condition][sel])
        # Find unique conditions
        values = list()
        for ii in np.unique(y):
            if (ii is not None) and (ii not in [np.nan]):
                try:
                    if np.isnan(ii):
                        continue
                    else:
                        values.append(ii)
                except TypeError:
                    values.append(ii)
        # Subsubselect for each unique condition
        y_sel = [np.where(y == value)[0] for value in values]
        # Mean condition:
        X_mean = np.zeros(np.hstack((len(y_sel), X.shape[1:])))
        y_mean = np.zeros(len(y_sel))
        for ii, sel_ in enumerate(y_sel):
            X_mean[ii, ...] = np.mean(X[sel_, ...], axis=0)
            if isinstance(y[sel_[0]], str):
                y_mean[ii] = ii
            else:
                y_mean[ii] = y[sel_[0]]
        if single_trial:
            X = X.take(np.hstack(y_sel), axis=0)  # ERROR COME FROM HERE
            y = y.take(np.hstack(y_sel), axis=0)
        else:
            X = X_mean
            y = y_mean
        # Store values to keep track
        sub_list = dict(X=X_mean, y=y_mean, sel=sel, query=query,
                        condition=condition, values=values,
                        single_trial=single_trial)
    elif isinstance(condition, list):
        # If condition is a list, we must recall the function to gather
        # the results of the lower levels
        sub_list = list()
        X_list = list()  # FIXME use numpy array
        for subcondition in condition:
            scores, sub = nested_analysis(
                X, df, subcondition['condition'], n_jobs=n_jobs,
                function=subcondition.get('function', None),
                query=subcondition.get('query', None))
            X_list.append(scores)
            sub_list.append(sub)
        X = np.array(X_list)
        if y is None:
            y = np.arange(len(condition))
        if len(y) != len(X):
            raise ValueError('X and y must be of identical shape: ' +
                             '%s <> %s') % (len(X), len(y))
        sub_list = dict(X=X, y=y, sub=sub_list, condition=condition)

    # Default function
    function = _default_analysis if function is None else function

    scores = pairwise(X, y, function, n_jobs=n_jobs)
    return scores, sub_list


def _default_analysis(X, y):
    # from sklearn.metrics import roc_auc_score
    from jr.stats import fast_mannwhitneyu
    # Binary contrast
    unique_y = np.unique(y)
    # if two condition, can only return contrast
    if len(y) == 2:
        y = np.where(y == unique_y[0], 1, -1)
        # Tile Y to across X dimension without allocating memory
        Y = tile_memory_free(y, X.shape[1:])
        return np.mean(X * Y, axis=0)
    elif len(unique_y) == 2:
        # if two conditions but multiple trials, can return AUC
        # auc = np.zeros_like(X[0])
        _, _, auc = fast_mannwhitneyu(X[y == unique_y[0], ...],
                                      X[y == unique_y[1], ...], n_jobs=1)
        # for ii, x in enumerate(X.T):
        #     auc[ii] = roc_auc_score(y, np.copy(x))
        return auc
    # Linear regression:
    elif len(unique_y) > 2:
        return repeated_spearman(X, y)
    else:
        raise RuntimeError('Please specify a function for this kind of data')

# MNE #########################################################################


def meg_to_gradmag(chan_types):
    """force separation of magnetometers and gradiometers"""
    from mne.channels import read_ch_connectivity
    if 'meg' in [chan['name'] for chan in chan_types]:
        mag_connectivity, _ = read_ch_connectivity('neuromag306mag')
        # FIXME grad connectivity? Need virtual sensor?
        # grad_connectivity, _ = read_ch_connectivity('neuromag306grad')
        chan_types = [dict(name='mag', connectivity=mag_connectivity),
                      dict(name='grad', connectivity='missing')] + \
                     [chan for chan in chan_types if chan['name'] != 'meg']
    return chan_types


def resample_epochs(epochs, sfreq):
    """faster resampling"""
    # from librosa import resample
    # librosa.resample(channel, o_sfreq, sfreq, res_type=res_type)
    from scipy.signal import resample

    # resample
    epochs._data = resample(
        epochs._data, epochs._data.shape[2] / epochs.info['sfreq'] * sfreq,
        axis=2)
    # update metadata
    epochs.info['sfreq'] = sfreq
    epochs.times = (np.arange(epochs._data.shape[2],
                              dtype=np.float) / sfreq + epochs.times[0])
    return epochs


def decim(inst, decim):
    """faster resampling"""
    from mne.io.base import _BaseRaw
    from mne.epochs import _BaseEpochs
    if isinstance(inst, _BaseRaw):
        inst._data = inst._data[:, ::decim]
        inst.info['sfreq'] /= decim
        inst._first_samps /= decim
        inst.first_samp /= decim
        inst._last_samps /= decim
        inst.last_samp /= decim
        inst._raw_lengths /= decim
        inst._times = inst._times[::decim]
    elif isinstance(inst, _BaseEpochs):
        inst._data = inst._data[:, :, ::decim]
        inst.info['sfreq'] /= decim
        inst.times = inst.times[::decim]
    return inst


# DECODING ####################################################################


def scorer_angle_tuning(truth, prediction, n_bins=19):
    """WIP XXX should be transformed into a scorer?"""
    prediction = np.array(prediction)
    truth = np.array(truth)
    error = (np.pi - prediction + truth) % (2 * np.pi) - np.pi
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    h, _ = np.histogram(error, bins)
    h /= sum(h)
    return h


def scorer_angle_discrete(truth, prediction):
    """WIP Scoring function dedicated to SVC_angle"""
    n_trials, n_angles = prediction.shape
    angles = np.linspace(0, 2 * np.pi * (1 - 1 / n_angles), n_angles)
    x_pred = prediction * tile_memory_free(np.cos(angles), [n_trials, 1]).T
    y_pred = prediction * tile_memory_free(np.sin(angles), [n_trials, 1]).T
    angle_pred = np.arctan2(y_pred.mean(1), x_pred.mean(1))
    angle_error = truth - angle_pred
    pi = np.pi
    score = np.mean(np.abs((angle_error + pi) % (2 * pi) - pi))
    return pi / 2 - score


def scorer_angle(truth, prediction):
    """Scoring function dedicated to SVR_angle"""
    angle_error = truth - prediction[:, 0]
    pi = np.pi
    score = np.mean(np.abs((angle_error + pi) % (2 * pi) - pi))
    return np.pi / 2 - score


def scorer_circLinear(y_line, y_circ):
    R, R2, pval = corr_linear_circular(y_line, y_circ)
    return R


def get_predict(gat, sel=None, toi=None, mean=True, typ='diagonal'):
    from jr.gat import get_diagonal_ypred
    from jr.utils import align_on_diag
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


# LOAD DATA ###################################################################


def fix_wrong_channel_names(inst):
    # FIX at loading. Remove function if it works
    return inst
    # from mne.epochs import EpochsArray
    # from mne.evoked import Evoked
    # inst.info['chs'] = inst.info['chs'][:306]
    # inst.info['ch_names'] = inst.info['ch_names'][:306]
    # inst.info['nchan'] = 306
    # if isinstance(inst, Evoked):
    #     inst.data = inst.data[:306, :]
    # elif isinstance(inst, EpochsArray):
    #     inst._data = inst._data[:, :306, :]
    # else:
    #     raise ValueError('Unknown instance')
    # return inst


def load_epochs_events(subject, paths=None, data_type='erf',
                       lock='target'):
    # Get MEG data
    meg_fname = paths('epoch', subject=subject, data_type=data_type, lock=lock)
    epochs = load_FieldTrip_data(meg_fname)
    # epochs = fix_wrong_channel_names(epochs)  FIX at loading
    # Get behavioral data
    bhv_fname = paths('behavior', subject=subject)
    events = get_events(bhv_fname)
    epochs.crop(-.200, 1.200)
    return epochs, events


def angle2circle(angles):
    """from degree to radians multipled by rm2"""
    return np.deg2rad(2 * (np.array(angles) + 7.5))


def load_FieldTrip_data(meg_fname):
    import scipy.io as sio
    from mne.io.meas_info import create_info
    from mne.epochs import EpochsArray
    """XXX Here explain what this does"""
    # import information from fieldtrip data to get data shape
    ft_data = sio.loadmat(meg_fname[:-4] + '.mat', squeeze_me=True,
                          struct_as_record=True)['data']
    # import binary MEG data
    bin_data = np.fromfile(meg_fname[:-4] + '.dat', dtype=np.float32)
    Xdim = ft_data['Xdim'].item()
    bin_data = np.reshape(bin_data, Xdim[[2, 1, 0]]).transpose([2, 1, 0])

    # Create an MNE Epoch
    n_trial, n_chans, n_time = bin_data.shape
    sfreq = ft_data['fsample'].item()
    time = ft_data['time'].item()[0]
    tmin = min(time)
    chan_names = [str(label) for label in ft_data['label'].item()]
    chan_types = np.squeeze(np.concatenate(
        (np.tile(['grad', 'grad', 'mag'], (1, 102)),
         np.tile('misc', (1, n_chans - 306))), axis=1))
    chan_names = np.array(chan_names)[:306].tolist()
    chan_types = np.array(chan_types)[:306].tolist()
    bin_data = bin_data[:, :306, :]
    info = create_info(chan_names, sfreq, chan_types)
    events = np.c_[np.cumsum(np.ones(n_trial)) * 5 * sfreq,
                   np.zeros(n_trial),
                   ft_data['trialinfo'].item()]
    epochs = EpochsArray(bin_data, info, events=np.array(events, int),
                         tmin=tmin, verbose=False)

    return epochs


def get_events(bhv_fname):
    import scipy.io as sio
    import pandas as pd
    # Load behavioral file
    trials = sio.loadmat(bhv_fname, squeeze_me=True,
                         struct_as_record=False)["trials"]

    def trial2event(trial):
        event = dict()
        # Change meaningless values with NaNs
        event['target_present'] = trial.present == 1
        event['discrim_pressed'] = trial.response_responsed == 1
        event['detect_pressed'] = trial.response_vis_responsed == 1
        nan_default = lambda check, value: value if check else np.nan
        check_present = lambda v: nan_default(event['target_present'], v)
        # discrim_pressed = lambda v: nan_default(event['discrim_pressed'], v)
        discrim_buttons = lambda v: nan_default(
            v in ['left_green', 'left_yellow'], 1. * (v == 'left_green'))
        detect_pressed = lambda v: nan_default(event['detect_pressed'], v)
        phasebin = lambda v: np.digitize(
            [v], np.linspace(0, 1, 7))[0] * 2 * np.pi / 6.
        # Target
        event['target_contrast'] = [0, .5, .75, 1][trial.contrast - 1]
        event['target_spatialFreq'] = check_present(
            trial.__getattribute__('lambda'))
        event['target_angle'] = check_present(trial.orientation * 30 - 15)
        event['target_circAngle'] = angle2circle(event['target_angle'])
        event['target_phase'] = check_present(phasebin(
                                              trial.gabors.target.phase))

        # Probe
        event['probe_angle'] = (trial.orientation * 30 - 15 +
                                trial.tilt * 30) % 180
        event['probe_circAngle'] = angle2circle(event['probe_angle'])
        event['probe_tilt'] = check_present(trial.tilt)
        event['probe_spatialFreq'] = \
            trial.gabors.probe.__getattribute__('lambda')
        event['probe_phase'] = phasebin(trial.gabors.probe.phase)
        # Response 1: forced choice discrimination
        event['discrim_button'] = discrim_buttons(trial.response_keyPressed)
        event['discrim_correct'] = check_present(trial.correct == 1)
        # Response 2: detection/visibility
        event['detect_button'] = \
            detect_pressed(trial.response_visibilityCode - 1)
        event['detect_seen'] = event['detect_button'] > 0
        if np.isnan(event['detect_button']):
            event['detect_seen'] = np.nan
        return event

    events = list()
    for t, trial in enumerate(trials):
        # define previous trial
        event = trial2event(trial)
        if t > 1:
            previous_event = trial2event(trials[t-1])
            for key in previous_event:
                event['previous_' + key] = previous_event[key]
        events.append(event)
    events = pd.DataFrame(events)
    return events


# SKLEARN #####################################################################
# FIXME: to be remove and use jr.gat instead


class clf_2class_proba(LogisticRegression):
    """Probabilistic SVC for 2 classes only"""
    def predict(self, x):
        probas = super(clf_2class_proba, self).predict_proba(x)
        return probas[:, 1]


class SVC_2class_proba(SVC):  # XXX not used
    """Probabilistic SVC for 2 classes only"""
    def predict(self, x):
        probas = super(clf_2class_proba, self).predict_proba(x)
        return probas[:, 1]


class SVR_angle(LinearSVR):

    def __init__(self):
        from sklearn.svm import LinearSVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        scaler_cos = StandardScaler()
        scaler_sin = StandardScaler()
        svr_cos = LinearSVR(C=1)
        svr_sin = LinearSVR(C=1)
        self.clf_cos = Pipeline([('scaler', scaler_cos), ('svr', svr_cos)])
        self.clf_sin = Pipeline([('scaler', scaler_sin), ('svr', svr_sin)])

    def fit(self, X, y):
        """
        Fit 2 regressors cos and sin of angles y
        Parameters
        ----------
        X : np.array, shape(n_trials, n_chans, n_time)
            MEG data
        y : list | np.array (n_trials)
            angles in degree
        """
        # Go from orientation space (0-180 degrees) to complex space
        # (0 - 2 pi radians)
        self.clf_cos.fit(X, np.cos(y))
        self.clf_sin.fit(X, np.sin(y))

    def predict(self, X):
        """
        Predict orientation from MEG data in radians
        Parameters
        ----------
        X : np.array, shape(n_trials, n_chans, n_time)
            MEG data
        Returns
        -------
        predict_angle : list | np.array, shape(n_trials)
            angle predictions in radian
        """
        predict_cos = self.clf_cos.predict(X)
        predict_sin = self.clf_sin.predict(X)
        predict_angle = np.arctan2(predict_sin, predict_cos)
        return predict_angle
