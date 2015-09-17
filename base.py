import numpy as np
from jr.utils import tile_memory_free, pairwise
from jr.stats import repeated_spearman, repeated_corr
from jr.gat.scorers import scorer_auc, scorer_spearman

# ANALYSES ####################################################################


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
    # Binary contrast
    unique_y = np.unique(y)
    if len(unique_y) == 2:
        y = np.where(y == unique_y[0], 1, -1)
        # Tile Y to across X dimension without allocating memory
        Y = tile_memory_free(y, X.shape[1:])
        return np.mean(X * Y, axis=0)

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

# def get_tuning_curve(truth, prediction, n_bin=20):
#     """WIP XXX should be transformed into a scorer?"""
#     from itertools import product
#     error = (np.pi - np.squeeze(prediction) +
#              np.transpose(tile_memory_free(truth, np.shape(prediction)[:2]),
#                           [1, 2, 0])) % (2 * np.pi) - np.pi
#     bins = np.linspace(-np.pi, np.pi, n_bin)
#     nT, nt, _, _ = np.shape(prediction)
#     h = np.zeros((nT, nt, n_bin - 1))
#     for T, t in product(range(nT), range(nt)):
#         h[T, t, :], _ = np.histogram(error[T, t, :], bins)
#         h[T, t, :] /= sum(h[T, t, :])
#     return h


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
    R, R2, pval = circular_linear_correlation(y_line, y_circ)
    return R
