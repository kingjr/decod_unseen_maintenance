import numpy as np
from jr.utils import tile_memory_free, pairwise
from jr.stats import repeated_spearman, fast_mannwhitneyu
from mne.stats import ttest_1samp_no_p
from mne.stats import spatio_temporal_cluster_1samp_test


# STATISTICS ##################################################################


def _stat_fun(x, sigma=0, method='relative'):
    """Aux. function of stats"""
    t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
    t_values[np.isnan(t_values)] = 0
    return t_values


def stats(X, connectivity=None, n_jobs=-1):
    """Cluster statistics to control for multiple comparisons.

    Parameters
    ----------
    X : array, shape (n_samples, n_space, n_times)
        The data, chance is assumed to be 0.
    connectivity : None | array, shape (n_space, n_times)
        The connectivity matrix to apply cluster correction. If None uses
        neighboring cells of X.
    n_jobs : int
        The number of parallel processors.
    """
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask', stat_fun=_stat_fun, n_permutations=2**12,
        n_jobs=n_jobs, connectivity=connectivity)
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
    return np.squeeze(p_values_).T

# ANALYSES ####################################################################


def nested_analysis(X, df, condition, function=None, query=None,
                    single_trial=False, y=None, n_jobs=-1):
    """ Apply a nested set of analyses. Note that this is an overkill in this
    study has only main effects are modeled.

    This pipeline is to manually identify main effects in the case of multiple
    independent variables. In the present study, it is an overkill, and could
    be simplified.

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
                        single_trial=True)
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
    """Aux. function to nested_analysis"""
    # Binary contrast
    unique_y = np.unique(y)
    # if two condition, can only return contrast
    if len(y) == 2:
        y = np.where(y == unique_y[0], 1, -1)
        # Tile Y to across X dimension without allocating memory
        Y = tile_memory_free(y, X.shape[1:])  # FIXME tile memory free is now automatic in numpy # noqa
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


# LOAD DATA ###################################################################


def read_events(bhv_fname):
    """Reads events from mat file, convert and clean them into readable
    variables."""
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
        # XXX contrast should be 25, 75 or 100 FIXME
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
        events.append(event)
    events = pd.DataFrame(events)
    return events


def angle2circle(angles):
    """from degree to radians multipled by 2"""
    return np.deg2rad(2 * (np.array(angles) + 7.5))
