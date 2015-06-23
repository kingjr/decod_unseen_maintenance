import sys
sys.path.insert(0, './')
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

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
        values = [ii for ii in np.unique(y) if ~np.isnan(ii)]
        # Subsubselect for each unique condition
        y_sel = [np.where(y == value)[0] for value in values]
        # Mean condition:
        X_mean = np.zeros(np.hstack((len(y_sel), X.shape[1:])))
        y_mean = np.zeros(len(y_sel))
        for ii, sel_ in enumerate(y_sel):
            X_mean[ii, ...] = np.mean(X[sel_, ...], axis=0)
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
            X, sub = nested_analysis(
                X, df, subcondition, n_jobs=n_jobs,
                function=subcondition.get('function', None),
                query=subcondition.get('query', None))
            X_list.append(X)
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


def tile_memory_free(y, shape):
    """
    Tile vector along multiple dimension without allocating new memory.

    Parameters
    ----------
     y : np.array, shape (n,)
        data
    shape : np.array, shape (m),
    Returns
    -------
    Y : np.array, shape (n, *shape)
    """
    y = np.lib.stride_tricks.as_strided(y,
                                        (np.prod(shape), y.size),
                                        (0, y.itemsize)).T
    return y.reshape(np.hstack((len(y), shape)))


def test_tile_memory_free():
    from nose.tools import assert_equal
    y = np.arange(100)
    Y = tile_memory_free(y, 33)
    assert_equal(y.shape[0], Y.shape[0])
    np.testing.assert_array_equal(y, Y[:, 0], Y[:, -1])


def pairwise(X, y, func, n_jobs=-1):
    """Applies pairwise operations on two matrices using multicore:
    function(X[:, jj, kk, ...], y[:, jj, kk, ...])

    Parameters
    ----------
        X : np.ndarray, shape(n, ...)
        y : np.array, shape(n, ...) | shape(n,)
            If shape == X.shape:
                parallel(X[:, chunk], y[:, chunk ] for chunk in n_chunks)
            If shape == X.shape[0]:
                parallel(X[:, chunk], y for chunk in n_chunks)
        func : function
        n_jobs : int, optional
            Number of parallel cpu.
    Returns
    -------
        out : np.array, shape(func(X, y))
    """
    import numpy as np
    from mne.parallel import parallel_func
    dims = X.shape
    if y.shape[0] != dims[0]:
        raise ValueError('X and y must have identical shapes')

    X.resize([dims[0], np.prod(dims[1:])])
    if y.ndim > 1:
        Y = np.reshape(y, [dims[0], np.prod(dims[1:])])

    parallel, pfunc, n_jobs = parallel_func(func, n_jobs)

    n_cols = X.shape[1]
    n_chunks = min(n_cols, n_jobs)
    chunks = np.array_split(range(n_cols), n_chunks)
    if y.ndim == 1:
        out = parallel(pfunc(X[:, chunk], y) for chunk in chunks)
    else:
        out = parallel(pfunc(X[:, chunk], Y[:, chunk]) for chunk in chunks)

    # size back in case higher dependencies
    X.resize(dims)

    # unpack
    if isinstance(out[0], tuple):
        return [np.reshape(out_, dims[1:]) for out_ in zip(*out)]
    else:
        return np.reshape(out, dims[1:])


def _dummy_function_1(x, y):
    return x[0, :]


def _dummy_function_2(x, y):
    return x[0, :], 0. * x[0, :]


def test_pairwise():
    from nose.tools import assert_equal, assert_raises
    n_obs = 20
    n_dims1 = 5
    n_dims2 = 10
    y = np.linspace(0, 1, n_obs)
    X = np.zeros((n_obs, n_dims1, n_dims2))
    for dim1 in range(n_dims1):
        for dim2 in range(n_dims2):
            X[:, dim1, dim2] = dim1 + 10*dim2

    # test size
    score = pairwise(X, y, _dummy_function_1, n_jobs=2)
    assert_equal(score.shape, X.shape[1:])
    np.testing.assert_array_equal(score[:, 0], np.arange(n_dims1))
    np.testing.assert_array_equal(score[0, :], 10 * np.arange(n_dims2))

    # Test that X has not changed becaus of resize
    np.testing.assert_array_equal(X.shape, [n_obs, n_dims1, n_dims2])

    # test multiple out
    score1, score2 = pairwise(X, y, _dummy_function_2, n_jobs=2)
    np.testing.assert_array_equal(score1[:, 0], np.arange(n_dims1))
    np.testing.assert_array_equal(score2[:, 0], 0 * np.arange(n_dims1))

    # Test array vs vector
    score1, score2 = pairwise(X, X, _dummy_function_2, n_jobs=1)

    # test error check
    assert_raises(ValueError, pairwise, X, y[1:], _dummy_function_1)
    assert_raises(ValueError, pairwise, y, X, _dummy_function_1)


# PLOT ########################################################################

def plot_eb(x, y, yerr, ax=None, alpha=0.3, color=None, line_args=dict(),
            err_args=dict()):
    """
    Parameters
    ----------
    x : list | np.array()
    y : list | np.array()
    yerr : list | np.array() | float
    ax
    alpha
    color
    line_args
    err_args

    Returns
    -------
    ax

    Adapted from http://tonysyu.github.io/plotting-error-bars.html#.VRE9msvmvEU
    """
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y,  color=color, **line_args)
    ax.fill_between(x, ymax, ymin, alpha=alpha, color=color, **err_args)

    return ax


def share_clim(axes, clim=None):
    """Share clim across multiple axes
    Parameters
    ----------
    axes : plt.axes
    clim : np.array | list, shape(2,), optional
        Defaults is min and max across axes.clim.
    """
    # Find min max of clims
    if clim is None:
        clim = list()
        for ax in axes:
            for im in ax.get_images():
                clim += np.array(im.get_clim()).flatten().tolist()
        clim = [np.min(clim), np.max(clim)]
    # apply common clim
    for ax in axes:
        for im in ax.get_images():
            im.set_clim(clim)
    plt.draw()

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


def scorer_angle(truth, prediction):
    """Scoring function dedicated to SVR_angle"""
    angle_error = truth - prediction[:, 0]
    pi = np.pi
    score = np.mean(np.abs((angle_error + pi) % (2 * pi) - pi))
    return score


def scorer_auc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelBinarizer
    """Dedicated to 2class probabilistic outputs"""
    le = LabelBinarizer()
    y_true = le.fit_transform(y_true)
    return roc_auc_score(y_true, y_pred)


def scorer_spearman(y_true, y_pred):
    """"Dedicated to standard SVR"""
    from scipy.stats import spearmanr
    rho, p = spearmanr(y_true, y_pred[:, 0])
    return rho


def scorer_circLinear(y_line, y_circ):
    R, R2, pval = circular_linear_correlation(y_line, y_circ)
    return R


def circular_linear_correlation(X, alpha):
    # Authors:  Jean-Remi King <jeanremi.king@gmail.com>
    #           Niccolo Pescetelli <niccolo.pescetelli@gmail.com>
    #
    # Licence : BSD-simplified
    """

    Parameters
    ----------
        X : numpy.array, shape (n_angles, n_dims)
            The linear data
        alpha : numpy.array, shape (n_angles,)
            The angular data (if n_dims == 1, repeated across all x dimensions)
    Returns
    -------
        R : numpy.array, shape (n_dims)
            R values
        R2 : numpy.array, shape (n_dims)
            R square values
        p_val : numpy.array, shape (n_dims)
            P values

    Adapted from:
        Circular Statistics Toolbox for Matlab
        By Philipp Berens, 2009
        berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html
        Equantion 27.47
    """

    from scipy.stats import chi2
    import numpy as np

    # computes correlation for sin and cos separately
    rxs = repeated_corr(X, np.sin(alpha))
    rxc = repeated_corr(X, np.cos(alpha))
    rcs = repeated_corr(np.sin(alpha), np.cos(alpha))

    # tile alpha across multiple dimension without requiring memory
    if X.ndim > 1 and alpha.ndim == 1:
        rcs = tile_memory_free(rcs, X.shape[1:])

    # Adapted from equation 27.47
    R = (rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2)

    # JR adhoc way of having a sign....
    R = np.sign(rxs) * np.sign(rxc) * R
    R2 = np.sqrt(R ** 2)

    # Get degrees of freedom
    n = len(alpha)
    pval = 1 - chi2.cdf(n * R2, 2)

    return R, R2, pval


def repeated_spearman(X, y, dtype=None):
    """Computes spearman correlations between a vector and a matrix.

    Parameters
    ----------
        X : np.array, shape (n_samples, n_measures)
            Data matrix onto which the vector is correlated.
        y : np.array, shape (n_samples)
            Data vector.
        dtype : type, optional
            Data type used to compute correlation values to optimize memory.

    Returns
    -------
        rho : np.array, shape (n_measures)
    """
    if X.ndim not in [1, 2] or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError('y must be a vector, and X a matrix with an equal'
                         'number of rows.')
    if X.ndim == 1:
        X = X[:, None]

    # Rank
    X = np.argsort(X, axis=0)
    y = np.argsort(y, axis=0)
    # Double rank to ensure that normalization step of compute_corr
    # (X -= mean(X)) remains an integer.
    if (dtype is None and X.shape[0] < 2 ** 8) or\
       (dtype in [int, np.int16, np.int32, np.int64]):
        X *= 2
        y *= 2
        dtype = np.int16
    else:
        dtype = type(y[0])
    X = np.array(X, dtype=dtype)
    y = np.array(y, dtype=dtype)
    return repeated_corr(X, y, dtype=type(y[0]))


def repeated_corr(X, y, dtype=float):
    """Computes pearson correlations between a vector and a matrix.

    Adapted from Jona-Sassenhagen's PR #L1772 on mne-python.

    Parameters
    ----------
        y : np.array, shape (n_samples)
            Data vector.
        X : np.array, shape (n_samples, n_measures)
            Data matrix onto which the vector is correlated.
        dtype : type, optional
            Data type used to compute correlation values to optimize memory.

    Returns
    -------
        rho : np.array, shape (n_measures)
    """
    from sklearn.utils.extmath import fast_dot
    if X.ndim not in [1, 2] or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError('y must be a vector, and X a matrix with an equal'
                         'number of rows.')
    if X.ndim == 1:
        X = X[:, None]
    y -= np.array(y.mean(0), dtype=dtype)
    X -= np.array(X.mean(0), dtype=dtype)
    y_sd = y.std(0, ddof=1)
    X_sd = X.std(0, ddof=1)[:, None if y.shape == X.shape else Ellipsis]
    return (fast_dot(y.T, X) / float(len(y) - 1)) / (y_sd * X_sd)


def test_corr_functions():
    from scipy.stats import spearmanr
    test_corr(np.corrcoef, repeated_corr, 1)
    test_corr(spearmanr, repeated_spearman, 0)


def test_corr(old_func, new_func, sel_item):
    from nose.tools import assert_equal, assert_raises
    n_obs = 20
    n_dims = 10
    y = np.linspace(0, 1, n_obs)
    X = np.tile(y, [n_dims, 1]).T + np.random.randn(n_obs, n_dims)
    rho_fast = new_func(X, y)
    # test dimensionality
    assert_equal(rho_fast.ndim, 1)
    assert_equal(rho_fast.shape[0], n_dims)
    # test data
    rho_slow = np.ones(n_dims)
    for dim in range(n_dims):
        rho_slow[dim] = np.array(old_func(X[:, dim], y)).item(sel_item)
    np.testing.assert_array_equal(rho_fast.shape, rho_slow.shape)
    np.testing.assert_array_almost_equal(rho_fast, rho_slow)
    # test errors
    new_func(np.squeeze(X[:, 0]), y)
    assert_raises(ValueError, new_func, y, X)
    assert_raises(ValueError, new_func, X, y[1:])
    # test dtype
    X = np.argsort(X, axis=0) * 2  # ensure no bug at normalization
    y = np.argsort(y, axis=0) * 2
    rho_fast = new_func(X, y, dtype=int)
    rho_slow = np.ones(n_dims)
    for dim in range(n_dims):
        rho_slow[dim] = np.array(old_func(X[:, dim], y)).item(sel_item)
    np.testing.assert_array_almost_equal(rho_fast, rho_slow)
