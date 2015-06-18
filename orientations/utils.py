import os.path as op
import pickle
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVR, SVC
from sklearn.linear_model import LogisticRegression


def load_epochs_events(subject, paths=None, data_type='erf',
                       lock='target'):
    # Get MEG data
    meg_fname = paths('epoch', subject=subject, data_type=data_type, lock=lock)
    epochs = load_FieldTrip_data(meg_fname)
    # Get behavioral data
    bhv_fname = paths('behavior', subject=subject)
    events = get_events(bhv_fname)
    return epochs, events


def meg_to_gradmag(chan_types):
    """force separation of magnetometers and gradiometers"""
    if 'meg' in [chan['name'] for chan in chan_types]:
        chan_types = [dict(name='mag'), dict(name='grad')] + \
                     [dict(name=chan['name'])
                      for chan in chan_types
                      if chan['name'] != 'meg']
    return chan_types


def angle2circle(angles):
    """from degree to radians multipled by rm2"""
    return np.deg2rad(2 * (np.array(angles) + 7.5))


def load_FieldTrip_data(meg_fname):
    from mne.io.meas_info import create_info
    from mne.epochs import EpochsArray
    """load behavioural and meg data (erf and time freq)"""
    # import information from fieldtrip data to get data shape
    ft_data = sio.loadmat(meg_fname + '.mat', squeeze_me=True,
                          struct_as_record=True)['data']
    if meg_fname[-3:] == '.dat':
        # import binary MEG data
        bin_data = np.fromfile(meg_fname + '.dat', dtype=np.float32)
        Xdim = ft_data['Xdim'].item()
        bin_data = np.reshape(bin_data, Xdim[[2, 1, 0]]).transpose([2, 1, 0])
        # define data
        data = bin_data
        time = ft_data['time'].item()[0]

    else:
        # import structure MEG data
        matdata = ft_data['powspctrm'].item()
        # define data
        data = matdata
        time = ft_data['time'].item()

    # Create an MNE Epoch
    n_trial, n_chans, n_time = data.shape
    sfreq = 1. / (time[1] - time[0])
    tmin = min(time)
    chan_names = [str(label) for label in ft_data['label'].item()]
    chan_types = np.squeeze(np.concatenate(
        (np.tile(['grad', 'grad', 'mag'], (1, 102)),
         np.tile('misc', (1, n_chans - 306))), axis=1))
    info = create_info(chan_names, sfreq, chan_types)
    events = np.c_[np.cumsum(np.ones(n_trial)) * 5 * sfreq,
                   np.zeros(n_trial),
                   ft_data['trialinfo'].item()]
    epochs = EpochsArray(data, info, events=events, tmin=tmin)
    return epochs


def get_events(bhv_fname):
    # Load behavioral file
    trials = sio.loadmat(bhv_fname, squeeze_me=True,
                         struct_as_record=True)["trials"]

    keys = ['ISI', 'response_keyPressed', 'time_jitter',
            'block', 'response_responsed', 'time_maskIn',
            'break', 'response_tilt', 'time_preparation',
            'contrast', 'response_time', 'time_probe',
            'correct', 'response_vis_RT', 'time_prompt',
            'feedback', 'response_vis_keyPressed', 'time_response',
            'gabors', 'response_vis_responsed', 'time_targetIn',
            'lambda', 'response_vis_time', 'time_targetOut',
            'localizer', 'response_visibilityCode', 'trialid',
            'orientation', 'tilt', 'ttl_value',
            'present', 'time_delay', 'response_RT', 'time_feedback_on']

    contrasts = [0, .5, .75, 1]
    events = list()
    for t, trial in enumerate(trials):
        event = dict()
        # define previous trial
        prevtrial = trials[t-1]
        for key in keys:
                event[key] = trial[key]
        # manual new keys
        event['orientation_target'] = event['orientation']*30-15
        event['orientation_probe'] = (event['orientation']*30-15 +
                                      event['tilt'] * 30) % 180
        event['orientation_target_rad'] = angle2circle(
            event['orientation_target'])
        event['orientation_probe_rad'] = angle2circle(
            event['orientation_probe'])
        event['targetContrast'] = contrasts[event['contrast']-1]
        event['seen_unseen'] = event['response_visibilityCode'] > 1
        event['previous_trial_visibility'] = prevtrial[
            'response_visibilityCode']
        event['previous_orientation_target'] = prevtrial[
            'orientation']*30-15
        event['previous_orientation_probe'] = (prevtrial[
            'orientation']*30-15 +
            trial['tilt'] * 30) % 180

        # append to all events
        events.append(event)
    events = pd.DataFrame(events)
    return events


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


class clf_2class_proba(LogisticRegression):
    """Probabilistic SVC for 2 classes only"""
    def predict(self, x):
        probas = super(clf_2class_proba, self).predict_proba(x)
        return probas[:, 1]


class SVC_2class_proba(SVC):
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


def format_analysis(contrast):
    """This functions takes the contrasts defined for decoding  and format it
    so as to be usable by the univariate scripts

    We need to homogeneize the two types of analysis definitions
     """
    name = contrast['name']
    if contrast['scorer'] == scorer_spearman:
        operator = evoked_spearman
    elif contrast['scorer'] == scorer_auc:
        operator = evoked_subtract
    elif contrast['scorer'] == scorer_angle:
        # TODO evoked_vtest
        return
    # exclude
    exclude = dict()
    for exclude_ in contrast['exclude']:
        cond = exclude_['cond']
        exclude[cond] = exclude_['values']

    # include
    conditions = list()
    cond = contrast['include']['cond']
    for value in contrast['include']['values']:
        include_ = dict()
        include_[cond] = value
        conditions.append(dict(name=cond + str(value), include=include_,
                               exclude=exclude))
    analysis = dict(name=name, operator=operator, conditions=conditions)
    return analysis


def build_analysis(evoked_list, epochs, events, operator=None):
    """Builds a n-deep analysis where n represents different levels of analyses
    Parameters
    ----------
    evoked_list : dict
    epochs
    events
    operator

    Returns
    -------
    ceof : evoked
        contrast
    evokeds : list
        list of average evoked conditions
    e.g. XXX Make example
    """

    evokeds = dict()
    evokeds['evokeds'] = list()  # list of all evoked from lower level
    evokeds['coef'] = list()  # evoked of analysis

    # Accept passing lists only
    if type(evoked_list) is list:
        evoked_list_ = evoked_list
        evoked_list = dict(conditions=evoked_list_)

    # Gather coef at each sublevel
    for evoked in evoked_list['conditions']:
        if 'include' in evoked.keys():
            # Default exclude condition
            if 'exclude' not in evoked.keys():
                evoked['exclude'] = dict()
            # Find corresponding samples
            sel = find_in_df(events, evoked['include'], evoked['exclude'])
            # if no sample in conditions, throw error: XXX JRK: need fix
            if not len(sel):
                raise RuntimeError('no epoch in %s' % evoked['name'])
            # Average
            avg = epochs[sel].average()
            # Keep info
            avg.comment = evoked['name']
            evokeds['coef'].append(avg)
        else:
            if 'operator' not in evoked_list.keys():
                evoked_list['operator'] = None
            coef_, evokeds_ = build_analysis(evoked, epochs, events,
                                             evoked_list['operator'])
            evokeds['evokeds'].append(evokeds_)
            evokeds['coef'].append(coef_)
    else:
        # Set default operation
        if operator is None:
            if len(evokeds['coef']) == 2:
                operator = evoked_subtract
            elif len(evokeds['coef']) > 2:
                operator = evoked_spearman

        coef = operator(evokeds)

    return coef, evokeds


def evoked_subtract(evokeds):
    evokeds['coef'][0].nave = 1
    evokeds['coef'][-1].nave = 1
    coef = evoked_weighted_subtract(evokeds)
    return coef


def evoked_weighted_subtract(evokeds):
    import warnings
    if len(evokeds['coef']) > 2:
        warnings.warn('More than 2 categories. Subtract last from last'
                      'category!')
    coef = evokeds['coef'][0] - evokeds['coef'][-1]
    return coef


def evoked_spearman(evokeds):
    from scipy.stats import spearmanr
    n_chan, n_time = evokeds['coef'][0].data.shape
    coef = np.zeros((n_chan, n_time))
    # TODO: need parallelization
    for chan in range(n_chan):
        for t in range(n_time):
            y = range(len(evokeds['coef']))
            X = list()
            for i in y:
                X.append(evokeds['coef'][i].data[chan, t])
            coef[chan, t], _ = spearmanr(X, y)
    evoked = evokeds['coef'][0]
    evoked.data = coef
    return evoked


def evoked_circularlinear(epochs, angles):
    # from pycircstat.regression import CL1stOrderRegression
    # regress = CL1stOrderRegression()
    n_trials, n_chans, n_times = epochs._data.shape
    # transform to 2D array
    x = epochs._data.reshape(n_trials, n_chans * n_times)
    # duplicate angles for each time / chan
    angles = np.tile(angles, [x.shape[1], 1]).T
    # compute
    r, r2, pval = circular_linear_correlation(angles, x)
    # store in format readable to build_analysis
    evoked = epochs.average()
    evoked.data = r.reshape(n_chans, n_times)
    return evoked


def circular_linear_correlation(alpha, x):
    # Authors:  Jean-Remi King <jeanremi.king@gmail.com>
    #           Niccolo Pescetelli <niccolo.pescetelli@gmail.com>
    #
    # Licence : BSD-simplified
    """

    Parameters
    ----------
        alpha : numpy.array, shape (n_angles, n_dims)
            The angular data (if n_dims == 1, repeated across all x dimensions)
        x : numpy.array, shape (n_angles, n_dims)
            The linear data
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

    def corr(X, Y):
        if X.ndim == 1:
            X = X[:, None]
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.shape != X.shape:
            raise ValueError('X and Y must have identical shapes.')
        coef = np.nan * np.zeros(X.shape[1])
        for idx, (x, y) in enumerate(zip(X.T, Y.T)):
            coef[idx] = np.corrcoef(x, y)[0, 1]
        return coef

    # computes correlation for sin and cos separately
    alpha = alpha % (2 * np.pi) - np.pi
    rxs = corr(x, np.sin(alpha))
    rxc = corr(x, np.cos(alpha))
    rcs = corr(np.sin(alpha), np.cos(alpha))

    # Adapted from equation 27.47
    R = (rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2)
    R2 = np.sqrt(R ** 2)

    # Get degrees of freedom
    n = len(alpha)
    pval = 1 - chi2.cdf(n * R2, 2)

    return R, R2, pval


def save_to_dict(fname, data, overwrite=False):
    """Add pickle object to file without replacing its content using a
    dictionary format which keys' correspond to the names of the variables.
    Parameters
    ----------
    fname : str
        file name
    data : dict
    overwrite : bool
        Default: False
    """
    # Identify whether the file exists
    if op.isfile(fname) and not overwrite:
        data_dict = load_from_dict(fname)
    else:
        data_dict = dict()

    for key in data.keys():
        data_dict[key] = data[key]

    # Save
    with open(fname, 'wb') as f:
        pickle.dump(data_dict, f)


def load_from_dict(fname, varnames=None, out_type='dict'):
    """Load pickle object from file using a dictionary format which keys'
     correspond to the names of the variables.
    Parameters
    ----------
    fname : str
        file name
    varnames : None | str | list (optional)
        Variables to load. By default, load all of them.
    out_type : str
        'list', 'dict': default: dict
    Returns
    -------
    vars : dict
        dictionary of loaded variables which keys corresponds to varnames
    """

    # Identify whether the file exists
    if not op.isfile(fname):
        raise RuntimeError('%s not found' % fname)

    # Load original data
    with open(fname, 'rb') as f:
        data_dict = pickle.load(f)

    # Specify variables to load
    if not varnames:
        varnames = data_dict.keys()
    elif varnames is str:
        varnames = [varnames]

    # Append result in a list
    if out_type == 'dict':
        out = dict()
        for key in varnames:
            out[key] = data_dict[key]
    elif out_type == 'list':
        out = list()
        for key in varnames:
            out.append(data_dict[key])

    return out


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


def fill_betweenx_discontinuous(ax, ymin, ymax, x, freq=1, **kwargs):
    """Fill betwwen x even if x is discontinuous clusters
    Parameters
    ----------
    ax : axis
    x : list

    Returns
    -------
    ax : axis
    """
    x = np.array(x)
    min_gap = (1.1 / freq)
    while np.any(x):
        # If with single time point
        if len(x) > 1:
            xmax = np.where((x[1:] - x[:-1]) > min_gap)[0]
        else:
            xmax = [0]

        # If continuous
        if not np.any(xmax):
            xmax = [len(x) - 1]

        ax.fill_betweenx((ymin, ymax), x[0], x[xmax[0]], **kwargs)

        # remove from list
        x = x[(xmax[0] + 1):]
    return ax


def Evokeds_to_Epochs(inst, info=None, events=None):
    """Convert list of evoked into single epochs

    Parameters
    ----------
    inst: list
        list of evoked objects.
    info : dict
        By default copy dict from inst[0]
    events : np.array (dims: n, 3)
    Returns
    -------
    epochs: epochs object"""
    from mne.epochs import EpochsArray
    from mne.evoked import Evoked

    if (not(isinstance(inst, list)) or
            not np.all([isinstance(x, Evoked) for x in inst])):
        raise('inst mus be a list of evoked')

    # concatenate signals
    data = [x.data for x in inst]
    # extract meta data
    if info is None:
        info = inst[0].info
    if events is None:
        n = len(inst)
        events = np.c_[np.cumsum(np.ones(n)) * info['sfreq'],
                       np.zeros(n), np.ones(n)]

    return EpochsArray(data, info, events=events, tmin=inst[0].times.min())


class cluster_stat(dict):
    """ Cluster statistics """
    def __init__(self, epochs, alpha=0.05, **kwargs):
        """
        Parameters
        ----------
        X : np.array (dims = n * space * time)
            data array
        alpha : float
            significance level

        Can take spatio_temporal_cluster_1samp_test() parameters.

        """
        from mne.stats import spatio_temporal_cluster_1samp_test

        # Convert lists of evoked in Epochs
        if isinstance(epochs, list):
            epochs = Evokeds_to_Epochs(epochs)
        X = epochs._data.transpose((0, 2, 1))

        # Apply contrast: n * space * time

        # Run stats
        self.T_obs_, clusters, p_values, _ = \
            spatio_temporal_cluster_1samp_test(X, out_type='mask', **kwargs)

        # Save sorted sig clusters
        inds = np.argsort(p_values)
        clusters = np.array(clusters)[inds, :, :]
        p_values = p_values[inds]
        inds = np.where(p_values < alpha)[0]
        self.sig_clusters_ = clusters[inds, :, :]
        self.p_values_ = p_values[inds]

        # By default, keep meta data from first epoch
        self.epochs = epochs
        self.times = self.epochs[0].times
        self.info = self.epochs[0].info
        self.ch_names = self.epochs[0].ch_names

        return

    def _get_mask(self, i_clu):
        """
        Selects or combine clusters

        Parameters
        ----------
        i_clu : int | list | array
            cluster index. If list or array, returns average across multiple
            clusters.

        Returns
        -------
        mask : np.array
        space_inds : np.array
        times_inds : np.array
        """
        # Select or combine clusters
        if i_clu is None:
            i_clu = range(len(self.sig_clusters_))
        if isinstance(i_clu, int):
            mask = self.sig_clusters_[i_clu]
        else:
            mask = np.sum(self.sig_clusters_[i_clu], axis=0)

        # unpack cluster infomation, get unique indices
        space_inds = np.where(np.sum(mask, axis=0))[0]
        time_inds = np.where(np.sum(mask, axis=1))[0]

        return mask, space_inds, time_inds

    def plot_topo(self, i_clu=None, pos=None, **kwargs):
        """
        Plots fmap of one or several clusters.

        Parameters
        ----------
        i_clu : int
            cluster index

        Can take evoked.plot_topomap() parameters.

        Returns
        -------
        fig
        """
        from mne import find_layout
        from mne.viz import plot_topomap

        # Channel positions
        pos = find_layout(self.info).pos
        # create topomap mask from sig cluster
        mask, space_inds, time_inds = self._get_mask(i_clu)

        if pos is None:
            pos = find_layout(self.info).pos

        # plot average test statistic and mark significant sensors
        topo = self.T_obs_[time_inds, :].mean(axis=0)
        fig = plot_topomap(topo, pos, **kwargs)

        return fig

    def plot_topomap(self, i_clu=None, **kwargs):
        """
        Plots effect topography and highlights significant selected clusters.

        Parameters
        ----------
        i_clu : int
            cluster index

        Can take evoked.plot_topomap() parameters.

        Returns
        -------
        fig
        """
        # create topomap mask from sig cluster
        mask, space_inds, time_inds = self._get_mask(i_clu)

        # plot average test statistic and mark significant sensors
        evoked = self.epochs.average()
        evoked.data = self.T_obs_.transpose()
        fig = evoked.plot_topomap(mask=np.transpose(mask), **kwargs)

        return fig

    def plot(self, plot_type='butterfly', i_clus=None, axes=None, show=True,
             **kwargs):
        """
        Plots effect time course and highlights significant selected clusters.

        Parameters
        ----------
        i_clus : None | list | int
            cluster indices
        plot_type : str
            'butterfly' to plot differential response across all channels
            'cluster' to plot cluster time course for each condition

        Can take evoked.plot() parameters.

        Returns
        -------
        fig
        """
        import matplotlib.pyplot as plt

        times = self.times * 1000

        # if axes is None:
        if True:
            fig = plt.figure()
            fig.add_subplot(111)
            axes = fig.axes[0]

        # By default, plot separate clusters
        if i_clus is None:
            if plot_type == 'butterfly':
                i_clus = [None]
            else:
                i_clus = range(len(self.sig_clusters_))
        elif isinstance(i_clus, int):
            i_clus = [i_clus]

        # Time course
        if plot_type == 'butterfly':
            # Plot butterfly of difference
            evoked = self.epochs.average()
            fig = evoked.plot(axes=axes, show=False, **kwargs)

        # Significant times
        ymin, ymax = axes.get_ylim()
        for i_clu in i_clus:
            _, _, time_inds = self._get_mask(i_clu)
            sig_times = times[time_inds]

            fill_betweenx_discontinuous(axes, ymin, ymax, sig_times,
                                        freq=(self.info['sfreq'] / 1000),
                                        color='orange', alpha=0.3)

        axes.legend(loc='lower right')
        axes.set_ylim(ymin, ymax)

        # add information
        axes.axvline(0, color='k', linestyle=':', label='stimulus onset')
        axes.set_xlim([times[0], times[-1]])
        axes.set_xlabel('Time [s]')
        axes.set_ylabel('Evoked magnetic fields [fT]')

        if show:
            plt.show()

        return fig


def find_in_df(df, include, exclude=dict(), max_n=np.inf):
    """Find instance in pd.dataFrame that correspond to include and exlcuding
    criteria.

    Parameters
    ----------
    df : pd.dataFrame
    includes : list | dict()
    excludes : list | dict()
    Returns
    -------
    inds : np.array"""
    import random

    # Find included trials
    include_inds = _find_in_df(df, include)
    # Find excluded trials
    exclude_inds = _find_in_df(df, exclude)

    # Select condition
    inds = [i for i in include_inds if i not in exclude_inds]

    # reduce number or trials if too many
    if len(inds) > max_n:
        random.shuffle(inds)
        inds = inds[:max_n]

    return inds


def _find_in_df(df, le_dict):
    """Find all instances in pd dataframe that match one of the specified
    conditions"""
    inds = []
    for key in le_dict.keys():
        if type(le_dict[key]) not in (list, np.ndarray):
            le_dict[key] = [le_dict[key]]
        for value in le_dict[key]:
            for i in np.where(df[key] == value)[0]:
                inds.append(i)
    inds = np.unique(inds)
    return inds
