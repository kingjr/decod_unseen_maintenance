import numpy as np

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


def resample_epochs(epochs, sfreq):
    """faster resampling"""
    # from librosa import resample
    # librosa.resample(channel, o_sfreq, sfreq, res_type=res_type)
    from scipy.signal import resample

    # resample
    epochs._data = resample(epochs._data,
                            epochs._data.shape[2] / epochs.info['sfreq'] * sfreq,
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
         inst._data =  inst._data[:,::decim]
         inst.info['sfreq'] /= decim
         inst._first_samps /= decim
         inst.first_samp /= decim
         inst._last_samps /= decim
         inst.last_samp /= decim
         inst._raw_lengths /= decim
         inst._times =  inst._times[::decim]
    elif isinstance(inst, _BaseEpochs):
        inst._data = inst._data[:,:,::decim]
        inst.info['sfreq'] /= decim
        inst.times = inst.times[::decim]
    return inst


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
    def __init__(self, insts, alpha=0.05, **kwargs):
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
        from mne.channels import read_ch_connectivity

        # Convert lists of evoked in Epochs
        insts = [Evokeds_to_Epochs(i) if type(i) is list else i for i in insts]

        # Apply contrast: n * space * time
        X = np.array(insts[0]._data - insts[1]._data).transpose([0, 2, 1])

        # Run stats
        self.T_obs_, clusters, p_values, _ = \
            spatio_temporal_cluster_1samp_test(X, out_type='mask', **kwargs)

        # Save sorted sig clusters
        inds = np.argsort(p_values)
        clusters = np.array(clusters)[inds,:,:]
        p_values = p_values[inds]
        inds = np.where(p_values < alpha)[0]
        self.sig_clusters_ = clusters[inds,:,:]
        self.p_values_ = p_values[inds]

        # By default, keep meta data from first epoch
        self.insts = insts
        self.times = self.insts[0].times
        self.info = self.insts[0].info
        self.ch_names = self.insts[0].ch_names

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
        evoked = self.insts[0].average()
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
        from mne.viz.utils import COLORS

        times = self.times  * 1000

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
            evoked = self.insts[0].average() - self.insts[1].average()
            fig = evoked.plot(axes=axes, show=False, **kwargs)
        elif plot_type == 'cluster':
            evokeds = [x.average() for x in self.insts]
            for i_clu in i_clus:
                _, space_inds, _ = self._get_mask(i_clu)
                for i_evo, evoked in enumerate(evokeds):
                    signal = np.mean(evoked.data[space_inds,:],
                                     axis=0)
                    _kwargs = kwargs.copy()
                    _kwargs['color'] = COLORS[i_evo % len(COLORS)]
                    axes.plot(times, signal, **_kwargs)

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
