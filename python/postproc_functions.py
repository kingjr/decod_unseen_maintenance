""" In this page I store all functions used to post-process classifiers
results.
Niccolo Pescetelli niccolo.pescetelli@psy.ox.ac.uk
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import spatio_temporal_cluster_1samp_test
from toolbox.utils import (plot_eb, fill_betweenx_discontinuous)

def recombine_svr_prediction(gatx, gaty):
    """
    This function takes two classifiers SVR predictions, typically
    sine and cosine of an angle, and combine them into a predicted angle in
    radians
    """

    pi = np.pi

    # get true angle
    true_x = gatx.y_train_
    true_y = gaty.y_train_
    true_angles, _ = cart2pol(true_x, true_y)
    true_angles = np.squeeze(true_angles)

    # get x and y regressors (cos and sin)
    predict_x = np.array(gatx.y_pred_)
    predict_y = np.array(gaty.y_pred_)
    predict_angles, _ = cart2pol(predict_x, predict_y)
    predict_angles = np.squeeze(predict_angles)

    return predict_angles, true_angles


def compute_error_svr(predict_angles, true_angles):
    """ Add explanation and motivatin here """ # XXX
    # compute angle error
    pi = np.pi
    n_T, n_t, n = predict_angles.shape
    true_angles = np.tile(true_angles, (n_T, n_t, 1))
    # compute angle error
    angle_error = predict_angles - true_angles
    # center around 0
    angle_errors = (angle_error + pi) % (2 * pi) - pi
    angle_errors = np.abs(angle_errors)
    return angle_errors

def compute_error_svc(gat, weighted_mean=True):
    """ Add explanation and motivatin here """ # XXX
    # transform 6 categories into single angle: there are two options here,
    # try it on the pilot subject, and use weighted mean if the two are
    # equivalent
    pi = np.pi
    # realign to 0th angle category
    n_angle = 6
    angle = 2. * pi / n_angle
    angles = np.arange(n_angle) * angle
    # realign predictions so that the first class corresponds to a 0 degree angle error
    # Note that eh classifiers were trained on the 6 orientations:
    # 1. the y is is in degree
    # 2. the first angle starts at 15
    # 3. orientation can only spread on half of the circle
    weights = realign_angle(gat, np.rad2deg(angles / 2) + 15)
    if weighted_mean:
        n_T, n_t, n_trials, n_categories = weights.shape
        # multiply category ind (1, 2, 3, ) by angle, remove pi to be between
        # -pi and pi and remove pi/6 to center on 0
        angles = np.tile(angles, (n_T, n_t, n_trials, 1))
        # weighted mean in complex space
        x = np.mean(weights * np.cos(angles), axis=3)
        y = np.mean(weights * np.sin(angles), axis=3)
        angle_error, _ = cart2pol(x, y)
    else:
        # Chose the angle with the maximum weight
        angle_error = (np.argmax(weights, axis=3) * angle)
        # center around 0
        angle_error = (angle_error + pi) % (2 * pi) - pi
    angle_error = np.abs(angle_error)

    return angle_error

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    radius = np.sqrt(x ** 2 + y ** 2)
    return(theta, radius)

def realign_angle(gat, angles):
    """
    This function realign classes output by a classifier so to give a
    distance in terms of categories of the predicted class and the real class.
    Input:
    gat : the gat object output by a classifier
    optional values: angles
    Output:
    probas : a time x time x trials x class array
    """
    # define dimensionality of the data
    dims = np.array(np.shape(gat.y_pred_))
    # realign to 4th angle category
    probas = np.zeros(dims)
    n_classes = len(angles)
    for a, angle in enumerate(angles):
        sel = gat.y_train_ == angle
        prediction = np.array(gat.y_pred_)
        order = np.array(range(a,n_classes)+range(0,(a)))
        probas[:, :, sel, :] = prediction[:, :, sel, :][:, :, :, order]
        # shift so that the correct class is in the middle
    #probas = probas[:,:,:,np.append(np.arange(4,n_classes),np.arange(0,4))]
    return probas



def hist_tuning_curve(angle_errors, res=30):

    # define bin_edges
    bin_edges = lambda m, M, n: np.arange(m+(M-m)/n/2,(M+(M-m)/n/2),(M-m)/n)
    # compute proportion of trials correctly predicted
    N = histogramnd(angle_errors.squeeze(),
                                    bins=bin_edges(-np.pi/2,np.pi/2,res+1),
                                    axis=2)
    # extract frequencies
    trial_freq = N[0]

    # Wrap around first and last bin if identical
    if False:
        trial_freq[1,:,:] = trial_freq[1,:,:]+trial_freq[-1,:,:]
        trial_freq[-1,:,:] = trial_freq[1,:,:]

    #compute totals
    totals = np.tile(np.sum(trial_freq,axis=2),[res,1,1])
    # compute the proportion of trials in each bin
    trial_prop = trial_freq / totals.astype(np.float).transpose([1, 2, 0])
    return trial_prop

def plot_circ_hist(alpha, bins=10, measure='radians'):
    """
    This function plots a polar histogram of the distribution of circular data.
    Input:
    alpha: 1D distribution of circular data in radians
    bins=number of bins the distribution has to be broken in. Default = 10
    measure= plot labels: radians (default) | degrees
    Output:
    none
    """

    bottom = 8
    max_height = 4

    hist, bin_edges = np.histogram(np.array(alpha),bins=80)
    N = len(hist)
    radii = np.ones(bin_edges.shape)
    width = (2*np.pi) / N

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(bin_edges[0:N], hist, width=width, bottom=bottom)

    if measure == 'radians':
        xT=plt.xticks()[0]
        xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
           r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
        plt.xticks(xT, xL)

    # Use custom colors and opacity
    # for r, bar in zip(radii, bars):
    #     bar.set_facecolor(plt.cm.jet(r / 10.))
    #     bar.set_alpha(0.8)

    plt.show()


# def hist3d(sample, bins=10, range=None, normed=False, weights=None):
#     r=np.random.randn(100,3,3)
#     r_ = r.reshape((r.shape[0], np.prod(r.shape[1:])))
#     H, edges = histogramnd(r_,tile(bins))
#     H.reshape((H.shape[0], r.shape[1:])


def histogramnd(a, bins=10, range=None, normed=False, weights=None, axis=None):
    """histogram(a, bins=10, range=None, normed=False, weights=None, axis=None)
                                                                   -> H, dict

    Return the distribution of sample.

    Parameters
    ----------
    a:       Array sample.
    bins:    Number of bins, or
             an array of bin edges, in which case the range is not used.
    range:   Lower and upper bin edges, default: [min, max].
    normed:  Boolean, if False, return the number of samples in each bin,
             if True, return a frequency distribution.
    weights: Sample weights. The weights are normed only if normed is True.
             Should weights.sum() not equal len(a), the total bin count will
             not be equal to the number of samples.
    axis:    Specifies the dimension along which the histogram is computed.
             Defaults to None, which aggregates the entire sample array.

    Output
    ------
    H:            The number of samples in each bin.
                  If normed is True, H is a frequency distribution.
    dict{
    'edges':      The bin edges, including the rightmost edge.
    'upper':      Upper outliers.
    'lower':      Lower outliers.
    'bincenters': Center of bins.
    }

    Examples
    --------
    x = random.rand(100,10)
    H, Dict = histogram(x, bins=10, range=[0,1], normed=True)
    H2, Dict = histogram(x, bins=10, range=[0,1], normed=True, axis=0)

    See also: histogramnd
    # License: Scipy compatible
    # Author: David Huard, 2006
    # http://projects.scipy.org/numpy/attachment/ticket/189/histogram1d.py
    """

    a = np.asarray(a)
    if axis is None:
        a = atleast_1d(a.ravel())
        axis = 0

    # Bin edges.
    if not np.iterable(bins):
        if range is None:
            range = (a.min(), a.max())
        mn, mx = [mi+0.0 for mi in range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        edges = np.linspace(mn, mx, bins+1, endpoint=True)
    else:
        edges = np.asarray(bins, float)

    dedges = np.diff(edges)
    decimal = int(-np.log10(dedges.min())+6)
    bincenters = edges[:-1] + dedges/2.

    # apply_along_axis accepts only one array input, but we need to pass the
    # weights along with the sample. The strategy here is to concatenate the
    # weights array along axis, so the passed array contains [sample, weights].
    # The array is then split back in  __hist1d.
    if weights is not None:
        aw = concatenate((a, weights), axis)
        weighted = True
    else:
        aw = a
        weighted = False

    count = np.apply_along_axis(__hist1d, axis, aw, edges, decimal, weighted, normed)

    # Outlier count
    upper = count.take(np.array([-1]), axis)
    lower = count.take(np.array([0]), axis)

    # Non-outlier count
    core = a.ndim*[slice(None)]
    core[axis] = slice(1, -1)
    hist = count[core]

    if normed:
        normalize = lambda x: atleast_1d(x/(x*dedges).sum())
        hist = np.apply_along_axis(normalize, axis, hist)

    return hist, {'edges':edges, 'lower':lower, 'upper':upper, \
        'bincenters':bincenters}


def __hist1d(aw, edges, decimal, weighted, normed):
    """Internal routine to compute the 1d histogram.
    aw: sample, [weights]
    edges: bin edges
    decimal: approximation to put values lying on the rightmost edge in the last
             bin.
    weighted: Means that the weights are appended to array a.
    Return the bin count or frequency if normed.

    # License: Scipy compatible
    # Author: David Huard, 2006
    # http://projects.scipy.org/numpy/attachment/ticket/189/histogram1d.py
    """
    nbin = edges.shape[0]+1
    if weighted:
        count = zeros(nbin, dtype=float)
        a,w = hsplit(aw,2)
        if normed:
            w = w/w.mean()
    else:
        a = aw
        count = np.zeros(nbin, dtype=int)
        w = None

    binindex = np.digitize(a, edges)

    # Values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    on_edge = np.where(np.around(a,decimal) == np.around(edges[-1], decimal))[0]
    binindex[on_edge] -= 1

    # Count the number of identical indices.
    flatcount = np.bincount(binindex, w)

    # Place the count in the histogram array.
    i = np.arange(len(flatcount))
    count[i] = flatcount

    return count

def cluster_test_main(gat, A, chance_level = np.pi/6,
                        alpha = 0.05, n_permutations = 2 ** 11,
                        threshold = dict(start=1., step=.2), lims=None,
                        ylabel='Performance', title=None):
    """This function takes as inputs one array X and computes cluster analysis
    and plots associated graphs.
    Input:
    gat: gat object is used only to retrieve useful information to plot, like
        time points and gat.plot functions.
        gat.scores_ is replaced by X
    X: ndimensional array representing gat or diagonal performance.
        If A is diagonal, its dimensions should be n_subjects * n_time
        If A is GAT, its dimensions should be n_subjects * n_time * n_time
    chance_level: chance level to test against.
        pi/6 for circular data and 0 for deviations are normally used
    """
    # check that X is array otherwise convert
    if not(type(A).__module__ == np.__name__):
        A = np.array(A)

    # define X
    X = A - chance_level

    # define time points
    times = gat.train_times['times_']

    # ------ Run stats
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
                                           X,
                                           out_type='mask',
                                           n_permutations=n_permutations,
                                           connectivity=None,
                                           threshold=threshold,
                                           n_jobs=-1)

    # ------ combine clusters and retrieve min p_values for each feature
    p_values = np.min(np.logical_not(clusters) +
                      [clusters[c] * p for c, p in enumerate(p_values)],
                      axis=0)
    x, y = np.meshgrid(gat.train_times['times_'],
                       gat.test_times_['times_'][0],
                       copy=False, indexing='xy')


    # PLOT
    # ------ Plot GAT
    gat.scores_ = np.mean(A, axis=0)
    if lims==None:
        lims = [np.min(gat.scores_),np.max(gat.scores_)]
    fig = gat.plot(vmin=lims[0], vmax=lims[1],
                   show=False,
                   extent=[np.min(times), np.max(times),
                       np.min(times), np.max(times)])
    ax = fig.axes[0]
    ax.contour(x, y, p_values < alpha, colors='black', levels=[0])
    #plt.title(title)
    plt.show()

    # ------ Plot Decoding
    scores_diag = np.transpose([A[:, t, t] for t in range(len(times))])
    fig, ax = plt.subplots(1)
    plot_eb(times, np.mean(scores_diag, axis=0),
            np.std(scores_diag, axis=0) / np.sqrt(scores_diag.shape[0]),
            color='blue', ax=ax)
    ymin, ymax = ax.get_ylim()
    sig_times = times[np.where(np.diag(p_values) < alpha)[0]]
    sfreq = (times[1] - times[0]) / 1000
    fill_betweenx_discontinuous(ax, ymin, ymax, sig_times, freq=sfreq,
                                color='gray', alpha=.3)
    ax.axhline(chance_level, color='k', linestyle='--', label="Chance level")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    #plt.title(title)
    plt.show()

# def cluster_test_interaction(gats, A, chance_level):
#     # XXX WIP
#     # check that X is array otherwise convert
#     if not(type(A).__module__ == np.__name__):
#         A = np.array(A)
#
#     # define X
#     X = A - chance_level
#
#     # define dimensions
#     n_subjects, n_conds, n_time, n_testtime  = dims = np.shape(A)
#
#     # define time stamps
#     times = gat.train_times['times_']
#
#     # apply method
#     if method == 'correlation':
#         R = np.zeros([n_subjects, n_time, n_time])
#         P = np.zeros([n_subjects, n_time, n_time])
#         for s,subject in enumerate(range(n_subjects)):
#             for T in range(n_time):
#                 for t in range(n_time):
#                     r, p = stats.spearmanr(A[s,:,T,t],np.arange(4))
#                     R[s,T,t] = r
#                     P[s,T,t] = p
#         imshow(np.mean(R,axis=0), interpolation='none',origin='lower',
#                                     extent=[np.min(times), np.max(times),
#                                         np.min(times), np.max(times)])
#         plt.colorbar()
#
#     elif method == 'regression':
#
#     elif method == 'subtraction':
