""" In this page I store all functions used to post-process classifiers
results.
Niccolo Pescetelli niccolo.pescetelli@psy.ox.ac.uk
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


def recombine_svr_prediction(gatx,gaty, res=40):
    """
    This function takes the paths of two classifiers SVR predictions, typically
    sine and cosine of an angle, and combine them into a predicted angle in
    radians
    """

    # The way it is done in Matlab XXX to be removed
    #
    # % realign to get single tuning curve across angles
    # predict_error = [];
    # for a = 6:-1:1
    #     % select trials with angles a
    #     sel = angles==a;
    #     % relign across angle categories
    #     predict_error(sel,:,:) = mod((pi+mod(theta(sel,:,:),2*pi)-2*deg2rad(-15+30*a))/2,pi);
    # end
    # sel = isnan(angles);
    # predict_error(sel,:,:) = NaN;
    #
    #
    # % compute proportion of trials correctly predicted
    # trial_proportion = hist(predict_error-pi/2,borns(-pi/2,pi/2,res));
    # trial_proportion = reshape(trial_proportion,[size(trial_proportion,1) size(predict_error,2) size(predict_error,3)]);
    # trial_proportion(1,:,:) = trial_proportion(1,:,:)+trial_proportion(end,:,:);
    # trial_proportion(end,:,:) = trial_proportion(1,:,:);
    # % get proportion of trials
    # trial_proportion = trial_proportion./repmat(sum(trial_proportion),[size(trial_proportion,1),1,1]);

    # define angles in degrees
    angles = np.deg2rad(np.linspace(15, 165, 6))

    #load first regressor (cosine)
    x = np.array(gatx.y_pred_)

    #load second regressor (sine)
    y = np.array(gaty.y_pred_)

    # cartesian 2 polar (radians) transformation
    theta, _ = cart2pol(x, y)

    # take only values within 0 to 2 pi radians range
    predict_angle = (np.squeeze(theta) % (2 * np.pi))

    # true angle in degrees
    # XXX CAREFUL, HERE IT WORKS BECAUSE ORIENTATIONS GO FROM 0 to 180
    # retrieve true predictor (eg. true cosine)
    trueX = gatx.y_train_# try to remove from return?
    true_angle = np.arccos(trueX)

    # realign to get single tuning curve across angles
    predict_error = np.zeros(predict_angle.shape)
    r = lambda i: np.round(100000 * i) / 100000  # to avoid float error
    for a in angles:
        # select trials with angles a
        # XXX this is soon to be replaced by the new classification.
        sel = r(true_angle) == r(a)
        # relign across angle categories
        predict_error[:,:,sel] = ((predict_angle[:, :, sel] - 2 * a) / 2) % np.pi

    # define bin_edges
    bin_edges = lambda m, M, n: np.arange(m+(M-m)/n/2,(M+(M-m)/n/2),(M-m)/n)
    # compute proportion of trials correctly predicted
    N = histogramnd(predict_error.squeeze() - np.pi/2,
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

    return (theta, trueX, trial_prop)

def cart2pol(x, y):
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return(theta, radius)

def realign_angle(gat, angles = [15, 45, 75, 105, 135, 165] ):
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


# License: Scipy compatible
# Author: David Huard, 2006
# http://projects.scipy.org/numpy/attachment/ticket/189/histogram1d.py
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


# License: Scipy compatible
# Author: David Huard, 2006
# http://projects.scipy.org/numpy/attachment/ticket/189/histogram1d.py
def __hist1d(aw, edges, decimal, weighted, normed):
    """Internal routine to compute the 1d histogram.
    aw: sample, [weights]
    edges: bin edges
    decimal: approximation to put values lying on the rightmost edge in the last
             bin.
    weighted: Means that the weights are appended to array a.
    Return the bin count or frequency if normed.
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