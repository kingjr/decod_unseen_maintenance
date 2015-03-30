""" In this page I store all functions used to post-process classifiers
results.
Niccolo Pescetelli niccolo.pescetelli@psy.ox.ac.uk
"""



def recombine_svr_prediction(path_x,path_y):
    """
    This function takes the paths of two classifiers SVR predictions, typically
    sine and cosine of an angle, and combine them into a predicted angle in
    radians
    """
    import pickle
    import numpy as np

    #load first regressor (cosine)
    with open(path_x) as f:
        gat, contrast = pickle.load(f)
    x = np.array(gat.y_pred_)
    true_x = gat.y_train_

    #load second regressor (sine)
    with open(path_y) as f:
        gat, contrast = pickle.load(f)
    y = np.array(gat.y_pred_)

    # cartesian 2 polar (radians) transformation
    angle, radius = cart2pol(x, y)

    # take only values within 2 pi radians
    angle = (angle % (2 * np.pi)) - np.pi
    return (angle, radius, true_x)


def cart2pol(x, y):
    import numpy as np
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return(theta,radius)

def realign_angle(gat, dims, angles = [15, 45, 75, 105, 135, 165] ):
    """
    This function realign classes output by a classifier so to give a
    distance in terms of categories of the predicted class and the real class.
    Input:
    gat : the gat object output by a classifier
    optional values: angles
    Output:
    probas : a time x time x trials x class array
    """
    import numpy as np
    # realign to 4th angle category
    probas = np.zeros(dims)
    n_classes = len(angles)
    for a, angle in enumerate(angles):
        sel = gat.y_train_ == angle
        prediction = np.array(gat.y_pred_)
        order = np.array(range(a,n_classes)+range(0,(a)))
        probas[:, :, sel, :] = prediction[:, :, sel, :][:, :, :, order]
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
    import numpy as np
    import matplotlib.pyplot as plt


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
