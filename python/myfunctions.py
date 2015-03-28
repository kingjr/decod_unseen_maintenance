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
    radius, angle = cart2pol(x, y)
    angle = (angle % (2 * np.pi)) - np.pi
    return (angle, radius, true_x)


def cart2pol(x, y):
    import numpy as np
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

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
