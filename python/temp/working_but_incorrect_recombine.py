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
