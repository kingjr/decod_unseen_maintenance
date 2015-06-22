import os.path as op
import pickle
import numpy as np

# import mne

# from meeg_preprocessing.utils import setup_provenance

from config import (
    subjects,
    data_types,
    pyoutput_path,
    angle2circle,
    absent,
    scorer_angle,
)


def sel_events(events, contrast):
    # Find excluded trials
    exclude = np.any([
        events[x['cond']] == ii for x in contrast['exclude']
        for ii in x['values']],
        axis=0)

    # Select condition
    include = list()
    cond_name = contrast['include']['cond']
    for value in contrast['include']['values']:
        # Find included trials
        include.append(events[cond_name] == value)
    sel = np.any(include, axis=0) * (exclude == False)
    sel = np.where(sel)[0]
    return sel


def pkl_fname(typ, subject, name):
    # define meg_path appendix
    if data_type == 'erf':
        fname_appendix = ''
    else:
        fname_appendix = '_Tfoi_mtm_' + data_type[4:] + 'Hz'

    # define path to file to be loaded
    pkl_fname = op.join(
        pyoutput_path, subject, 'mvpas',
        '{}-decod_{}{}.pickle'.format(subject, name, fname_appendix))
    return pkl_fname


def gat_subscore(gat, sel, y=None, scorer=None):
    """Subscores a GAT.

    Parameters
    ----------
        gat : GeneralizationAcrossTime object
        sel : list or array, shape (n_predictions)
        y : None | list or array, shape (n_selected_predictions,)
            If None, y set to gat.y_true_. Defaults to None.

    Returns
    -------
    scores
    """
    # TODO FIXME XXX
    import copy
    gat_ = copy.deepcopy(gat)
    # Subselection of trials
    gat_.y_pred_ = list()
    for train in range(len(gat.y_pred_)):
        y_pred_ = list()
        for test in range(len(gat.y_pred_[train])):
            y_pred_.append(gat.y_pred_[train][test][sel, :])
        gat_.y_pred_.append(y_pred_)
    gat_.y_train_ = gat.y_train_[sel]
    gat_.scorer = scorer
    return gat_.score(y=y)


def gat_order_y(gat_list, order=None, n_pred=None):
    """Combines multiple gat.y_pred_ & gat.y_train_ into a single gat.

    Parameters
    ----------
        gat_list : list of GeneralizationAcrossTime objects, shape (n_gat)
            The gats must have been predicted (gat.predict(epochs))
        order : None | list, shape (n_gat), optional
            Order of the prediction, to be recombined. Defaults to None.
        n_pred : None | int, optional
            Maximum number of predictions. If None, set to max(sel). Defaults
            to None.
    Returns
    -------
        gat : GeneralizationAcrossTime object"""
    import copy
    from mne.decoding.time_gen import GeneralizationAcrossTime as GAT
    if isinstance(gat_list, GAT):
        gat_list = [gat_list]
        order = [order]

    for gat in gat_list:
        if not isinstance(gat, GAT):
            raise ValueError('gat must be a GeneralizationAcrossTime object')

    if order is not None:
        if len(gat_list) != len(order):
            raise ValueError('len(order) must equal len(gat_list)')
    else:
        order = [range(len(gat.y_pred_[0][0])) for gat in gat_list]
        for idx in range(1, len(order)):
            order[idx] += len(order[idx-1])
    # Identifiy trial number
    if n_pred is None:
        n_pred = np.max([np.max(sel) for sel in order])
    n_dims = np.shape(gat_list[0].y_pred_[0][0])[1]
    # initialize combined gat
    cmb_gat = copy.deepcopy(gat_list[0])
    # initialize y_pred
    cmb_gat.y_pred_ = list()
    for train in range(len(gat.y_pred_)):
        y_pred_ = list()
        for test in range(len(gat.y_pred_[train])):
            y_pred_.append(np.nan * np.ones((n_pred, n_dims)))
        cmb_gat.y_pred_.append(y_pred_)
    # initialize y_train
    cmb_gat.y_train_ = np.ones((n_pred,))

    for gat, sel in zip(gat_list, order):
        for train in range(len(gat.y_pred_)):
            for test in range(len(gat.y_pred_[train])):
                cmb_gat.y_pred_[train][test][sel, :] = gat.y_pred_[train][test]
        cmb_gat.y_train_[sel] = gat.y_train_
    return cmb_gat


def mean_pred(gat, y=None):
    """Provides mean prediction for each category.

    Parameters
    ----------
        gat : GeneralizationAcrossTime object
        y : None | list or array, shape (n_predictions,)
            If None, y set to gat.y_train_. Defaults to None.

    Returns
    -------
    mean_y_pred : list of list of (float | array),
                  shape (train_time, test_time, classes, predict_shape)
        The mean prediction for each training and each testing time point for
        each class.
    """
    if y is None:
        y = gat.y_train_
    y_pred = list()
    for train in range(len(gat.y_pred_)):
        y_pred_ = list()
        for test in range(len(gat.y_pred_[train])):
            y_pred__ = list()
            for c in np.unique(y):
                m = np.mean(gat.y_pred_[train][test][y == c, :], axis=0)
                y_pred__.append(m)
            y_pred_.append(y_pred__)
        y_pred.append(y_pred_)
    return y_pred

# Setup logs:

all_vis = [dict(cond='response_visibilityCode', values=[ii])
           for ii in range(1, 5)]


def default_subscore(sel):
    selected_vis = [vis for ii, vis in enumerate(all_vis) if ii != sel]
    subscore = dict(name='targetAngle' + str(vis),
                    contrast='targetAngle',
                    include=dict(cond='orientation_target_rad',
                    values=angle2circle([15, 45, 75, 105, 135, 165])),
                    exclude=[absent] + selected_vis, chance=1. / 6.,
                    scorer=scorer_angle)
    return subscore

subscores = [default_subscore(ii) for ii in range(0, 4)]

typ = data_types[-1]
scores_list = list()
for subject in subjects:
    print(subject)
    scores = list()
    # Load CV data
    file = pkl_fname(typ, subject, 'targetAngle')
    with open(file) as f:
        gat, _, sel, events = pickle.load(f)

    # Put back trials predictions in order according to original
    # selection
    gat = gat_order_y(gat, order=sel, n_pred=len(events))

    for subscore in subscores:
        # Subscore overall from new selection
        sel = sel_events(events, subscore)
        key = subscore['include']['cond']
        y = np.array(events[key][sel].tolist())
        # HACK to skip few subjects that do not have enough trials in some
        # conditions of interest
        try:
            score = gat_subscore(gat, sel, y=y, scorer=subscore['scorer'])
        except:
            score = list()
        scores.append(score)
    scores_list.append(scores)

gat.y_pred_ = []
gat.estimators_ = []
fname = '/home/niccolo/Dropbox/DOCUP/scripts/python/TargetAngle_subscoreVis.pickle'
with open(fname, 'w') as f:
    pickle.dump([scores_list, gat], f)
