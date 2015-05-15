import op
import pickle
import numpy as np

import mne

from meeg_preprocessing.utils import setup_provenance
from toolbox.utils import find_in_df

from scripts.config import (
    results_dir, pyoutput_path,
    subjects,
    epochs_params,
    open_browser,
    subscores
)


def pkl_fname(type, subject, contrast):
    # define meg_path appendix
    if typ['name'] == 'erf':
        fname_appendix = ''
    else:
        fname_appendix = op.join('_Tfoi_mtm_',
                                 typ['name'][4:], 'Hz')

    # define path to file to be loaded
    pkl_fname = op.join(
        pyoutput_path, subject, 'mvpas',
        '{}-decod_{}{}.pickle'.format(
            subject, contrast['name'], fname_appendix))

def combine_gat_y(gat_list, order_list=None):
    """Combines multiple gat.y_pred_ & gat.y_train_ into a single gat.

    Parameters
    ----------
        gat_list : list of GeneralizationAcrossTime objects, shape (n_gat)
            The gats must have been predicted (gat.predict(epochs))
        order_list : None | list, shape (n_gat), optional
            Order of the prediction, to be recombined. Defaults to None.
    Returns
    -------
        gat : GeneralizationAcrossTime object"""

    if order_list is not None:
        if len(gat_list) != len(order_list):
            raise ValueError('len(order_list) must equal len(gat_list)')
    else:
        order = [range(len(gat.y_pred_[0][0])) for gat in gat_list]
        for idx in range(1, len(order)):
            order[idx] += len(order[idx-1])
    # Identifiy trial number
    n_trials = np.max([np.max(sel) for sel in order_list])
    n_dims = np.shape(gat_list[0].y_pred_[0][0])[1]
    # initialize combined gat
    cmb_gat = gat_list[0]
    # initialize y_pred
    for train in range(gat.y_pred_):
        for test in range(gat.y_pred_[train]):
            cmb_gat.y_pred_[train][test] = np.nan * np.ones((n_trials, n_dims))
    # initialize y_train
    cmb_gat.y_train_ = np.ones((n_trials,))

    for gat, sel in zip(gat_list, order_list):
        for train in range(gat.y_pred_):
            for test in range(gat.y_pred_[train]):
                cmb_gat.y_pred_[train][test][sel, :] = gat.y_pred_[train][test]
        cmb_gat.y_train_[sel] = gat.y_train_
    return cmb_gat


def subscore(gat, sel, y=None, scorer=None):
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
    # Subselection of trials
    for train in range(gat.y_pred_):
        for test in range(gat.y_pred_[train]):
            gat.y_pred_ = gat.y_pred_[train][test][sel, :]
    gat.y_train_ = gat.y_train_[sel]
    return gat.score(y=y, scorer=scorer)


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
    for train in range(gat.y_pred_):
        y_pred_ = list()
        for test in range(gat.y_pred_[train]):
            y_pred__ = list()
            for c in np.unique(y):
                m = np.mean(gat.y_pred_[train][test][y == c, :], axis=0)
                y_pred__.append(m)
            y_pred_.append(y_pred__)
        y_pred.append(y_pred_)
    return y_pred

# Setup logs:
mne.set_log_level('INFO')
report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir=results_dir)

for ep in epochs_params:
    ep = epochs_params[0]
    logger.info(ep['name'])
    for analysis in subscores:
        logger.info(subscore['name'])
        # Gather subjects
        scores_list = list()
        y_pred_list = list()
        for subject in subjects:
            # Load CV data
            if analysis['contrast'] is not None:
                file = pkl_fname(typ, subject, analysis['contrast'])
                with open(file) as f:
                    gat_dec, _, events, sel_dec = pickle.load(f)
            # Load Generalize data
            if analysis['generalization'] is not None:
                file = pkl_fname(typ, subject, analysis['contrast'])
                with open(file) as f:
                    gat_gen, _, events, sel_gen = pickle.load(f)

            # Combine multiple gat predictions
            gat = combine_gat_y([gat_dec, gat_gen], [sel_dec, sel_gen])

            # Subscore overall
            sel = find_in_df(events, analysis['include'], analysis['exclude'])
            key = analysis['include'].keys()[0]
            y = np.array(events[key][sel].tolist())
            scores_list.append(subscore(gat, sel, y=y,
                                        scorer=analysis['scorer']))

            # Keep mean prediction
            y_pred_list.append(mean_pred(gat, y))

        # TODO STATS
        # TODO PLOTS
