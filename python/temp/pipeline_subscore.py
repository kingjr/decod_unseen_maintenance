import op
import pickle
import numpy as np

import mne

from meeg_preprocessing.utils import setup_provenance
from toolbox.utils import find_in_df

from scripts.config import (
    results_dir, pyoutput_path,
    subjects,
    inputTypes,
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
    return pkl_fname


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

for typ in inputTypes:
    logger.info(ep['name'])
    for analysis in subscores:
        logger.info(subscore['name'])
        # Gather subjects
        scores_list = list()
        y_pred_list = list()
        for subject in subjects:
            # Load CV data
            file = pkl_fname(typ, subject, analysis['contrast'])
            with open(file) as f:
                gat, _, events, sel = pickle.load(f)

            # Subscore overall
            sel = find_in_df(events, analysis['include'], analysis['exclude'])
            key = analysis['include'].keys()[0]
            y = np.array(events[key][sel].tolist())
            score = subscore(gat, sel, y=y, scorer=analysis['scorer'])
            scores_list.append(score)

            # Keep mean prediction
            y_pred_list.append(mean_pred(gat, y))

        # TODO STATS
        # TODO PLOTS
