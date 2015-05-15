import os.path as op
import pickle
import numpy as np

# import mne
from mne.stats import spatio_temporal_cluster_1samp_test

# from meeg_preprocessing.utils import setup_provenance
from toolbox.utils import fill_betweenx_discontinuous, plot_eb

from config import (
    # results_dir,
    pyoutput_path,
    subjects,
    inputTypes,
    # open_browser,
    subscores
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
    if typ['name'] == 'erf':
        fname_appendix = ''
    else:
        fname_appendix = op.join('_Tfoi_mtm_',
                                 typ['name'][4:], 'Hz')

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
    # Subselection of trials
    for train in range(gat.y_pred_):
        for test in range(gat.y_pred_[train]):
            gat.y_pred_ = gat.y_pred_[train][test][sel, :]
    gat.y_train_ = gat.y_train_[sel]
    return gat.score(y=y, scorer=scorer)


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
# mne.set_log_level('INFO')
# report, run_id, results_dir, logger = setup_provenance(
#     script=__file__, results_dir=results_dir)

for typ in inputTypes:
    # logger.info(typ['name'])
    for subscore in subscores:
        # logger.info(subscore['name'])
        # Gather subjects data
        scores_list = list()
        y_pred_list = list()
        for subject in subjects:
            # Load CV data
            file = pkl_fname(typ, subject, subscore['contrast'])
            with open(file) as f:
                gat, _, sel, events = pickle.load(f)

            # Put back trials predictions in order according to original
            # selection
            gat = gat_order_y(gat, order=sel, n_pred=len(events))

            # Subscore overall from new selection
            sel = sel_events(events, subscore)
            key = subscore['include']['cond'].keys()[0]
            y = np.array(events[key][sel].tolist())
            score = gat_subscore(gat, sel, y=y, scorer=subscore['scorer'])
            scores_list.append(score)

            # Keep mean prediction
            y_pred_list.append(mean_pred(gat, y))

        # STATS
        scores = np.array(scores_list)
        # ------ Parameters XXX to be transfered to config?
        # XXX JRK set stats level for each type of subscore
        alpha = 0.05
        n_permutations = 2 ** 11
        threshold = dict(start=.2, step=.2)

        X = scores.transpose((2, 0, 1)) - subscore['chance']

        # ------ Run stats
        T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
            X,
            out_type='mask',
            n_permutations=n_permutations,
            connectivity=None,
            threshold=threshold,
            n_jobs=-1)

        # ------ combine clusters & retrieve min p_values for each feature
        cluster_p = [clusters[c] * p for c, p in enumerate(p_values)]
        p_values = np.min(np.logical_not(clusters) + cluster_p, axis=0)
        x, y = np.meshgrid(gat.train_times['times_'],
                           gat.test_times_['times_'][0],
                           copy=False, indexing='xy')

        # PLOT
        # ------ Plot GAT
        gat.scores_ = np.mean(scores, axis=2)
        fig = gat.plot(vmin=np.min(gat.scores_), vmax=np.max(gat.scores_),
                       show=False)
        ax = fig.axes[0]
        ax.contour(x, y, p_values < alpha, colors='black', levels=[0])
        # plt.show()
        # report.add_figs_to_section(
        #     fig, '%s (%s) : Decoding GAT' %
        #     (typ['name'], subscore['name']), typ['name'])

        # ------ Plot Decoding
        fig = gat.plot_diagonal(show=False)
        ax = fig.axes[0]
        ymin, ymax = ax.get_ylim()

        scores_diag = np.array([np.diag(s) for s in
                                scores.transpose((2, 0, 1))])
        times = gat.train_times['times_']

        sig_times = times[np.where(np.diag(p_values) < alpha)[0]]
        sfreq = (times[1] - times[0]) / 1000
        fill_betweenx_discontinuous(ax, ymin, ymax, sig_times, freq=sfreq,
                                    color='orange')

        plot_eb(times, np.mean(scores_diag, axis=0),
                np.std(scores_diag, axis=0) / np.sqrt(scores.shape[2]),
                ax=ax, color='blue')
        # plt.show()
        # report.add_figs_to_section(
        #     fig, '%s (%s): Decoding diag' %
        #     (typ['name'], subscore['name']), typ['name'])

        # SAVE
        fname = pkl_fname(typ, subject, subscore['name'])
        with open(pkl_fname, 'wb') as f:
            pickle.dump([scores, p_values], f)

# report.save(open_browser=open_browser)
