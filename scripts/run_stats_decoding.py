import sys
sys.path.insert(0, './')
# import matplotlib
# matplotlib.use('Agg')

import pickle
from itertools import product
import os.path as op
import numpy as np

from mne.stats import spatio_temporal_cluster_1samp_test
# from meeg_preprocessing.utils import setup_provenance
from gat.utils import mean_ypred, subscore

from scripts.config import (
    paths,
    subjects,
    # open_browser,
    data_types,
    subscores as analyses
)
# from scripts.transfer_data import upload_report


# report, run_id, _, logger = setup_provenance(
#     script=__file__, results_dir=paths('report'))

# Apply contrast to ERFs or frequency power
for data_type, analysis in product(data_types, analyses):
    # DATA
    scores = list()
    y_pred = list()
    for subject in subjects:
        # define path to file to be loaded
        score_fname = paths('score', subject=subject, data_type=data_type,
                            analysis=analysis['name'])
        if not op.exists(score_fname):
            # load
            gat_fname = paths('decod', subject=subject, data_type=data_type,
                              analysis=analysis['train_analysis'])
            # FIXME
            gat_fname = '/media/jrking/My Passport/Niccolo/' + gat_fname
            with open(gat_fname, 'rb') as f:
                gat, _, sel, events = pickle.load(f)

            # subsel
            query, condition = analysis['query'], analysis['condition']
            sel = range(len(events)) if query is None \
                else events.query(query).index
            sel = [ii for ii in sel if ~np.isnan(events[condition][sel][ii])]
            y = np.array(events[condition], dtype=np.float32)

            # subscore
            gat.scores = subscore(gat, sel, y[sel])

            # optimize memory
            gat.estimators_ = list()
            gat.y_pred_ = mean_ypred(gat)

            # save
            with open(score_fname, 'wb') as f:
                pickle.dump([gat, analysis, sel, events], f)
        else:
            with open(score_fname, 'rb') as f:
                gat, _, sel, events = pickle.load(f)

        scores.append(gat.scores_)
        y_pred.append(gat.y_pred_)

    # STATS
    # ------ Parameters XXX to be transfered to config?
    # XXX JRK set stats level for each type of analysis
    alpha = 0.05

    X = np.transpose(scores, [2, 0, 1]) - analysis['chance']

    # ------ Run stats
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X,
        out_type='mask',
        n_permutations=2 ** 11,
        connectivity=None,
        threshold=dict(start=.2, step=.2),
        n_jobs=-1)

    # ------ combine clusters & retrieve min p_values for each feature
    cluster_p = [clusters[c] * p for c, p in enumerate(p_values)]
    p_values = np.min(np.logical_not(clusters) + cluster_p, axis=0)

    # SAVE
    stats_fname = paths('score', subject='fsaverage', data_type=data_type,
                        analysis=('stats_' + analysis['name']),
                        log=True)
    with open(stats_fname, 'wb') as f:
        pickle.dump([scores, p_values], f)

# report.save(open_browser=open_browser)
# upload_report(report)
