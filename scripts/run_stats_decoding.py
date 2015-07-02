import sys
sys.path.insert(0, './')
import pickle
from itertools import product
import numpy as np

from mne.stats import spatio_temporal_cluster_1samp_test

from scripts.config import (
    paths,
    subjects,
    data_types,
    analyses
)


for data_type, analysis in product(data_types, analyses):
    print analysis['name']
    # DATA
    scores = list()
    y_pred = list()
    for subject in subjects:
        print subject
        # define path to file to be loaded
        score_fname = paths('score', subject=subject, data_type=data_type,
                            analysis=analysis['name'])
        with open(score_fname, 'rb') as f:
            out = pickle.load(f)
            gat = out[0]
        scores.append(gat.scores_)
        y_pred.append(gat.y_pred_)

    # STATS
    def stat_fun(x, sigma=0, method='relative'):
        from mne.stats import ttest_1samp_no_p
        t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
        t_values[np.isnan(t_values)] = 0
        return t_values

    def stats(X):
        T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
            X,
            out_type='mask',
            stat_fun=stat_fun,
            n_permutations=2**11,
            threshold=dict(start=0., step=.1),
            n_jobs=3)
        return p_values.reshape(X.shape[1:])

    scores = [score for score in scores if not np.isnan(score[0][0])]
    if len(scores) < 7:
        print('%s: not enough subjects' % analysis['name'])
        continue
    chance = analysis['chance']
    alpha = 0.05
    times = gat.train_times_['times'] * 1000

    # STATS
    p_values = stats(np.array(scores) - chance)
    diag_offdiag = scores - np.tile([np.diag(score) for score in scores],
                                    [len(times), 1, 1]).transpose(1, 0, 2)
    p_values_off = stats(diag_offdiag)
    scores = out['scores']
    scores_diag = [np.diag(score) for score in scores]
    p_values_diag = stats(np.array(scores_diag)[:, :, None] - chance)

    # SAVE
    stats_fname = paths('score', subject='fsaverage', data_type=data_type,
                        analysis=('stats_' + analysis['name']), log=True)
    with open(stats_fname, 'wb') as f:
        out = dict(scores=scores, p_values=p_values, p_values_off=p_values_off,
                   times=times, analysis=analysis, p_values_diag=p_values_diag)
        pickle.dump(out, f)
