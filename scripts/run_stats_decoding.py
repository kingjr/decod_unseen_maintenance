"""Performs stats across subjects of decoding scores fitted within subjects"""
import pickle
import numpy as np
from scripts.base import stats
from scripts.config import (paths, subjects, analyses)

# For each analysis of interest
for analysis in analyses:
    print analysis['name']
    # Load decoding results
    scores = list()
    y_pred = list()
    for subject in subjects:
        print subject
        # define path to file to be loaded
        score_fname = paths('score', subject=subject,
                            analysis=analysis['name'])
        with open(score_fname, 'rb') as f:
            out = pickle.load(f)
            gat = out[0]
        scores.append(gat.scores_)
        y_pred.append(gat.y_pred_)

    scores = [score for score in scores if not np.isnan(score[0][0])]
    if len(scores) < 7:
        print('%s: not enough subjects' % analysis['name'])
        continue
    chance = analysis['chance']
    alpha = 0.05
    times = gat.train_times_['times'] * 1000

    # Compute stats: is decoding different from theoretical chance level (using
    # permutations across subjects)
    p_values = stats(np.array(scores) - chance)
    diag_offdiag = scores - np.tile([np.diag(score) for score in scores],
                                    [len(times), 1, 1]).transpose(1, 0, 2)
    p_values_off = stats(diag_offdiag)

    scores_diag = [np.diag(score) for score in scores]
    p_values_diag = stats(np.array(scores_diag)[:, :, None] - chance)

    # Save stats results
    stats_fname = paths('score',  analysis=('stats_' + analysis['name']))
    with open(stats_fname, 'wb') as f:
        out = dict(scores=scores, p_values=p_values, p_values_off=p_values_off,
                   times=times, analysis=analysis, p_values_diag=p_values_diag)
        pickle.dump(out, f)
