"""Performs stats across subjects of decoding scores fitted within subjects"""
import numpy as np
from base import stats
from config import subjects, load, save
from conditions import analyses

# For each analysis of interest
for analysis in analyses:
    # Load decoding results
    print('load', analysis['name'])
    scores = list()
    for subject in subjects:
        # define path to file to be loaded
        score, times = load('score', subject=subject,
                            analysis=analysis['name'])
        scores.append(score)

    scores = [sc for sc in scores if not np.isnan(sc[0][0])]
    if len(scores) < 7:
        print('%s: not enough subjects' % analysis['name'])
        continue
    chance = analysis['chance']
    alpha = 0.05

    # Compute stats: is decoding different from theoretical chance level (using
    # permutations across subjects)
    print('stats', analysis['name'])
    p_values = stats(np.array(scores) - chance)
    diag_offdiag = scores - np.tile([np.diag(sc) for sc in scores],
                                    [len(times), 1, 1]).transpose(1, 0, 2)
    p_values_off = stats(diag_offdiag)

    scores_diag = [np.diag(sc) for sc in scores]
    p_values_diag = stats(np.array(scores_diag)[:, :, None] - chance)

    # Save stats results
    print('save', analysis['name'])
    out = dict(scores=scores, p_values=p_values, p_values_off=p_values_off,
               times=times, analysis=analysis, p_values_diag=p_values_diag)
    save(out, 'score',  analysis=('stats_' + analysis['name']))
