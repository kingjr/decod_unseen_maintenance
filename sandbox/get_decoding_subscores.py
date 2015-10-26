import numpy as np
import pickle
from itertools import product

from scripts.config import (
    paths,
    subjects,
    data_types,
    analyses_order2 as analyses,
)

for data_type, analysis, subject in product(data_types, analyses, subjects):
    print analysis['name'], subject
    scores = list()
    y_preds = list()

    # Load data
    for subanalysis in analysis['condition']:
        score_fname = paths('score', subject=subject, data_type=data_type,
                            analysis=subanalysis['name'])
        with open(score_fname, 'rb') as f:
                gat, _, _, _, _ = pickle.load(f)
        scores.append(gat.scores_)
        y_preds.append(gat.y_pred_)

    # Apply first order analysis
    if len(scores) > 2:
        raise RuntimeError('only binary contrasts for now')
    gat.scores_ = np.array(scores[0]) - np.array(scores[1])
    gat.y_pred_ = np.array(y_preds[0]) - np.array(y_preds[1])

    # Save
    score_fname = paths('score', subject=subject, data_type=data_type,
                        analysis=analysis['name'])
    with open(score_fname, 'wb') as f:
        pickle.dump([gat, analysis], f)
