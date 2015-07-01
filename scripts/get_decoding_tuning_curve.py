import copy
import pickle
# import os.path as op
from itertools import product
import numpy as np

from gat.utils import subscore

from scripts.config import (
    paths,
    subjects,
    data_types,
    subscores,
)
from orientations.conditions import analysis as make_analysis
from base import tile_memory_free


analyses = [
    make_analysis('target_circAngle', 'circ_regress'),
    make_analysis('probe_circAngle', 'circ_regress')
]

# subscores = [('seen', 'detect_seen == True'),
#              ('unseen', 'detect_seen == False')]
# for pas in [1., 2., 3.]:
#     subscores.append(('pas%s' % pas, 'detect_button == %s' % pas))


def tuning(y_pred, y_train):
    error = (np.pi - np.squeeze(y_pred) +
             np.transpose(tile_memory_free(y_train, np.shape(y_pred)[:2]),
                          [1, 2, 0])) % (2 * np.pi) - np.pi
    bins = np.linspace(-np.pi, np.pi, 20)
    nT, nt, _, _ = np.shape(y_pred)
    h = np.zeros((nT, nt, len(bins) - 1))
    for T, t in product(range(nT), range(nt)):
        h[T, t, :], _ = np.histogram(error[T, t, :], bins)
        h[T, t, :] /= sum(h[T, t, :])
    return h


for data_type, analysis, subject in product(data_types, analyses, subjects):
    print analysis['name'], subject

    # load
    gat_fname = paths('decod', subject=subject, data_type=data_type,
                      analysis=analysis['name'])
    gat_fname = gat_fname
    with open(gat_fname, 'rb') as f:
        gat, _, sel, events = pickle.load(f)

    # optimize memory
    gat.estimators_ = list()

    # Save main score
    gat_ = copy.deepcopy(gat)
    gat_.y_pred_ = tuning(gat_.y_pred_, gat_.y_train_)
    score_fname = paths('score', subject=subject, data_type=data_type,
                        analysis=analysis['name'] + '-tuning')
    with open(score_fname, 'wb') as f:
        pickle.dump([gat_, analysis, sel, events], f)

    # Save subscores
    for subanalysis in subscores:
        score_fname = paths(
            'score', subject=subject, data_type=data_type,
            analysis=analysis['name'] + '-tuning-' + subanalysis[0])
        # if op.exists(score_fname):
        #     continue
        print analysis['name'], subject, subanalysis[0]
        subevents = events.iloc[sel].reset_index()
        query = subanalysis[1] if analysis['query'] is None\
            else '(' + analysis['query'] + ') and ' + subanalysis[1]
        subsel = range(len(subevents)) if query is None \
            else subevents.query(query).index
        subsel = [ii for ii in subsel
                  if ~np.isnan(np.array(subevents[analysis['condition']])[ii])]
        y = np.array(subevents[analysis['condition']], dtype=np.float32)
        # subscore
        gat_ = copy.deepcopy(gat)
        if len(np.unique(y[subsel])) > 1:
            gat_.scores_ = subscore(gat_, subsel, y[subsel])
        else:
            gat_.scores_ = np.nan * np.array(gat_.scores_)

        gat_.y_pred_ = tuning(np.array(gat_.y_pred_)[:, :, subsel, :],
                              gat_.y_train_[subsel])
        with open(score_fname, 'wb') as f:
            pickle.dump([gat_, analysis, sel, events, subsel], f)
