import copy
import pickle
from itertools import product
import numpy as np
from jr.gat import mean_ypred, subscore

from scripts.config import (
    paths,
    subjects,
    data_types,
    analyses,
    subscores,
)

# XXX This script needs to be simplified. There's no need to compute the
# subscore here, as you compute it separately in run_decoding_*.py

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

    # FIXME needs to be removed after next AWS round---------------------------
    if 'Angle' in analysis['name']:
        gat.scores_ = np.pi / 2 - np.array(gat.scores_)
    # FIXME--------------------------------------------------------------------

    # Save main score
    gat_ = copy.deepcopy(gat)
    gat_.y_pred_ = mean_ypred(gat_)
    score_fname = paths('score', subject=subject, data_type=data_type,
                        analysis=analysis['name'])
    with open(score_fname, 'wb') as f:
        pickle.dump([gat_, analysis, sel, events], f)

    # Save subscores
    for subanalysis in subscores:
        # check if exists
        score_fname = paths('score', subject=subject, data_type=data_type,
                            analysis=analysis['name'] + '-' + subanalysis[0])
        # if op.exists(score_fname):
        #     continue
        print analysis['name'], subject, subanalysis[0]
        # subselect
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
        gat_.y_pred_ = mean_ypred(gat_, sel=subsel)
        # save
        with open(score_fname, 'wb') as f:
            pickle.dump([gat_, analysis, sel, events, subsel], f)
