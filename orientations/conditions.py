# Decoding parameters
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from utils import clf_2class_proba, SVR_angle
from base import (scorer_angle, scorer_auc, scorer_spearman, scorer_circLinear)


def analysis(name, typ, condition=None, query=None):
    single_trial = False
    erf_function = None
    if typ == 'categorize':
        clf = Pipeline([('scaler', StandardScaler()),
                        ('svc', clf_2class_proba(C=1, class_weight='auto'))])
        scorer = scorer_auc
        chance = .5
    elif typ == 'regress':
        clf = Pipeline([('scaler', StandardScaler()), ('svr', LinearSVR(C=1))])
        scorer = scorer_spearman
        single_trial = True  # with non param need single trial here
        chance = 0.
    elif typ == 'circ_regress':
        clf = SVR_angle()
        scorer = scorer_angle
        chance = 1. / 6.
        single_trial = True
        erf_function = scorer_circLinear
    if condition is None:
        condition = name
    return dict(name=name, condition=condition, query=query, clf=clf,
                scorer=scorer, chance=chance, erf_function=erf_function,
                single_trial=single_trial, cv=8)

analyses = (
    analysis('target_present',      'categorize'),
    analysis('target_contrast',     'regress'),
    analysis('target_contrast_pst', 'regress', condition='target_contrast',
             query='target_present == True'),
    analysis('target_spatialFreq',  'categorize'),
    analysis('target_circAngle',    'circ_regress'),
    analysis('probe_circAngle',     'circ_regress'),
    analysis('probe_tilt',          'categorize'),
    analysis('discrim_button',      'categorize'),
    analysis('discrim_correct',     'categorize'),
    analysis('detect_button',       'regress'),
    analysis('detect_button_pst',   'regress', condition='detect_button',
             query='target_present == True'),
    analysis('detect_seen',         'categorize'),
    analysis('detect_seen_pst',     'categorize', condition='detect_seen',
             query='target_present == True')
)

# ###################### Define subscores #####################################

subscores = []
for analysis in analyses:
    analysis['train_analysis'] = analysis['name']
    subscores.append(analysis)
    query = '(%s) and ' % analysis['query'] if analysis['query'] else ''
    # Subdivide by visibility
    if analysis['name'] not in ['detect_button', 'detect_button_pst',
                                'detect_seen', 'detect_seen_pst']:
        # Unseen
        analysis_ = copy.deepcopy(analysis)
        analysis_['name'] += '-unseen'
        analysis_['query'] = query + 'detect_seen == False'
        subscores.append(analysis_)
        # Seen
        analysis_ = copy.deepcopy(analysis)
        analysis_['name'] += '-seen'
        analysis_['query'] = query + 'detect_seen == True'
        subscores.append(analysis_)
        # Seen1
        analysis_ = copy.deepcopy(analysis)
        analysis_['name'] += '-pas1'
        analysis_['query'] = query + 'detect_button == 1.'
        subscores.append(analysis_)
        # Seen2
        analysis_ = copy.deepcopy(analysis)
        analysis_['name'] += '-pas2'
        analysis_['query'] = query + 'detect_button == 2.'
        subscores.append(analysis_)
        # Seen3
        analysis_ = copy.deepcopy(analysis)
        analysis_['name'] += '-pas3'
        analysis_['query'] = query + 'detect_button == 3.'
        subscores.append(analysis_)
    # Subdivide by accuracy
    if analysis['name'] not in ['discrim_correct']:
        # Correct
        analysis_ = copy.deepcopy(analysis)
        analysis_['name'] += '-correct'
        analysis_['query'] = query + 'discrim_correct == True'
        subscores.append(analysis_)
        # Incorrect
        analysis_ = copy.deepcopy(analysis)
        analysis_['name'] += '-incorrect'
        analysis_['query'] = query + 'discrim_correct == False'
        subscores.append(analysis_)

# ############# Define second-order subscores #################################
subscores2 = []

for analysis in analyses:
    # XXX
    pass
