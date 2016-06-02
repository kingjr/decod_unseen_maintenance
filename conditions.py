"""Define each analysis and its corresponding classifier"""
# Decoding parameters
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from jr.gat import (scorer_angle, force_predict,
                    scorer_spearman, AngularRegression)
from base import scorer_circlin


def scorer_auc(y_true, y_pred):
    """Aux. function to return AUC score from a probabilistic prediction"""
    # FIXME jr.gat.scorers.scorer_auc crashes when too many values with the
    # same proba.
    return roc_auc_score(y_true == np.max(y_true), y_pred[:, 0])


def analysis(name, typ, condition=None, query=None, title=None):
    """Wrapper to ensure that we attribute the same function for each type
    of analyses: e.g. categorical, regression, circular regression."""
    erf_function = None
    if typ == 'categorize':
        clf = make_pipeline(
            StandardScaler(),
            force_predict(LogisticRegression(class_weight='balanced'), axis=1))
        scorer = scorer_auc
        chance = .5
    elif typ == 'regress':
        clf = make_pipeline(StandardScaler(), LinearSVR(C=1))
        scorer = scorer_spearman
        chance = 0.
    elif typ == 'circ_regress':
        clf = make_pipeline(StandardScaler(),
                            AngularRegression(LinearSVR(C=1)))
        scorer = scorer_angle
        chance = 0.
        erf_function = scorer_circlin
    if condition is None:
        condition = name
    return dict(name=name, condition=condition, query=query, clf=clf,
                scorer=scorer, chance=chance, erf_function=erf_function,
                cv=8, typ=typ, title=title)


# For each analysis we need to specifically analyze a subset of trials to avoid
# finding trivial effects (e.g. visibility is correlated with target presence)
analyses = (
    analysis('target_present',      'categorize', title='Target Presence'),
    analysis('target_contrast_pst', 'regress', condition='target_contrast',
             query='target_present == True', title='Target Contrast'),
    analysis('target_spatialFreq',  'categorize',
             title='Target Spatial Frequency'),
    analysis('target_phase',        'circ_regress', title='Target Phase'),
    analysis('target_circAngle',    'circ_regress', title='Target Angle'),
    analysis('probe_circAngle',     'circ_regress', title='Probe Angle'),
    analysis('probe_tilt',          'categorize', title='Target - Probe Tilt'),
    analysis('probe_phase',    'circ_regress', title='Probe Phase'),
    analysis('discrim_button', 'categorize', title='Tilt Decision'),
    analysis('detect_button_pst',   'regress', condition='detect_button',
             query='target_present == True', title='Visibility Decision'),
)

# Define a specific color for each analysis
cmap = plt.get_cmap('gist_rainbow')
for ii in range(len(analyses)):
    analyses[ii]['color'] = np.array(cmap(float(ii) / len(analyses)))


# To control for correlation across variables, we'll score the classifiers on a
# subset of trials.
subscores = [('seen', 'detect_seen == True'),
             ('unseen', 'detect_seen == False')]
for pas in [1., 2., 3.]:
    subscores.append(('pas%s' % pas,
                      'detect_button == %s' % pas))
for contrast in [0., .5, .75, 1.]:
    subscores.append(('contrast%s' % contrast,
                      'target_contrast == %s' % contrast)),
for pas, contrast in product([0., 1., 2., 3.], [.5, .75, 1.]):
    subscores.append(('pas%s-contrast%s' % (pas, contrast),
                      'detect_button == %s and target_contrast == %s' % (
                      pas, contrast)))
