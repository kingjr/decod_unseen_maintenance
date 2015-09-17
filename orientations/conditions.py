# Decoding parameters
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from utils import clf_2class_proba, SVR_angle
from base import scorer_angle, scorer_circLinear
from jr.gat import scorer_auc, scorer_spearman


def analysis(name, typ, condition=None, query=None, title=None):
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
        chance = 0.
        single_trial = True
        erf_function = scorer_circLinear
    if condition is None:
        condition = name
    return dict(name=name, condition=condition, query=query, clf=clf,
                scorer=scorer, chance=chance, erf_function=erf_function,
                single_trial=single_trial, cv=8, typ=typ, title=title)

analyses = (
    analysis('target_present',      'categorize', title='Target Present'),
    # analysis('target_contrast',     'regress'),
    analysis('target_contrast_pst', 'regress', condition='target_contrast',
             query='target_present == True', title='Target Contrast'),
    analysis('target_spatialFreq',  'categorize',
             title='Target Spatial Frequency'),
    analysis('target_circAngle',    'circ_regress', title='Target Angle'),
    analysis('probe_circAngle',     'circ_regress', title='Probe Angle'),
    analysis('probe_tilt',          'categorize', title='Target - Probe Tilt'),
    # analysis('discrim_button',      'categorize'),
    # analysis('discrim_correct',     'categorize'),
    # analysis('detect_button',       'regress'),
    analysis('detect_button_pst',   'regress', condition='detect_button',
             query='target_present == True', title='Visibility Response'),
    # analysis('detect_seen',         'categorize'),
    analysis('detect_seen_pst',     'categorize', condition='detect_seen',
             query='target_present == True',
             title='Tilt Discrimination Response')
)

# ###################### Define subscores #####################################

subscores = [
    ('seen', 'detect_seen == True'),
    ('unseen', 'detect_seen == False'),
    ('correct', 'discrim_correct == True'),
    ('incorrect', 'discrim_correct == False')
]
for pas in [1., 2., 3.]:
    subscores.append(('pas%s' % pas,
                      'detect_button == %s' % pas))
for contrast in [0., .5, .75, 1.]:
    subscores.append(('contrast%s' % contrast,
                      'target_contrast == %s' % contrast)),
for pas, contrast in product([1., 2., 3.], [0., .5, .75, 1.]):
    subscores.append(('pas%s-contrast%s' % (pas, contrast),
                      'detect_button == %s and target_contrast == %s' % (
                      pas, contrast)))

analyses_order2 = [
    analysis('angle_vis', 'categorize', condition=[
        analysis('target_circAngle-seen', 'circ_regress'),
        analysis('target_circAngle-unseen', 'circ_regress')])
]
