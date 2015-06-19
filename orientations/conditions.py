# Decoding parameters
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from utils import (
    clf_2class_proba, SVR_angle, angle2circle,
    scorer_angle, scorer_auc, scorer_spearman)

scaler = StandardScaler()

# SVC
svc = clf_2class_proba(C=1, class_weight='auto')
pipeline_svc = Pipeline([('scaler', scaler), ('svc', svc)])

# SVR
svr = LinearSVR(C=1)
pipeline_svr = Pipeline([('scaler', scaler), ('svr', svr)])

# SVR angles
pipeline_svrangle = SVR_angle()

absent = dict(cond='present', values=[0])
unseen = dict(cond='seen_unseen', values=[0])
seen = dict(cond='seen_unseen', values=[1])
missed = dict(cond='response_tilt', values=[0])
angles = angle2circle([15, 45, 75, 105, 135, 165])


from .utils import evoked_spearman, evoked_subtract, evoked_circularlinear


def analysis(name, include, exclude=[absent], contrast=None, typ=None):
    if typ == 'contrast' or len(include['values']) == 2:
        clf = pipeline_svc
        scorer = scorer_auc
        operator = evoked_subtract
        chance = .5
    elif typ == 'regression' or len(include['values']) > 2:
        clf = pipeline_svr
        scorer = scorer_spearman
        chance = 0.
        operator = evoked_spearman
    elif typ == 'circ_regression':
        clf = pipeline_svrangle
        scorer = scorer_angle
        chance = 1. / 6.
        operator = evoked_circularlinear
    return dict(name=name, include=include, exclude=exclude, clf=clf,
                chance=chance, scorer=scorer, contrast=None, operator=operator)

analyses = (
    analysis('s_presence', dict(cond='present', values=[0, 1]), exclude=[]),
    analysis('s_targetContrast',
             dict(cond='targetContrast', values=[0, .5, .75, 1]), exclude=[]),
    analysis('s_lambda', dict(cond='lambda', values=[1, 2])),
    analysis('s_targetAngle', dict(cond='orientation_target_rad', values=angles),
             typ='circ_regression'),
    analysis('s_probeAngle', dict(cond='orientation_probe_rad', values=angles),
             typ='circ_regression'),
    analysis('s_tilt', dict(cond='tilt', values=[-1, 1])),
    analysis('m_responseButton', dict(cond='response_tilt', values=[-1, 1]),
             exclude=[missed]),
    analysis('m_accuracy', dict(cond='correct', values=[0, 1])),
    analysis('m_visibilities',
             dict(cond='response_visibilityCode', values=[1, 2, 3, 4])),
    analysis('m_seen', dict(cond='seen_unseen', values=[0, 1])),
)

# ###################### Define subscores #####################################

subscores = []
for analysis in analyses:
    analysis['contrast'] = analysis['name']
    subscores.append(analysis)
    # subdivide by visibility
    if analysis['name'] not in ['m_visibilities', 'm_seen']:
        analysis_ = analysis
        analysis_['name'] += '-seen'
        analysis_['exclude'] += [unseen]
        subscores.append(analysis_)
        analysis_ = analysis
        analysis_['name'] += '-unseen'
        analysis_['exclude'] += [seen]
        subscores.append(analysis_)

# ############# Define second-order subscores #################################
subscores2 = []

for analysis in analyses:
    if analysis['name'] not in ['m_visibilities', 'm_seen']:
        analysis['contrast1'] = analysis_['name'] + '-seen'
        analysis['contrast2'] = analysis_['name'] + '-unseen'
        analysis['chance'] = 0
        subscores2.append(analysis)


def format_analysis(contrast):
    """This functions takes the contrasts defined for decoding  and format it
    so as to be usable by the univariate scripts

    We need to homogeneize the two types of analysis definitions
    """

    # exclude
    exclude = dict()
    for exclude_ in contrast['exclude']:
        cond = exclude_['cond']
        exclude[cond] = exclude_['values']

    # include
    conditions = list()
    cond = contrast['include']['cond']
    for value in contrast['include']['values']:
        include_ = dict()
        include_[cond] = value
        conditions.append(dict(name=cond + str(value), include=include_,
                               exclude=exclude))
    contrast['conditions'] = conditions
    return contrast
