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
angles = angle2circle([15, 45, 75, 105, 135, 165])


def analysis(name, include, exclude=[absent], clf=None, scorer=None,
             change=None, chance=None, contrast=None):
    if len(include['values']) == 2:
        # Default contrast analyses
        clf = pipeline_svc if clf is None else clf
        scorer = scorer_auc if scorer is None else scorer
        chance = .5 if chance is None else chance
    else:
        # Default regression analysis
        clf = pipeline_svr if clf is None else clf
        scorer = scorer_spearman if scorer is None else scorer
        chance = 0. if chance is None else chance
    return dict(name=name, include=include, exclude=exclude, clf=clf,
                chance=chance, scorer=scorer, contrast=None)

analyses = (
    analysis('s_presence', dict(cond='present', values=[0, 1]), exclude=[]),
    analysis('s_targetContrast',
             dict(cond='targetContrast', values=[0, .5, .75, 1]), exclude=[]),
    analysis('s_lambda', dict(cond='lambda', values=[1, 2])),
    analysis('s_targetAngle', dict(cond='orientation_target_rad', values=angles),
             clf=pipeline_svrangle, chance=1. / 6., scorer=scorer_angle),
    analysis('s_probeAngle', dict(cond='orientation_probe_rad', values=angles),
             clf=pipeline_svrangle, chance=1. / 6., scorer=scorer_angle),
    analysis('s_tilt', dict(cond='tilt', values=[-1, 1])),
    analysis('m_responseButton', dict(cond='response_tilt', values=[-1, 1]),
             exclude=[dict(cond='response_tilt', values=[0])]),
    analysis('m_accuracy', dict(cond='correct', values=[0, 1])),  # XXX Absent?
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
subscores2 = [
    dict(name='presentAbsent:seenVSunseen',
         contrast1='presentAbsentANDseen',
         contrast2='presentAbsentANDunseen',
         include=dict(cond='present', values=[0, 1]),
         exclude=[unseen],
         clf=pipeline_svc, chance=0,
         scorer=scorer_auc),
    dict(name='accuracy:seenVSunseen',
         contrast1='accuracyANDseen',
         contrast2='accuracyANDunseen',
         include=dict(cond='correct', values=[0, 1]),
         exclude=[dict(cond='correct', values=[float('NaN')]), unseen],
         clf=pipeline_svc, chance=0,
         scorer=scorer_auc),
    dict(name='lambda:seenVSunseen',
         contrast1='lambdaANDseen',
         contrast2='lambdaANDunseen',
         include=dict(cond='lambda', values=[1, 2]),
         exclude=[absent, unseen],
         clf=pipeline_svc, chance=0,
         scorer=scorer_auc),
    dict(name='tilt:seenVSunseen',
         contrast1='tiltANDseen',
         contrast2='tiltANDunseen',
         include=dict(cond='tilt', values=[-1, 1]),
         exclude=[absent, unseen],
         clf=pipeline_svc, chance=0,
         scorer=scorer_auc),
    dict(name='responseButton:seenVSunseen',
         contrast1='responseButtonANDseen',
         contrast2='responseButtonANDseen',
         include=dict(cond='response_tilt', values=[-1, 1]),
         exclude=[dict(cond='response_tilt', values=[0]), unseen],
         clf=pipeline_svc, chance=0,
         scorer=scorer_auc),

]


def format_analysis(contrast):
    """This functions takes the contrasts defined for decoding  and format it
    so as to be usable by the univariate scripts

    We need to homogeneize the two types of analysis definitions
     """
    from .utils import evoked_spearman, evoked_subtract
    name = contrast['name']
    if contrast['scorer'] == scorer_spearman:
        operator = evoked_spearman
    elif contrast['scorer'] == scorer_auc:
        operator = evoked_subtract
    elif contrast['scorer'] == scorer_angle:
        # TODO evoked_vtest
        return
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
    analysis = dict(name=name, operator=operator, conditions=conditions)
    return analysis
