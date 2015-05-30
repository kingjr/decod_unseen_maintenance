import os
import os.path as op

# Experiment parameters
open_browser = True
base_path = op.dirname(op.dirname(__file__))
data_path = op.join(base_path, '/media', 'niccolo', 'Paris', 'data')
script_path = '/home/niccolo/Dropbox/DOCUP/scripts/python/'
pyoutput_path = op.join(base_path, '/media', 'niccolo', 'ParisPy', 'data')
results_dir = op.join(base_path, 'results')
if not op.exists(results_dir):
    os.mkdir(results_dir)

subjects = [
    'ak130184', 'el130086', 'ga130053', 'gm130176', 'hn120493',
    'ia130315', 'jd110235', 'jm120476', 'ma130185', 'mc130295',
    'mj130216', 'mr080072', 'oa130317', 'rg110386', 'sb120316',
    'tc120199', 'ts130283', 'yp130276', 'av130322', 'ps120458']

# Define type of sensors used (useful for ICA correction, plotting etc)
from mne.channels import read_ch_connectivity
meg_connectivity, _ = read_ch_connectivity('neuromag306mag')
chan_types = [dict(name='meg', connectivity=meg_connectivity)]

# Decoding preprocessing steps
preproc = dict()

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

# ###################### Define contrasts #####################
absent = dict(cond='present', values=[0])
unseen = dict(cond='seen_unseen', values=[0])
seen = dict(cond='seen_unseen', values=[1])


contrasts = (
    dict(name='4visibilitiesPresent',
         include=dict(cond='response_visibilityCode', values=[1, 2, 3, 4]),
         exclude=[absent],
         clf=pipeline_svr, chance=0,
         scorer=scorer_spearman),
    dict(name='visibilityPresent',
         include=dict(cond='seen_unseen', values=[0, 1]),
         exclude=[absent],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='presentAbsent',
         include=dict(cond='present', values=[0, 1]),
         exclude=[],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='accuracy',
         include=dict(cond='correct', values=[0, 1]),
         exclude=[dict(cond='correct', values=[float('NaN')])],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='lambda',
         include=dict(cond='lambda', values=[1, 2]),
         exclude=[absent],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='tilt',
         include=dict(cond='tilt', values=[-1, 1]),
         exclude=[absent],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='responseButton',
         include=dict(cond='response_tilt', values=[-1, 1]),
         exclude=[dict(cond='response_tilt', values=[0])],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='targetAngle',
         include=dict(cond='orientation_target_rad',
                      values=angle2circle([15, 45, 75, 105, 135, 165])),
         exclude=[absent],
         clf=pipeline_svrangle, chance=1. / 6.,
         scorer=scorer_angle),
    dict(name='probeAngle',
         include=dict(cond='orientation_probe_rad',
                      values=angle2circle([15, 45, 75, 105, 135, 165])),
         exclude=[absent],
         clf=pipeline_svrangle, chance=1. / 6.,
         scorer=scorer_angle),
    dict(name='targetContrast',
         include=dict(cond='targetContrast', values=[0, .5, .75, 1]),
         exclude=[],
         clf=pipeline_svr, chance=0.,
         scorer=scorer_spearman),
)

# ###################### Define subscores #####################################
subscores = [
    dict(name='4visibilitiesPresentAll',
         contrast='4visibilitiesPresent',
         include=dict(cond='response_visibilityCode', values=[1, 2, 3, 4]),
         exclude=[absent],
         clf=pipeline_svr, chance=0,
         scorer=scorer_spearman),
    dict(name='visibilityPresentAll',
         contrast='visibilityPresent',
         include=dict(cond='seen_unseen', values=[0, 1]),
         exclude=[absent],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='presentAbsentAll',
         contrast='presentAbsent',
         include=dict(cond='present', values=[0, 1]),
         exclude=[],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='presentAbsentANDseen',
         contrast='presentAbsent',
         include=dict(cond='present', values=[0, 1]),
         exclude=[unseen],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='presentAbsentANDunseen',
         contrast='presentAbsent',
         include=dict(cond='present', values=[0, 1]),
         exclude=[seen],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='accuracyAll',
         contrast='accuracy',
         include=dict(cond='correct', values=[0, 1]),
         exclude=[dict(cond='correct', values=[float('NaN')])],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='accuracyANDseen',
         contrast='accuracy',
         include=dict(cond='correct', values=[0, 1]),
         exclude=[dict(cond='correct', values=[float('NaN')]), unseen],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='accuracyANDunseen',
         contrast='accuracy',
         include=dict(cond='correct', values=[0, 1]),
         exclude=[dict(cond='correct', values=[float('NaN')]), seen],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='lambdaAll',
         contrast='lambda',
         include=dict(cond='lambda', values=[1, 2]),
         exclude=[absent],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='lambdaANDseen',
         contrast='lambda',
         include=dict(cond='lambda', values=[1, 2]),
         exclude=[absent, unseen],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='lambdaANDunseen',
         contrast='lambda',
         include=dict(cond='lambda', values=[1, 2]),
         exclude=[absent, seen],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='tiltAll',
         contrast='tilt',
         include=dict(cond='tilt', values=[-1, 1]),
         exclude=[absent],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='tiltANDseen',
         contrast='tilt',
         include=dict(cond='tilt', values=[-1, 1]),
         exclude=[absent, unseen],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='tiltANDunseen',
         contrast='tilt',
         include=dict(cond='tilt', values=[-1, 1]),
         exclude=[absent, seen],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='responseButtonAll',
         contrast='responseButton',
         include=dict(cond='response_tilt', values=[-1, 1]),
         exclude=[dict(cond='response_tilt', values=[0])],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='responseButtonANDseen',
         contrast='responseButton',
         include=dict(cond='response_tilt', values=[-1, 1]),
         exclude=[dict(cond='response_tilt', values=[0]), unseen],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='responseButtonANDunseen',
         contrast='responseButton',
         include=dict(cond='response_tilt', values=[-1, 1]),
         exclude=[dict(cond='response_tilt', values=[0]), seen],
         clf=pipeline_svc, chance=.5,
         scorer=scorer_auc),
    dict(name='targetAngleAll',
         contrast='targetAngle',
         include=dict(cond='orientation_target_rad',
                      values=angle2circle([15, 45, 75, 105, 135, 165])),
         exclude=[absent],
         clf=pipeline_svrangle, chance=1. / 6.,
         scorer=scorer_angle),
    dict(name='targetAngleANDseen',
         contrast='targetAngle',
         include=dict(cond='orientation_target_rad',
                      values=angle2circle([15, 45, 75, 105, 135, 165])),
         exclude=[absent, unseen], chance=1. / 6.,
         scorer=scorer_angle),
    dict(name='targetAngleANDunseen',
         contrast='targetAngle',
         include=dict(cond='orientation_target_rad',
                      values=angle2circle([15, 45, 75, 105, 135, 165])),
         exclude=[absent, seen], chance=1. / 6.,
         scorer=scorer_angle),
    dict(name='probeAngleAll',
         contrast='probeAngle',
         include=dict(cond='orientation_probe_rad',
                      values=angle2circle([15, 45, 75, 105, 135, 165])),
         exclude=[absent],
         clf=pipeline_svrangle, chance=1. / 6.,
         scorer=scorer_angle),
    dict(name='probeAngleANDseen',
         contrast='probeAngle',
         include=dict(cond='orientation_probe_rad',
                      values=angle2circle([15, 45, 75, 105, 135, 165])),
         exclude=[absent, unseen],
         clf=pipeline_svrangle, chance=1. / 6.,
         scorer=scorer_angle),
    dict(name='probeAngleANDunseen',
         contrast='probeAngle',
         include=dict(cond='orientation_probe_rad',
                      values=angle2circle([15, 45, 75, 105, 135, 165])),
         exclude=[absent, seen],
         clf=pipeline_svrangle, chance=1. / 6.,
         scorer=scorer_angle),
    dict(name='targetContrastAll',
         contrast='targetContrast',
         include=dict(cond='targetContrast', values=[0, .5, .75, 1]),
         exclude=[],
         clf=pipeline_svr, chance=0.,
         scorer=scorer_spearman),
    dict(name='targetContrastANDseen',
         contrast='targetContrast',
         include=dict(cond='targetContrast', values=[0, .5, .75, 1]),
         exclude=[unseen],
         clf=pipeline_svr, chance=0.,
         scorer=scorer_spearman),
    dict(name='targetContrastANDunseen',
         contrast='targetContrast',
         include=dict(cond='targetContrast', values=[0, .5, .75, 1]),
         exclude=[seen],
         clf=pipeline_svr, chance=0.,
         scorer=scorer_spearman),
]

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
# #############################################################################
# univariate analyses definition: transform the input used for the decoding to
# the univariate analyse. Yes this NEEDS to be cleaned
from analyses_definition_univariate import format_analysis
analyses = [format_analysis(contrast) for contrast in contrasts]

# ############## Define type of input (erf,frequenct etc...) ##################
inputTypes = (
    dict(name='freq7'),
    dict(name='freq10'),
    dict(name='freq12'),
    dict(name='freq18'),
    dict(name='freq29'),
    dict(name='freq70'),
    dict(name='freq105'),
    dict(name='erf'),
)

# ##################################""
# # UNCOMMENT TO SUBSELECTION FOR FAST PROCESSING
# #
# subjects = [subjects[9]]
inputTypes = [inputTypes[-1]]
analyses = [ana for ana in analyses if ana['name'] == 'targetAngle']
preproc = dict(decim=2, crop=dict(tmin=-.2, tmax=1.200))
# preproc = dict(decim=2, crop=dict(tmin=-.1, tmax=1.100))
