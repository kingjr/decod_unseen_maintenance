import numpy as np
import os
import os.path as op
from jr.utils import OnlineReport

# Setup online HTML report
report = OnlineReport()

# Experiment parameters
base_path = op.dirname(op.dirname(__file__))
print base_path
data_path = op.join(base_path, 'data/')
data_path = '/media/jrking/harddrive/Niccolo/data/'


def paths(typ, subject='fsaverage', data_type='erf', lock='target',
          analysis='analysis', pyscript='config.py', run=1, log=False):
    # FIXME: cleanup epochs filenames
    this_path = op.join(data_path, subject, typ)
    path_template = dict(
        base_path=base_path,
        data_path=data_path,
        report=op.join(base_path, 'results'),
        log=op.join(base_path, pyscript.strip('.py') + '.log'),
        behavior=op.join(this_path, '%s_fixed.mat' % subject),
        raw=op.join(this_path, '%s_run%02i.fif' % (subject, run)),
        sss=op.join(this_path, '%s_run%02i_sss.fif' % (subject, run)),
        epoch=op.join(this_path, '%s_%s_%s.mat' % (subject, lock, data_type)),
        fwd=op.join(this_path, '%s-fwd.fif' % subject),
        cov=op.join(this_path, '%s-cov.fif' % subject),
        inv=op.join(this_path, '%s-inv.fif' % subject),
        evoked=op.join(this_path, '%s_%s_%s_%s.pickle' % (
            subject, lock, data_type, analysis)),
        decod=op.join(this_path, '%s_%s_%s_%s.pickle' % (
            subject, lock, data_type, analysis)),
        score=op.join(this_path, '%s_%s_%s_%s_scores.pickle' % (
            subject, lock, data_type, analysis)),
        freesurfer=op.join(data_path, 'subjects'),
        covariance=op.join(this_path, '%s-meg-cov.fif' % (subject)))
    file = path_template[typ]

    # Option to Log the file in order to facilitate the upload and download
    # onto and from the Amazon Servers
    if log:
        fname = paths('log')
        print '%s: %s ' % (fname, file)
        with open(fname, "a") as myfile:
            myfile.write("%s \n" % file)

    # Create subfolder if necessary
    folder = os.path.dirname(file)
    if (folder != '') and (not op.exists(folder)):
        os.makedirs(folder)

    return file

# Subjects pseudonimized ID
subjects = [
    'ak130184', 'el130086', 'ga130053', 'gm130176', 'hn120493',
    'ia130315', 'jd110235', 'jm120476', 'ma130185', 'mc130295',
    'mj130216', 'mr080072', 'oa130317', 'rg110386', 'sb120316',
    'tc120199', 'ts130283', 'yp130276', 'av130322', 'ps120458']

missing_mri = [
    'gm130176',  'ia130315', 'jm120476', 'mc130295', 'ts130283', 'yp130276']

runs = range(1, 6)

# Define type of sensors used (useful for ICA correction, plotting etc)
chan_types = [dict(name='meg')]

# Decoding preprocessing steps
preproc = dict()

# ###################### Define contrasts #####################
from scripts.conditions import analyses, subscores, analyses_order2

# #############################################################################
# univariate analyses definition: transform the input used for the decoding to
# the univariate analyse. Yes this NEEDS to be cleaned
# from analyses_definition_univariate import format_analysis
# analyses = [format_analysis(contrast) for contrast in contrasts]

# EXTERNAL PARAMETER ##########################################################
# import argparse
# # This is to facilitate cluster analyses by passing parameters externally
# parser = argparse.ArgumentParser()
# parser.add_argument('--time_id', default='')
# parser.add_argument('--subject', default=None)
# parser.add_argument('--analysis', default=analyses)
# parser.add_argument('--pyscript', default='config.py')
# args = parser.parse_args()
#
# time_id = args.time_id
# subjects = [args.subject] if args.subject is not None else subjects
# if isinstance(args.analysis, str):
#     idx = np.where([d['name'] == args.analysis for d in analyses])[0]
#     analyses = [analyses[idx]]
# pyscript = args.pyscript

# Analysis Parameters
tois = [(-.100, 0.050), (.100, .250), (.300, .800), (.900, 1.050)]

# ##################################""
# # UNCOMMENT TO SUBSELECTION FOR FAST PROCESSING
# subjects = [subjects[0]]
# analyses = [ana for ana in analyses if ana['name'] == 'target_present']
preproc = dict(crop=dict(tmin=-.1, tmax=1.100))
