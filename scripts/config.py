import sys
sys.path.insert(0, './')
import matplotlib
matplotlib.use('Agg')
import numpy as np
# from time import gmtime, strftime
# launch_time = strftime("_%Y-%m-%d_%H-%M-%S_", gmtime())
import os
import os.path as op

# Experiment parameters
open_browser = False
base_path = op.dirname(op.dirname(__file__))
print base_path
data_path = op.join(base_path, 'data/')
data_path = '/media/jrking/My Passport/Niccolo/data/'
# XXX what to do with this ad hoc paths?
# script_path = '/home/niccolo/Dropbox/DOCUP/scripts/python/'
# pyoutput_path = op.join(base_path, '/media', 'niccolo', 'ParisPy', 'data')


def paths(typ, subject='fsaverage', data_type='erf', lock='target',
          analysis='analysis', pyscript='config.py', log=False):
    # FIXME: cleanup epochs filenames
    this_path = op.join(data_path, subject, typ)
    path_template = dict(
        base_path=base_path,
        data_path=data_path,
        report=op.join(base_path, 'results'),
        log=op.join(base_path, pyscript.strip('.py') + '.log'),
        behavior=op.join(this_path, '%s_fixed.mat' % subject),
        epoch=op.join(this_path, '%s_%s_%s.mat' % (subject, lock, data_type)),
        evoked=op.join(this_path, '%s_%s_%s_%s.pickle' % (
            subject, lock, data_type, analysis)),
        decod=op.join(this_path, '%s_%s_%s_%s.pickle' % (
            subject, lock, data_type, analysis)),
        score=op.join(this_path, '%s_%s_%s_%s_scores.pickle' % (
            subject, lock, data_type, analysis)))
    file = path_template[typ]

    # Log file ?
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

subjects = [
    'ak130184', 'el130086', 'ga130053', 'gm130176', 'hn120493',
    'ia130315', 'jd110235', 'jm120476', 'ma130185', 'mc130295',
    'mj130216', 'mr080072', 'oa130317', 'rg110386', 'sb120316',
    'tc120199', 'ts130283', 'yp130276', 'av130322', 'ps120458']

# Define type of sensors used (useful for ICA correction, plotting etc)
# FIXME unknown connectivity; must be mag
# from mne.channels import read_ch_connectivity
# meg_connectivity, _ = read_ch_connectivity('neuromag306meg')
chan_types = [dict(name='meg')]

# Decoding preprocessing steps
preproc = dict()

# ###################### Define contrasts #####################
from orientations.conditions import analyses, subscores, analyses_order2

# #############################################################################
# univariate analyses definition: transform the input used for the decoding to
# the univariate analyse. Yes this NEEDS to be cleaned
# from analyses_definition_univariate import format_analysis
# analyses = [format_analysis(contrast) for contrast in contrasts]

# ############## Define type of input (erf,frequenct etc...) ##################
data_types = ['erf'] + ['freq%s' % f for f in [7, 10, 12, 18, 29, 70, 105]]


# EXTERNAL PARAMETER ##########################################################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--time_id', default='')
parser.add_argument('--subject', default=None)
parser.add_argument('--data_type', default=data_types)
parser.add_argument('--analysis', default=analyses)
parser.add_argument('--pyscript', default='config.py')
args = parser.parse_args()


time_id = args.time_id
subjects = [args.subject] if args.subject is not None else subjects
if isinstance(args.analysis, str):
    idx = np.where([d['name'] == args.analysis for d in analyses])[0]
    analyses = [analyses[idx]]
pyscript = args.pyscript

# ##################################""
# # UNCOMMENT TO SUBSELECTION FOR FAST PROCESSING
# #
# subjects = [subjects[0]]
data_types = [data_types[0]]
# analyses = [ana for ana in analyses if ana['name'] == 'target_present']
preproc = dict(crop=dict(tmin=-.1, tmax=1.100))
# preproc = dict(decim=2, crop=dict(tmin=-.1, tmax=1.100))
