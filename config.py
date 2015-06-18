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

# ###################### Define contrasts #####################

# #############################################################################
# univariate analyses definition: transform the input used for the decoding to
# the univariate analyse. Yes this NEEDS to be cleaned
# from analyses_definition_univariate import format_analysis
# analyses = [format_analysis(contrast) for contrast in contrasts]

# ############## Define type of input (erf,frequenct etc...) ##################
data_types = ['erf'] + ['freq' + freq for freq in [7, 10, 12, 18, 29, 70, 105]]

# ##################################""
# # UNCOMMENT TO SUBSELECTION FOR FAST PROCESSING
# #
# subjects = [subjects[9]]
data_types = [data_types[0]]
# analyses = [ana for ana in analyses if ana['name'] == 'targetAngle']
preproc = dict(decim=2, crop=dict(tmin=-.2, tmax=1.200))
# preproc = dict(decim=2, crop=dict(tmin=-.1, tmax=1.100))
