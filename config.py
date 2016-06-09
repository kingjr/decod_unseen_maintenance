import numpy as np
import pickle
import os
import os.path as op
from mne import read_epochs
from mne.io import Raw
from jr.utils import OnlineReport
from jr.cloud import Client

# Setup paths
aws = False
if 'aws' in os.environ.keys() and os.environ['aws'] == 'True':
    aws = True
if aws:
    data_path = '/home/ubuntu/decod_unseen_maintenance/data/'
else:
    data_path = '/media/jrking/harddrive/Niccolo/data/'

client = Client('S3', bucket='meg.niccolo', client_root=data_path)

# Setup online HTML report
report = OnlineReport()

# Experiment parameters
base_path = op.dirname(op.dirname(__file__))
print base_path

subjects = range(1, 21)

subjects_id = ['ak130184', 'el130086', 'ga130053', 'gm130176', 'hn120493',
               'ia130315', 'jd110235', 'jm120476', 'ma130185', 'mc130295',
               'mj130216', 'mr080072', 'oa130317', 'rg110386', 'sb120316',
               'tc120199', 'ts130283', 'yp130276', 'av130322', 'ps120458']
missing_mri = ['gm130176',  'ia130315', 'jm120476', 'mc130295', 'ts130283',
               'yp130276']
bad_watershed = ['jd110235', 'oa130317']
bad_mri = ['av130322']  # missing temporal cortex!


def paths(typ, subject='fsaverage', analysis='analysis', block=999):
    subject = 's%i' % subject if isinstance(subject, int) else subject
    this_path = op.join(data_path, subject, typ)
    if typ in ['fwd', 'inv', 'cov', 'morph', 'trans', 'stc']:
        this_path = op.join(data_path, subject, 'source')
    path_template = dict(
        base_path=base_path,
        data_path=data_path,
        behavior=op.join(this_path, '%s_behav.mat' % subject),
        # XXX FIXME CLEAN sss naming
        sss=op.join(this_path, '%s_%i-sss.fif' % (subject, block)),
        epo_block=op.join(this_path,
                          '%s_%i_filtered-epo.fif' % (subject, block)),
        epochs=op.join(this_path, '%s-epo.fif' % subject),
        epochs_decim=op.join(this_path, '%s_decim-epo.fif' % subject),
        epochs_vhp=op.join(this_path, '%s_vhp-epo.fif' % subject),
        trans=op.join(this_path, '%s-trans.fif' % subject),
        fwd=op.join(this_path, '%s-fwd.fif' % subject),
        cov=op.join(this_path, '%s-cov.fif' % subject),
        inv=op.join(this_path, '%s-inv.fif' % subject),
        morph=op.join(this_path, '%s-morph.npz' % subject),
        # XXX FIXME no pickle!
        evoked=op.join(this_path, '%s_%s.pickle' % (subject, analysis)),
        evoked_source=op.join(this_path, '%s_%s.pickle' % (subject, analysis)),
        decod=op.join(this_path, '%s_%s.pickle' % (subject, analysis)),
        decod_tfr=op.join(this_path, '%s_%s_tfr.pickle' % (subject, analysis)),
        score=op.join(this_path, '%s_%s_scores.pickle' % (subject, analysis)),
        score_tfr=op.join(this_path,
                          '%s_%s_tfr_scores.pickle' % (subject, analysis)),
        score_source=op.join(
            this_path, '%s_%s_scores.npy' % (subject, analysis)),
        score_pval=op.join(
            this_path, '%s_%s_pval.npy' % (subject, analysis)),
        freesurfer=op.join('data/'.join(data_path.split('data/')[:-1]),
                           'subjects'))
    this_file = path_template[typ]

    # Create subfolder if necessary
    folder = os.path.dirname(this_file)
    if (folder != '') and (not op.exists(folder)):
        os.makedirs(folder)

    return this_file


def load(typ, subject='fsaverage', analysis='analysis', block=999,
         download=True, preload=False):
    """Auxiliary saving function."""
    # get file name
    fname = paths(typ, subject=subject, analysis=analysis, block=block)

    # check if file exists
    if not op.exists(fname) and download:
        client.download(fname)

    # different data format depending file type
    if typ == 'behavior':
        from base import read_events
        out = read_events(fname)
    elif typ == 'sss':
        out = Raw(fname, preload=preload)
    elif typ in ['epo_block', 'epochs', 'epochs_decim', 'epochs_vhp']:
        out = read_epochs(fname, preload=preload)
    elif typ in ['cov']:
        from mne.cov import read_cov
        out = read_cov(fname)
    elif typ in ['fwd']:
        from mne import read_forward_solution
        out = read_forward_solution(fname, surf_ori=True)
    elif typ in ['inv']:
        from mne.minimum_norm import read_inverse_operator
        out = read_inverse_operator(fname)
    elif typ in ['evoked', 'decod', 'decod_tfr', 'score', 'score_tfr',
                 'evoked_source']:
        with open(fname, 'rb') as f:
            out = pickle.load(f)
    elif typ == 'morph':
        from scipy.sparse import csr_matrix
        loader = np.load(fname)
        out = csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])
    elif typ in ['score_source', 'score_pval']:
        out = np.load(fname)
    else:
        raise NotImplementedError()
    return out


def save(var, typ, subject='fsaverage', analysis='analysis', block=999,
         upload=aws, overwrite=False):
    """Auxiliary saving function."""
    # get file name
    fname = paths(typ, subject=subject, analysis=analysis, block=block)

    # check if file exists
    if op.exists(fname) and not overwrite:
        print('%s already exists. Skipped' % fname)
        return False

    # different data format depending file type
    if typ in ['epo_block', 'epochs', 'epochs_decim', 'cov', 'epochs_vhp']:
        var.save(fname)
    elif typ in ['evoked', 'decod', 'decod_tfr', 'score', 'score_tfr',
                 'evoked_source']:
        with open(fname, 'wb') as f:
            pickle.dump(var, f)
    elif typ in ['inv']:
        from mne.minimum_norm import write_inverse_operator
        write_inverse_operator(fname, var)
    elif typ in ['fwd']:
        from mne import write_forward_solution
        write_forward_solution(fname, var)
    elif typ == 'morph':
        np.savez(fname, data=var.data, indices=var.indices,
                 indptr=var.indptr, shape=var.shape)
    elif typ in ['score_source', 'score_pval']:
        np.save(fname, var)
    else:
        raise NotImplementedError()
    if upload:
        client.upload(fname)
    return True

# Analysis Parameters
tois = np.array([[-.100, 0.050], [.100, .250], [.300, .800], [.900, 1.050]])
