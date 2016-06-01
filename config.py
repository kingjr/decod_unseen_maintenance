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
    data_path = '/media/jrking/harddrive/Niccolo/'

client = Client('S3', bucket='meg.niccolo', client_root=data_path)

# Setup online HTML report
report = OnlineReport()

# Experiment parameters
base_path = op.dirname(op.dirname(__file__))
print base_path

subjects = range(1, 21)


def paths(typ, subject='fsaverage', analysis='analysis', block=999):
    subject = 's%i' % subject if isinstance(subject, int) else subject
    this_path = op.join(data_path, subject, typ)
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
        fwd=op.join(this_path, '%s-fwd.fif' % subject),
        cov=op.join(this_path, '%s-cov.fif' % subject),
        inv=op.join(this_path, '%s-inv.fif' % subject),
        # XXX FIXME no pickle!
        evoked=op.join(this_path, '%s_%s.pickle' % (subject, analysis)),
        decod=op.join(this_path, '%s_%s.pickle' % (subject, analysis)),
        score=op.join(this_path, '%s_%s_scores.pickle' % (subject, analysis)),
        freesurfer=op.join(data_path, 'subjects'),
        covariance=op.join(this_path, '%s-meg-cov.fif' % (subject)))
    this_file = path_template[typ]

    # Create subfolder if necessary
    folder = os.path.dirname(this_file)
    if (folder != '') and (not op.exists(folder)):
        os.makedirs(folder)

    return this_file


def load(typ, subject='fsaverage', analysis='analysis', block=999,
         download=False, preload=False):
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
    elif typ in ['epo_block', 'epochs']:
        out = read_epochs(fname, preload=preload)
    elif typ in ['evoked', 'decod', 'score']:
        with open(fname, 'rb') as f:
            out = pickle.load(f)
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
    if typ in ['epo_block', 'epochs']:
        var.save(fname)
    elif typ in ['evoked', 'decod', 'score']:
        with open(fname, 'wb') as f:
            pickle.dump(var, f)
    else:
        raise NotImplementedError()
    if upload:
        client.upload(fname)
    return True

# Analysis Parameters
tois = [(-.100, 0.050), (.100, .250), (.300, .800), (.900, 1.050)]
