import numpy as np
import os
import os.path as op
from mayavi import mlab  # noqa
from surfer import Brain  # noqa
from mne import read_source_spaces
from mne.cov import compute_covariance
from mne.minimum_norm import apply_inverse
import mne
from config import paths, load, subjects_id
os.environ['SUBJECTS_DIR'] = paths('freesurfer')

# select subject
subjects_dir = paths('freesurfer')
ii = 0
subject = subjects_id[ii]
meg_subject = ii+1

# files
raw_fname = paths('sss', subject=meg_subject, block=1)
save_dir = '/'.join(raw_fname.split('/')[:-1])
trans_fname = op.join(save_dir, subject + '-trans.fif')
fwd_fname = op.join(save_dir, subject + '-meg-fwd.fif')
bem_dir = op.join(subjects_dir, subject, 'bem')
bem_fname = op.join(bem_dir, subject + '-5120-bem-sol.fif')

# Times for covariance & baseline
tmin = -.200
tmax = 0.


# SOURCE
info = mne.io.read_info(raw_fname)
mne.viz.plot_trans(info, trans_fname, subject=subject, dig=True,
                   meg_sensors=True, subjects_dir=subjects_dir)
src = read_source_spaces(op.join(subjects_dir, subject, 'bem',
                         subject + '-oct-6-src.fif'))
brain = Brain(subject, 'lh', 'inflated')
surf = brain._geo
vertidx = np.where(src[0]['inuse'])[0]
mlab.points3d(surf.x[vertidx], surf.y[vertidx],
              surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)

# EVOKED
epochs = load('epochs_decim', subject=meg_subject, preload=True)
epochs.pick_types(meg=True, eeg=False, eog=False)
epochs.apply_baseline((tmin, tmax))
epochs.events[:, 2] = 1
evoked = epochs.average()
evoked.plot_joint(times=[0, .100, .180, .400])

# COVARIANCE
cov = compute_covariance(epochs, tmin=tmin, tmax=tmax)
cov.plot(epochs.info)
evoked = epochs.average()
evoked.plot_white(cov)

# DIPOLE
dip, residuals = mne.fit_dipole(
    evoked.copy().crop(0.18, 0.18),
    cov, bem_fname, trans_fname)
dip.plot_locations(trans_fname, subject)

# MNE
from mne.minimum_norm import make_inverse_operator
from mne import read_forward_solution
fwd = read_forward_solution(fwd_fname, surf_ori=True)
inv = make_inverse_operator(epochs.info, fwd, cov, loose=0.2, depth=0.8)
method = 'MNE'
snr = 3.0
stc = apply_inverse(evoked, inv, lambda2=1.0 / (2 ** snr),
                    method=method, pick_ori=None)
brain = stc.plot(hemi='both', subject=subject)
brain.set_time(180)
