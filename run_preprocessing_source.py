import os
import numpy as np
from mne.io import read_info
from config import paths, load, save, subjects_id, missing_mri


missing_mri += ['av130322',  # missing temporal cortex
                'jd110235',  # need to fix bem
                'oa130317',  # need to fix bem
                'ps120458']  # need to fix HPI

subjects_dir = paths('freesurfer')
os.environ['SUBJECTS_DIR'] = subjects_dir

# Anatomy pipeline ------------------------------------------------------------
from jr.meg import anatomy_pipeline
if False:
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject in missing_mri:
            continue
        # Check or compute bem segmentation
        anatomy_pipeline(subject=subject, subjects_dir=subjects_dir,
                         overwrite=False)

# check anat
from mne.viz import plot_bem
if False:
    figs = list()
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject in missing_mri or meg_subject <= 3:
            continue

        # Plot BEM
        fig = plot_bem(subject=subject, subjects_dir=subjects_dir, show=True)

        # plot source space
        import os.path as op
        from mne import read_source_spaces
        from surfer import Brain  # noqa
        from mayavi import mlab  # noqa
        brain = Brain(subject, 'both', 'inflated')
        src = read_source_spaces(op.join(subjects_dir, subject, 'bem',
                                 subject + '-oct-6-src.fif'))
        brain = Brain(subject, 'both', 'inflated')
        for ii in range(2):
            surf = brain.brains[ii]._geo
            vertidx = np.where(src[ii]['inuse'])[0]
            mlab.points3d(surf.x[vertidx], surf.y[vertidx],
                          surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)
        raw_input('next?')

# Coregistration --------------------------------------------------------------
from mne.viz import plot_trans
from mne.gui import coregistration
if False:
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject in missing_mri:
            continue
        raw_fname = paths('sss', subject=meg_subject, block=1)
        trans_fname = paths('trans', subject=meg_subject)
        # Manual coregistration
        coregistration(subject=subject, subjects_dir=subjects_dir,
                       inst=raw_fname)
        # Plot
        info = read_info(raw_fname)
        plot_trans(info, trans_fname, subject=subject, dig=True,
                   meg_sensors=True)

# Forward model ---------------------------------------------------------------
from jr.meg import forward_pipeline
if False:
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject in missing_mri:
            continue
        raw_fname = paths('sss', subject=meg_subject, block=1)
        fwd_fname = paths('fwd', subject=meg_subject)
        trans_fname = paths('trans', subject=meg_subject)
        forward_pipeline(raw_fname, fwd_fname=None, trans_fname=trans_fname,
                         subject=subject, subjects_dir=subjects_dir,
                         overwrite=False)

# Covariance -----------------------------------------------------------------
from mne.cov import compute_covariance
if True:
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject in missing_mri:
            continue
        # Preproc
        epochs = load('epochs_decim', subject=meg_subject, preload=True)
        epochs.pick_types(meg=True, eeg=False, eog=False)
        # epochs = epochs[:160]  # compute covariance from a single run
        epochs.apply_baseline((None, 0))
        # Compute covariance on same window as baseline
        cov = compute_covariance(epochs, tmin=epochs.times[0], tmax=0.,
                                 method='shrunk')
        save(cov, 'cov', subject=meg_subject, overwrite=True)

# Inverse --------------------------------------------------------------------
from mne.minimum_norm import make_inverse_operator
if True:
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject in missing_mri:
            continue
        raw_fname = paths('sss', subject=meg_subject, block=1)
        inv_fname = paths('inv', subject=meg_subject)
        cov = load('cov', subject=meg_subject)
        fwd = load('fwd', subject=meg_subject)
        info = read_info(raw_fname)
        inv = make_inverse_operator(info, fwd, cov, loose=0.2, depth=0.8)
        save(inv, 'inv', subject=meg_subject, overwrite=True)

# Morph ----------------------------------------------------------------------
from mne import EvokedArray
from mne import compute_morph_matrix
from mne.minimum_norm import apply_inverse
if False:
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject in missing_mri:
            continue
        raw_fname = paths('sss', subject=meg_subject, block=1)
        info = read_info(raw_fname)

        # precompute morphing matrix for faster processing
        inv = load('inv', subject=meg_subject)
        evoked = EvokedArray(np.zeros((len(info['chs']), 2)), info, 0)
        evoked.pick_types(eeg=False, meg=True)
        stc = apply_inverse(evoked, inv, lambda2=1.0 / (2 ** 3.0),
                            method='dSPM', pick_ori=None)
        morph = compute_morph_matrix(subject, 'fsaverage', stc.vertices,
                                     vertices_to=[np.arange(10242)] * 2,
                                     subjects_dir=subjects_dir)
        save(morph, 'morph', subject=meg_subject)
