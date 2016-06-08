import os
import numpy as np
from mne.io import read_info
from config import (paths, load, save, subjects_id,
                    missing_mri, bad_watershed, bad_mri)

subjects_dir = paths('freesurfer')
os.environ['SUBJECTS_DIR'] = subjects_dir

# Anatomy pipeline ------------------------------------------------------------
from jr.meg import anatomy_pipeline
if False:
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject not in missing_mri + bad_watershed + bad_mri:
            continue
        # Check or compute bem segmentation
        anatomy_pipeline(subject=subject, subjects_dir=subjects_dir,
                         overwrite=False)

if False:
    from mne.bem import (make_bem_model, write_bem_surfaces,
                         make_bem_solution, write_bem_solution)
    for subject in bad_watershed:
        bem_dir = os.path.join(subjects_dir, subject, 'bem')
        bem_fname = os.path.join(bem_dir, subject + '-5120-bem.fif')
        bem_sol_fname = os.path.join(bem_dir, subject + '-5120-bem-sol.fif')

        # single layer
        surfs = make_bem_model(subject=subject, subjects_dir=subjects_dir,
                               conductivity=(.3,))
        write_bem_surfaces(fname=bem_fname, surfs=surfs)
        bem = make_bem_solution(surfs)
        write_bem_solution(fname=bem_sol_fname, bem=bem)

# check anat
from mne.viz import plot_bem
if False:
    figs = list()
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject in missing_mri + bad_mri:
            continue

        # Plot BEM
        fig = plot_bem(subject=subject, subjects_dir=subjects_dir, show=True)

        # plot source space
        from mne import read_source_spaces
        from surfer import Brain  # noqa
        from mayavi import mlab  # noqa
        brain = Brain(subject, 'both', 'inflated')
        src = read_source_spaces(os.path.join(subjects_dir, subject, 'bem',
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
        if subject in missing_mri + bad_mri:
            continue
        raw_fname = paths('sss', subject=meg_subject, block=1)
        trans_fname = paths('trans', subject=meg_subject)
        # XXX for one subject the HPI were adequately triangulated before the
        # first block
        if subject == 'ps120458':
            raw_fname.split('1-sss.fif')[0] + '2-sss.fif'
        # Manual coregistration
        coregistration(subject=subject, subjects_dir=subjects_dir,
                       inst=raw_fname)
        # Plot
        info = read_info(raw_fname)
        plot_trans(info, trans_fname, subject=subject, dig=True,
                   meg_sensors=True)


def _copy_from_fsaverage(subject, subjects_dir, overwrite=False):
    """copy fsaverage files"""
    for this_dir in ['bem', 'surf']:
        bem_dir = os.path.join(subjects_dir, subject, this_dir)
        if not os.path.exists(bem_dir):
            os.makedirs(bem_dir)
    f_from = os.path.join(subjects_dir, 'fsaverage', 'bem',
                          'fsaverage-5120-bem.fif')
    f_to = os.path.join(subjects_dir, subject, 'bem',
                        '%s-5120-bem.fif' % subject)
    if overwrite or not os.path.exists(f_to):
        copyfile(f_from, f_to)
    surf_files = ['surf/lh.white', 'surf/rh.white',
                  'surf/lh.sphere', 'surf/rh.sphere',
                  'surf/lh.sphere.reg', 'surf/rh.sphere.reg']
    for fname in surf_files:
        f_from = os.path.join(subjects_dir, 'fsaverage', fname)
        f_to = os.path.join(subjects_dir, subject, fname)
        if overwrite or not(os.path.exists(f_to)):
            copyfile(f_from, f_to)

if False:
    # coregistration and source space for missing mri
    from shutil import copyfile
    anatomy_pipeline(subject='fsaverage', subjects_dir=subjects_dir,
                     overwrite=False)
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject not in missing_mri:
            continue
        print(subject)

        raw_fname = paths('sss', subject=meg_subject, block=1)
        trans_fname = paths('trans', subject=meg_subject)

        _copy_from_fsaverage(subject, subjects_dir=paths('freesurfer'),
                             overwrite=False)

        # Manual coregistration
        coregistration(subject='fsaverage', subjects_dir=subjects_dir,
                       inst=raw_fname)
        # Plot
        info = read_info(raw_fname)
        plot_trans(info, trans_fname, subject=subject, dig=True,
                   meg_sensors=True)

        # Compute BEM
        bem_dir = os.path.join(subjects_dir, subject, 'bem')
        src_fname = os.path.join(bem_dir, subject + '-oct-6-src.fif')

        # 3. Setup source space
        if not os.path.isfile(src_fname):
            from mne import setup_source_space
            setup_source_space(subject=subject, subjects_dir=subjects_dir,
                               fname=src_fname,
                               spacing='oct6', surface='white', overwrite=True,
                               add_dist=True, n_jobs=-1, verbose=None)


# Forward model ---------------------------------------------------------------
from jr.meg import forward_pipeline
if False:
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject in bad_mri:
            continue
        raw_fname = paths('sss', subject=meg_subject, block=1)
        fwd_fname = paths('fwd', subject=meg_subject)
        trans_fname = paths('trans', subject=meg_subject)
        forward_pipeline(raw_fname, fwd_fname=fwd_fname,
                         trans_fname=trans_fname,
                         subject=subject, subjects_dir=subjects_dir,
                         overwrite=False)

# Covariance -----------------------------------------------------------------
from mne.cov import compute_covariance
if False:
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject in bad_mri:
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
if False:
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject in bad_mri:
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
        if subject in bad_mri:
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
