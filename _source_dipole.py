from mne.datasets.sample import data_path as sample_path
from os.path import join
from mne import fit_dipole, compute_covariance, read_epochs

meg_path = '/media/jrking/harddrive/Niccolo/data/s1/'
fs_path = '/media/jrking/harddrive/Niccolo/subjects/'

subject_meg = 's1'        # my subject id for meg data
subject_mri = 'ak130184'  # my subject id for mri data

sss_fname = join(meg_path, 'sss', subject_meg + '_1-sss.fif')
epo_fname = join(meg_path, 'epochs_decim', subject_meg + '_decim-epo.fif')
trans_fname = join(meg_path, 'sss', subject_mri + '-trans.fif')
bem_fname = join(fs_path, subject_mri, 'bem',
                 subject_mri + '-5120-bem-sol.fif')

epochs = read_epochs(epo_fname, preload=True)
epochs.pick_types(eeg=False)
epochs.apply_baseline((-.200, 0))
evoked = epochs.average()
evoked.copy().pick_types(meg='mag', eeg=False).plot_joint(times=[.180])
cov = compute_covariance(epochs, tmin=-.200, tmax=0)

# test with sample works ok
for subject, subjects_dir in [
        ('sample', join(sample_path(), 'subjects')),  # sample
        (subject_mri, fs_path)]:                      # my subject
    # fit dipole using my subject data
    dip, residuals = fit_dipole(
        evoked.copy().crop(0.18, 0.18),
        cov, bem_fname, trans_fname)
    # plot using either sample or my subject
    dip.plot_locations(trans_fname, subject, subjects_dir)
