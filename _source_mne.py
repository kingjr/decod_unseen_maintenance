from mne.minimum_norm import apply_inverse
from config import load, subjects_id, paths

subject_meg = 1
subject = subjects_id[1]
subjects_dir = paths('freesurfer')

epochs = load('epochs_decim', subject=subject_meg, preload=True)
epochs.pick_types(eeg=False)
epochs.apply_baseline((-.200, 0))
evoked = epochs.average()
evoked.copy().pick_types(meg='mag', eeg=False).plot_joint(times=[.180])

cov = load('cov', subject=subject_meg)
inv = load('inv', subject=subject_meg)
fwd = load('fwd', subject=subject_meg)

snr = 3.
lambda2 = 1. / snr ** 2
stc = apply_inverse(evoked, inv)

brain = stc.plot(surface='inflated', hemi='both', subjects_dir=subjects_dir)
brain.set_time(180)
