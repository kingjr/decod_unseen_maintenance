from scripts.config import paths, subjects

from mne.gui import coregistration
subjects_dir = paths('freesurfer')
for subject in subjects:
    coregistration(subject=subject, subjects_dir=subjects_dir,
                   inst=paths('sss', subject=subject, run=1))
