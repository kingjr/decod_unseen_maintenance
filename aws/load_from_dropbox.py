from aws.utils import dropbox_download
from scripts.config import subjects, paths

for subject in subjects:
    # behavior
    f_local = paths('behavior', subject=subject)
    f_dropbox = f_local.split('/')[-1]
    dropbox_download(f_dropbox, f_local)
    # epoch light
    f_local = paths('epoch', subject=subject, data_type='erf')
    f_dropbox = subject + '_preprocessed.mat'
    dropbox_download(f_dropbox, f_local)
