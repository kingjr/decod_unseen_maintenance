from scripts.config import subjects, paths

from aws.utils import (upload, download)

import os.path as op

for subject in subjects:
    subject = subjects[0]
    # behavior
    f_local = paths('behavior', subject=subject)
    f_dropbox = op.join('/no_sync/niccolo_data/', f_local.split('/')[-1])
    download('dropbox', f_dropbox, f_local)
    upload('s3', f_local, f_local)
    # # epoch light
    # f_local = paths('epoch', subject=subject, data_type='erf')
    # f_dropbox = subject + '_preprocessed.mat'
    # download('dropbox', f_dropbox, f_local)
    # upload('s3', f_local, f_local)
