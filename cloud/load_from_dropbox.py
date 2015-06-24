from scripts.config import subjects, paths

from aws.utils import (upload, download)

import os
import os.path as op

for subject in subjects:
    # behavior
    f_local = paths('behavior', subject=subject)
    f_dropbox = op.join('/no_sync/niccolo_data/',
                        f_local.split('/')[-1])
    download('dropbox', f_dropbox, f_local, True)
    upload('s3', f_local, f_local, True)

    # epoch light
    f_local = paths('epoch', subject=subject, data_type='erf')
    f_dropbox = op.join('/no_sync/niccolo_data/',
                        subject + '_preprocessed.mat')
    download('dropbox', f_dropbox, f_local)
    upload('s3', f_local, f_local, True)

    # epoch heavy
    f_local = paths('epoch', subject=subject, data_type='erf')
    f_local = f_local[:-4] + '.dat'
    f_dropbox = op.join('/no_sync/niccolo_data/',
                        subject + '_preprocessed.dat')
    download('dropbox', f_dropbox, f_local)
    upload('s3', f_local, f_local, True)
    if subject in subjects[1:]:
        os.remove(f_local)
