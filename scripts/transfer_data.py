import sys
sys.path.insert(0, './')


def download_all(overwrite=False):
    from itertools import product
    from cloud.utils import download
    # XXX This will have to be specified for later scripts
    from scripts.config import paths, subjects, data_types

    file_list = list()
    # behavior
    for subject in subjects:
        file_list.append(paths('behavior', subject=subject))
    # epoch
    for subject, data_type in product(subjects, data_types):
        fname = paths('epoch', subject=subject, data_type=data_type)
        file_list.append(fname)
        file_list.append(fname[:-4] + '.dat')

    for fname in file_list:
        for fname in file_list:
            download('s3', fname, fname, overwrite)


def upload_all(overwrite=False):
    from scripts.config import paths
    from cloud.utils import upload

    with open(paths('log'), 'r') as f:
        file_list = f.readlines()
    file_list = [line.rstrip(' \n') for line in file_list]

    for fname in file_list:
        for fname in file_list:
            upload('s3', fname, fname, overwrite)
