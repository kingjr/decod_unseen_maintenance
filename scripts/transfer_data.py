import sys
sys.path.insert(0, './')
import matplotlib
matplotlib.use('Agg')

from cloud.utils import download, upload


def download_all(overwrite=False):
    from itertools import product
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

    with open(paths('log'), 'r') as f:
        file_list = f.readlines()
    file_list = [line.rstrip(' \n') for line in file_list]

    for fname in file_list:
        for fname in file_list:
            upload('s3', fname, fname, overwrite)


def upload_report(report):
    import os
    import os.path as op
    base_path = op.dirname(__file__)
    for root, dirnames, filenames in os.walk(report.data_path):
        for filename in filenames:
            fname = os.path.join(root, filename)
            upload('dropbox', fname,
                   op.join('reports/', base_path, fname), True)
