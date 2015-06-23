def download_depencies(overwrite=False):
    from itertools import product
    from cloud.utils import download
    # XXX This will have to be specified for later scripts
    from scripts.config import paths, subjects, data_types

    file_list = list()
    for subject, data_type in product(subjects, data_types):
        file_list.append(paths('epoch', subject=subject, data_type=data_type))

    for fname in file_list:
        for fname in file_list:
            download('s3', fname, fname, overwrite)


def upload_results(overwrite=False):
    from scripts.config import paths
    from cloud.utils import upload

    with open(paths('log'), 'r') as f:
        file_list = f.readlines()
    file_list = [line.rstrip(' \n') for line in file_list]

    for fname in file_list:
        for fname in file_list:
            upload('s3', fname, fname, overwrite)
