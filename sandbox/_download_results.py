from scripts.config import (
    paths,
    subjects,
    data_types,
    analyses
)

import numpy as np
import os.path as op
from itertools import product
from scripts.transfer_data import download
from cloud.utils import boto_client

for analysis, subject, data_type in product(analyses, subjects, data_types):
    client_fname = paths('decod', subject=subject, analysis=analysis['name'],
                         data_type=data_type)
    server_fname = './data/' + client_fname.split('data/')[-1]
    # print analysis['name'], subject
    if op.exists(client_fname):
        try:
            filesize = np.abs(boto_client.get_key(server_fname).size -
                              op.getsize(client_fname))
            if filesize > 0:
                print('%s %s: %f' % (subject, analysis['name'], filesize))
                download('s3', server_fname, client_fname, True)
        except RuntimeError:
            print('%s %s not online' % (subject, analysis['name']))
    else:
        try:
            download('s3', server_fname, client_fname)
        except RuntimeError:
            print('%s %s not online' % (subject, analysis['name']))
