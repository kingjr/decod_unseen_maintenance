import sys
sys.path.insert(0, './')
import os.path as op

from scripts.config import (
    paths,
    subjects
)

for subject in subjects:
    fname = op.join(paths('freesurfer'), subject, 'scripts', 'recon-all.log')
    if op.isfile(fname):
        with open(fname, 'rb') as fh:
            fh.seek(-1024, 2)
            last = fh.readlines()[-1].decode()
        print('{}: ok'.format(subject))
    else:
        print('{}: missing'.format(subject))
