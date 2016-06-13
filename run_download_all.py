import os
from conditions import analyses
from config import client, subjects, paths
# for subject in range(1, 21):
#     for block in range(1, 6):
#         print(subject, block)
#         fname = paths('sss', subject=subject, block=block)
#         client.download(fname.split('sss/')[-1], fname)
#         fname = paths('behavior', subject=subject)
#         client.download(fname.split('behavior/')[-1], fname)

for subject in subjects:
    for analysis in analyses:
        for typ in ['score_tfr']:
            fname = paths(typ, subject=subject, analysis=analysis['name'])
            to_path = fname.split('/')
            to_path = '/'.join(to_path[:-1] + ['_old_' + to_path[-1]])
            if os.path.exists(fname) and not os.path.exists(to_path):
                os.rename(fname, to_path)
            client.download(fname)
