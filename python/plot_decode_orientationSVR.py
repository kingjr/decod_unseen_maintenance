"""This script is used to loop across subjects to retrieve target
                and probe SVRs and compute the predicted reconstructed angle"""

import os.path as op
import pickle
from config import (
        subjects,
        data_path,
        clf_types)
contrasts = ('orientation_target','orientation_probe')

for contrast in contrasts:
    cond_name_sin = contast + '_sin'
    cond_name_cos = contast + '_cos'
    for subject in subjects:

        # Define path to retrieve cosine classifier
        path_x = op.join(data_path, subject, 'mvpas',
            '{}-decod_{}.pickle'.format(subject, cond_name_cos))

        # Define path to retrieve sine classifier
        path_y = op.join(data_path, subject, 'mvpas',
            '{}-decod_{}.pickle'.format(subject, cond_name_sin))

        predAngle, radius = recombine_svr_prediction(path_x,path_y)

        error = -np.log(predAngle - gat.train)
