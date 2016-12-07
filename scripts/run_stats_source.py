# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# Licence: BSD 3-clause

"""Run mass-univariate source analyses across subjects from their single-trial
effect sizes.
"""
import numpy as np
from mne import morph_data_precomputed
from mne import spatial_tris_connectivity, grade_to_tris

from config import load, save, bad_mri, subjects_id
from conditions import analyses
from base import stats


def _append_data(analysis):
    """append source scores across subjects"""
    try:
        stcs, connectivity = load('score_source', subject='fsaverage',
                                  analysis=analysis['name'])
    except Exception:
        stcs = list()
        for meg_subject, subject in zip(range(1, 21), subjects_id):
            if subject in bad_mri:
                continue
            # load
            stc, _, _ = load('evoked_source', subject=meg_subject,
                             analysis=analysis['name'])
            morph = load('morph', subject=meg_subject)
            vertices_to = [np.arange(10242)] * 2
            # fix angle error scale
            if 'circAngle' in analysis['name']:
                stc._data /= 2.

            # apply morph
            stc_morph = morph_data_precomputed(subject, 'fsaverage', stc,
                                               vertices_to, morph)
            stcs.append(stc_morph.data)
        stcs = np.array(stcs)
        save([stcs, connectivity], 'score_source', subject='fsaverage',
             analysis=analysis['name'], overwrite=True, upload=True)
    return stcs, connectivity

connectivity = spatial_tris_connectivity(grade_to_tris(5))
for analysis in analyses:
    print(analysis)
    # don't compute if already on S3
    try:
        load('score_pval', subject='fsaverage', analysis=analysis['name'])
        continue
    except Exception:
        pass

    # Retrieve data
    chance = analysis['chance']
    stcs, connectivity = _append_data(analysis)

    # Source Stats
    X = stcs - chance
    p_val = stats(X.transpose(0, 2, 1), connectivity=connectivity, n_jobs=1)

    # Save and upload
    save(p_val, 'score_pval', subject='fsaverage',
         analysis=analysis['name'], overwrite=True, upload=False)
