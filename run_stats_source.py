import numpy as np
from mne import morph_data_precomputed
from mne import spatial_tris_connectivity, grade_to_tris

from config import load, save, bad_mri, subjects_id
from conditions import analyses
from base import stats

connectivity = spatial_tris_connectivity(grade_to_tris(5))
for analysis in analyses:
    chance = analysis['chance']
    stcs = list()
    stcs2 = list()
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

    save([np.array(stcs), connectivity], 'score_source', subject='fsaverage',
         analysis=analysis['name'], overwrite=True, upload=True)

    X = np.array(stcs2) - chance
    p_val = stats(X.transpose(0, 2, 1), connectivity=connectivity, n_jobs=-1)
    save([np.array(stcs), connectivity], 'score_pval', subject='fsaverage',
         analysis=analysis['name'], overwrite=True, upload=True)
