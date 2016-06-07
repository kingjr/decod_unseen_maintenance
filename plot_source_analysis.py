import os
import numpy as np
from mne import compute_morph_matrix, morph_data_precomputed
from conditions import analyses
from config import load, missing_mri, subjects_id, paths, report
report._setup_provenance()
sel_analyses = ['target_present', 'target_circAngle', 'probe_circAngle']
analyses = [ana for ana in analyses if ana['name'] in sel_analyses]

morphs = dict()
for analysis in analyses:
    stcs = list()
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject in missing_mri:
            continue
        stc, _, _ = load('evoked_source', subject=meg_subject,
                         analysis=analysis['name'])
        # compute morph matrix
        if subject not in morphs:
            vertices_to = [np.arange(10242)] * 2
            morphs[subject] = compute_morph_matrix(
                subject, 'fsaverage', stc.vertices,
                vertices_to=vertices_to,
                subjects_dir=paths('freesurfer'))

        # apply morph
        stc_morph = morph_data_precomputed(subject, 'fsaverage', stc,
                                           vertices_to, morphs[subject])
        stcs.append(stc_morph.data)
    stcs = np.array(stcs) - analysis['chance']
    stc_morph._data = np.mean(stcs, axis=0)
    brain = stc_morph.plot(hemi='split', subject='fsaverage',
                           views=['lat', 'med'])
    # plot
    image_path = os.path.join(report.report.data_path, analysis['name'])
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    for t in range(0, 1100, 10):
        brain.set_time(t)
        fname = '%s_source_%3i.png' % (analysis['name'], t)
        brain.save_image(os.path.join(image_path, fname))
        print(t)
    brain.close()

    # plot squared values
    stc_morph._data = np.mean(stcs ** 2, axis=0)
    brain = stc_morph.plot(hemi='split', subject='fsaverage',
                           views=['lat', 'med'])
    # plot
    image_path = os.path.join(report.report.data_path,
                              analysis['name'] + '_square')
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    for t in range(0, 1100, 10):
        brain.set_time(t)
        fname = '%s_source_%3i_square.png' % (analysis['name'], t)
        brain.save_image(os.path.join(image_path, fname))
        print(t)
    brain.close()
# report.save()
