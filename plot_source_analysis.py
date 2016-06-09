import os
import os.path as op
import numpy as np
from mne import morph_data_precomputed
from matplotlib.colors import LinearSegmentedColormap
from conditions import analyses
from jr.plot import alpha_cmap
from config import load, bad_mri, subjects_id, report, paths, tois
report._setup_provenance()


morphs = dict()
for analysis in analyses:
    stcs = list()
    stcs2 = list()
    for meg_subject, subject in zip(range(1, 21), subjects_id):
        if subject in bad_mri:
            continue
        # load
        stc, _, _ = load('evoked_source', subject=meg_subject,
                         analysis=analysis['name'])
        morphs[subject] = load('morph', subject=meg_subject)
        vertices_to = [np.arange(10242)] * 2
        # fix angle error scale
        if 'circAngle' in analysis['name']:
            stc._data /= 2.

        # apply morph
        stc_morph = morph_data_precomputed(subject, 'fsaverage', stc,
                                           vertices_to, morphs[subject])
        stcs.append(stc_morph.data)
        # same with absolute values
        stc._data = np.abs(stc._data - analysis['chance'])
        stc_morph = morph_data_precomputed(subject, 'fsaverage', stc,
                                           vertices_to, morphs[subject])

        stcs2.append(stc_morph.data + analysis['chance'])

    # plot divergent and absolute score
    stcs = np.array(stcs)
    stcs2 = np.array(stcs2)
    cmaps = (alpha_cmap('RdBu_r'),
             alpha_cmap(LinearSegmentedColormap.from_list(
                        'RdBu', ['w', analysis['color'], 'k']), diverge=False))
    for data, suf, cmap in zip([stcs, stcs2], ['', '_abs'], cmaps):
        stc_morph._data = np.mean(data, axis=0)
        brain = stc_morph.plot(
            subject='fsaverage',
            hemi='split', views=['lat', 'med'], colormap=cmap,
            config_opts=dict(cortex='low_contrast', background='white'))

        # center colormap depending on whether abs value or divergent
        if suf == '':
            mM = np.percentile(abs(stc_morph.data), 99.99)
            clim = (2 * analysis['chance'] - mM, analysis['chance'], mM, False)
        else:
            low, mid, high = np.percentile(abs(stc_morph.data),
                                           [10, 50, 99.99])
            clim = (low, mid, high, False)
        brain.scale_data_colormap(*clim)

        time_handle = brain.texts_dict['time_label']['text']
        time_handle.get('property')['property'].color = (0., 0., 0.)

        # plot each time slice
        image_path = op.join(report.report.data_path, analysis['name'] + suf)
        if not op.exists(image_path):
            os.makedirs(image_path)
        for t in range(0, 1200, 10):
            brain.set_time(t)
            fname = '%s_source_%3i%s.png' % (analysis['name'], t, suf)
            brain.save_image(op.join(image_path, fname))

        # plot average in time region of interest
        stc_morph._data = np.mean(data, axis=0)
        for toi in tois:
            sel = np.where((stc_morph.times >= (toi[0])) &
                           (stc_morph.times <= (toi[1])))[0]
            stc_morph._data[:, sel[0]] = \
                np.mean(np.mean(data, axis=0)[:, sel], axis=1)
            print(sel)
        for toi in tois:
            sel = np.where((stc_morph.times >= (toi[0])) &
                           (stc_morph.times <= (toi[1])))[0]
            brain.set_time(stc_morph.times[sel[0]] * 1e3)
            fname = '%s_%s_%3i.png' % (analysis['name'], suf, toi[0] * 1e3)
            brain.save_image(op.join(report.report.data_path, fname))
        brain.close()

# Plot individual subject to inspect source reconstruction
for meg_subject, subject in zip(range(1, 21), subjects_id):
    if subject in bad_mri:
        continue
    stc, _, _ = load('evoked_source', subject=meg_subject,
                     analysis='target_present')
    brain = stc.plot(hemi='split', subject=subject, views=['lat', 'med'],
                     subjects_dir=paths('freesurfer'))
    brain.set_time(180)
    image_path = op.join(report.report.data_path)
    brain.save_image(op.join(image_path, subject + '.png'))
    brain.close()

# report.save()
