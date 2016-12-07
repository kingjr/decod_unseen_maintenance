"""Plot non-thresholded whole-brain source analyses.

Used to generate Figures 2.b and S4.
"""

import os
import os.path as op
import numpy as np
from mne import morph_data_precomputed
from matplotlib.colors import LinearSegmentedColormap
from conditions import analyses
from jr.plot import alpha_cmap
from config import load, subjects_id, report, tois
report._setup_provenance()

# Load a first source space to setup plotting
stc, _, _ = load('evoked_source', subject=1, analysis='target_present')
morph = load('morph', subject=1)
vertices_to = [np.arange(10242)] * 2
stc_morph = morph_data_precomputed(subjects_id[0], 'fsaverage', stc,
                                   vertices_to, morph)
# Loop across analyses (target presence, angle etc)
for analysis in analyses:
    stcs, connectivity = load('score_source', analysis=analysis['name'])
    p_val = load('score_pval', analysis=analysis['name'])

    # Morph pval in log space
    stc_morph._data[:, :] = np.log(p_val.T)
    stc_pval = stc_morph.morph('fsaverage', grade=None)
    sig = np.exp(stc_pval._data) < .05
    del stc_pval, p_val

    # Get absolute score value for plotting
    chance = analysis['chance']
    stc2 = np.mean(np.abs(stcs - chance) + chance, axis=0)
    stc = np.mean(stcs, axis=0)
    del stcs, connectivity

    # Plot effect size in each source
    cmaps = (alpha_cmap('RdBu_r'),
             alpha_cmap(LinearSegmentedColormap.from_list(
                        'RdBu', ['k', analysis['color']]), diverge=False))
    for data, suf, cmap in zip([stc, stc2], ['', '_abs'], cmaps):
        stc_morph._data = data
        brain = stc_morph.plot(
            subject='fsaverage',
            hemi='split', views=['lat', 'med'], colormap=cmap,
            config_opts=dict(cortex='low_contrast', background='white'))

        # center colormap depending on whether abs value or divergent
        if suf == '':
            mM = np.percentile(abs(stc_morph.data), 99.99)
            clim = (2 * chance - mM, chance, mM, False)
        else:
            low, mid, high = np.percentile(abs(stc_morph.data),
                                           [10, 50, 99.99])
            high = np.max(stc_morph.data)
            clim = (low, mid, high, False)
        brain.scale_data_colormap(*clim)

        time_handle = brain.texts_dict['time_label']['text']
        time_handle.get('property')['property'].color = (0., 0., 0.)

        # plot each time slice
        image_path = op.join(report.report.data_path, analysis['name'] + suf)
        if not op.exists(image_path):
            os.makedirs(image_path)
        contours = list()
        for t in range(0, 1200, 10):
            for ii in contours:
                ii['surface'].remove()
            brain.set_time(t)
            if False:  # plot significant clusters
                overlay = 1. * sig[:, np.where(brain._times >= t)[0][0]]
                left = brain.add_contour_overlay(
                    overlay[len(overlay)//2:], min=0., max=1.,
                    hemi='lh',
                    n_contours=2, colorbar=False, colormap='hot')
                contours = brain.contour_list
                right = brain.add_contour_overlay(
                    overlay[:len(overlay)//2], min=0., max=1.,
                    hemi='rh',
                    n_contours=2, colorbar=False, colormap='hot')
                contours += brain.contour_list
            fname = '%s_source_%3i%s.png' % (analysis['name'], t, suf)
            brain.save_image(op.join(image_path, fname))

        # Plot average over time region of interest
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

# report.save()
