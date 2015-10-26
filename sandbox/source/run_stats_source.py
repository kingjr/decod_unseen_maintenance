import sys
sys.path.insert(0, './')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colorbar, colors, gridspec
import mayavi
from scipy import stats as stats
from scipy.ndimage import zoom

import mne
from mne import (spatial_tris_connectivity, compute_morph_matrix,
                 grade_to_tris, SourceEstimate)
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)

from meeg_preprocessing.utils import setup_provenance

from scripts.config import (
    paths,
    subjects,
    missing_mri,
    epochs_params,
    contrasts,
    open_browser,
)


mne.set_log_level('INFO')
report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir=paths('report'))

subjects = [s for s in subjects if s not in missing_mri]  # XXX

# define connectivity
connectivity = spatial_tris_connectivity(grade_to_tris(5))

# inverse parameters
snr = 3.0
lambda2 = 1.0 / snr ** 2


# Prepare sources
inverse_operator = dict()
sample_vertices = dict()
morph_mat = dict()
smooth = 20
fsave_vertices = [np.arange(10242), np.arange(10242)]
for subject in subjects:
    # Read all inverse operators
    inv = read_inverse_operator(paths('inv', subject))
    inverse_operator[subject] = inv
    sample_vertices[subject] = [s['vertno'] for s in inv['src']]

    # Prepare morphing matrix : # XXX should be done externally?
    morph_mat[subject] = compute_morph_matrix(subject, 'fsaverage',
                                              sample_vertices[subject],
                                              fsave_vertices, smooth,
                                              paths('freesurfer'))

for epoch_params in epochs_params:
    for contrast in contrasts:
        logger.info([epoch_params['name'], contrast['name']])
        # Concatenate data across subjects
        X = list()
        for subject in subjects:
            # Read evoked data for each condition
            ave_fname = paths('evoked', subject, epoch=epoch_params['name'])
            srcs = list()
            key = contrast['include'].keys()[0]
            for v in contrast['include'][key]:
                evoked = mne.read_evokeds(ave_fname,
                                          condition=contrast['name']+str(v))

                # Apply inverse operator
                src = apply_inverse(evoked, inverse_operator[subject],
                                    lambda2, 'dSPM')
                src = src.morph_precomputed('fsaverage', fsave_vertices,
                                            morph_mat[subject], subject)
                srcs.append(src)

            # Contrast
            src = srcs[0] - srcs[-1]

            # Append across subjects
            X.append(src._data)
        # break

        # Stats
        threshold = -stats.distributions.t.ppf(.05 / 2., len(subjects) - 1)
        threshold = dict(start=10., step=5.)  # XXX to be  changed
        T_obs, clusters, cluster_p_values, H0 = clu = \
            spatio_temporal_cluster_1samp_test(np.transpose(X, (0, 2, 1)),
                                               connectivity=connectivity,
                                               n_jobs=-1, tail=0,
                                               threshold=threshold)
        # significant times
        good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
        times_inds = [int(np.mean([clusters[c][0]])) for c in good_cluster_inds]
        plot_times = [int(src.times[t] * 1000) for t in times_inds]

        # pass foor loop if nothing sig.
        if not good_cluster_inds:
            continue

        # only plot sig data
        x = np.mean(X, axis=0)
        mask = np.zeros(x.shape)
        for c in good_cluster_inds:
            for t, v in zip(clusters[c][0], clusters[c][1]):
                mask[v, t] = x[v, t]

        src = SourceEstimate(x, fsave_vertices, tmin=src.tmin,
                             tstep=src.tstep, subject='fsaverage')

        # brain = src.plot('fsaverage',
        #                   subjects_dir=paths('freesurfer'),
        #                   surface='inflated', colormap=cmap)
        # brain.add_contour_overlay(mask[:,0], 0, 1, 1)

        # cmap
        cmap = plt.get_cmap('RdBu_r')
        cmap._init()
        cmap = cmap._lut[:cmap.N, :] * 255
        logit = lambda x: 2 * (1 / (1 + np.exp(-3. * x)) - .5)
        cmap[:, -1] = [255 * np.abs(logit(i))
                       for i in np.linspace(-1.0, 1.0, cmap.shape[0])]

        brain = src.plot('fsaverage',
                         subjects_dir=paths('freesurfer'),
                         surface='inflated', hemi='split', colormap=cmap,
                         config_opts=dict(height=300., width=600,
                                          offscreen=True,
                                          cortex='low_contrast',
                                          background='white'))

        mM = np.percentile(abs(src.data), 99.5)
        brain.scale_data_colormap(-mM, 0, mM, False)

        # XXX pass to config
        if epoch_params['name'] == 'stim_lock':
            plot_times = np.linspace(0, 600, 12)
        else:
            plot_times = np.linspace(-500, 100, 12)

        # XXX externalize
        if 'imgs' in locals():
            del imgs  # XXX bad syntax scheme
        for t in plot_times:
            print(t)
            img = []
            for hemi in range(2):
                brain.set_time(t)
                x = brain.texts_dict['time_label']['text']
                x.set(text=x.get('text')['text'][5:-6])
                x.set(width=0.1 * len(x.get('text')['text']))
                x.get('property')['property'].color = (0., 0., 0.)
                img.append(np.vstack(brain.save_imageset(
                    None, views=['lateral', 'medial'], colorbar=None,
                    col=hemi)))
            img = np.vstack(img)
            img = np.array(
                [zoom(c, 600. / img.shape[0])
                 for c in img.transpose((2, 0, 1))]).transpose((1, 2, 0))
            if not 'imgs' in locals():
                imgs = img
            else:
                imgs = np.concatenate((imgs, img), axis=1)

        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.)
        ax0 = plt.subplot(gs[0])
        im = ax0.imshow(imgs)
        ax0.axis('off')
        # colorbar
        ax1 = plt.subplot(gs[1])
        cb1 = colorbar.ColorbarBase(ax1, cmap='RdBu_r',
                                    norm=colors.Normalize(-mM, mM))

        report.add_figs_to_section(
            fig, epoch_params['name'] + ': ' + contrast['name'], subject)

        # plot each significant cluster
logger.info('Finished with no error')
report.save(open_browser=open_browser)
