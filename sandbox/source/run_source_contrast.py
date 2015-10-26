import sys
sys.path.insert(0, './')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colorbar, colors, gridspec
from scipy.ndimage import zoom

import mne
from mne.minimum_norm import apply_inverse, read_inverse_operator

from meeg_preprocessing.utils import setup_provenance

from scripts.config import (
    paths,
    subjects,
    missing_mri,
    epochs_params,
    open_browser,
    contrasts
)


mne.set_log_level('INFO')
report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir=paths('report'))

subjects = [s for s in subjects if s not in missing_mri]

for subject in subjects:
    logger.info(subject)
    for epoch_params in epochs_params:
        logger.info(epoch_params['name'])

        # Read inverse operator
        inverse_operator = read_inverse_operator(paths('inverse', subject))

        # Apply each contrast
        ave_fname = paths('evoked', subject, epoch=epoch_params['name'])

        for contrast in contrasts:
            logger.info(contrast['name'])
            srcs = list()
            key = contrast['include'].keys()[0]
            for v in contrast['include'][key]:
                evoked = mne.read_evokeds(ave_fname,
                                          condition=contrast['name']+str(v))
                # Apply inverse operator
                snr = 3.0  # XXX pass to config
                lambda2 = 1.0 / snr ** 2
                src = apply_inverse(evoked, inverse_operator, lambda2, 'dSPM')
                srcs.append(src)

            # Contrast
            src = srcs[0] - srcs[-1]

            # Plot
            cmap = plt.get_cmap('RdBu_r')
            cmap._init()
            cmap = cmap._lut[:cmap.N, :] * 255
            logit = lambda x: 2 * (1 / (1 + np.exp(-3. * x)) - .5)
            alpha = np.linspace(-1.0, 1.0, cmap.shape[0])
            cmap[:, -1] = [255 * np.abs(logit(x)) for x in alpha]

            # cmap = mne_analyze_colormap()
            brain = src.plot(
                subject, subjects_dir=paths('freesurfer'),
                surface='inflated', hemi='split', colormap=cmap,
                config_opts=dict(height=300., width=600, offscreen=True,
                                 cortex='low_contrast', background='white'))

            mM = np.percentile(abs(src.data), 99.5)
            brain.scale_data_colormap(-mM, 0, mM, False)

            # XXX Pass to config
            if epoch_params['name'] == 'stim_lock':
                plot_times = np.linspace(0, 600, 12)
            else:
                plot_times = np.linspace(-500, 100, 12)

            # XXX need clean up
            imgs = []
            del imgs
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
                if 'imgs' not in locals():
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

            report.add_figs_to_section(fig, subject + ': ' +
                                       epoch_params['name'] + ': ' +
                                       contrast['name'], subject)

logger.info('Finished with no error')
report.save(open_browser=open_browser)
