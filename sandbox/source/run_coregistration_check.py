import sys
sys.path.insert(0, './')
import numpy as np
import matplotlib.pyplot as plt
import mayavi

from meeg_preprocessing.utils import setup_provenance

import mne
from mne.viz import plot_trans

from scripts.config import (
    paths,
    runs,
    subjects,
    open_browser,
    missing_mri
)

mne.set_log_level('INFO')
report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir=paths('report'))

subjects = [s for s in subjects if s not in missing_mri]

for subject in subjects:
    logger.info(subject)
    # Plot HPI trans matrix for each run
    fig = plt.figure()
    title = subject
    for r in runs:
        logger.info('run %s' % r)
        raw = mne.io.Raw(paths('raw', subject, run=r), preload=False)
        if r == runs[0]:
            trans1 = raw.info['dev_head_t']['trans']
        trans = raw.info['dev_head_t']['trans']
        plt.plot(trans)
        if not np.all(trans == trans1):
            logger.warning('%s has different head position across runs!'
                           % subject)
            title = subject + ' /!\ NEED HEAD REALIGNEMENT'

    report.add_figs_to_section(fig, 'HPI: ' + title, subject)

    # Plot head in helmet
    scene = plot_trans(raw.info,
                       trans=paths('tsss', subject),
                       subject=subject,
                       subjects_dir=paths('freesurfer'),
                       source='head')

    # Check if digitizers:
    # XXX WIP
    # eeg_locs = [l['eeg_loc'][:, 0] for l in raw.info['chs']
    #             if l['eeg_loc'] is not None]
    # trans_fname = op.join(data_path, 'MEG', subject, mri_fname_tmp.format(1))
    # trans = mne.read_trans(trans_fname)
    # locs = np.array([l['loc'][:3] for l in raw.info['chs']
    #                  if l['loc'] is not None])
    # locs = mne.transforms.apply_trans(trans['trans'], locs)
    #
    # if not eeg_locs:
    #     print('/!\ %s has no digitized points' % subject)
    # mayavi.mlab.points3d(locs[:, 0], locs[:, 1], locs[:, 2],
    #                      color=(0.0, 1.0, 0.0), scale_factor=0.005)

    views = np.array(([90, 90], [0, 90], [0, -90], [0, 0]))
    fig, ax = plt.subplots(2, 2)
    ax = np.reshape(ax, 4, 1)
    for i, v in enumerate(views):
        mayavi.mlab.view(v[0], v[1], figure=scene)
        img = mayavi.mlab.screenshot()
        ax[i].imshow(img)
        ax[i].axis('off')

    report.add_figs_to_section(fig, '3D: ' + title, subject)
    mayavi.mlab.close(scene=scene)

logger.info('Finished with no error')
report.save(open_browser=open_browser)
