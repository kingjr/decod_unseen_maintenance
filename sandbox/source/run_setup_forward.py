import sys
sys.path.insert(0, './')
import mne
from meeg_preprocessing.utils import setup_provenance
import matplotlib.pyplot as plt

from scripts.config import (
    paths,
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
    # setup paths
    raw_fname = paths('rawfilt', subject, run=1)
    tsss_fname = paths('tsss', subject)
    bem_fname = paths('bem', subject)
    src_fname = paths('oct', subject)

    # setup source space
    mne.setup_source_space(subject, subjects_dir=paths('freesurfer'),
                           overwrite=True, n_jobs=6)

    # forward solution
    # XXX if too many jobs, conflict between MKL and joblib
    # XXX eeg and meg should be inherited form config.ch_types
    raw = mne.io.Raw(raw_fname, preload=False)
    fwd = mne.make_forward_solution(raw_fname, tsss_fname, src_fname,
                                    bem_fname, fname=None, meg=True, eeg=False,
                                    mindist=5.0, n_jobs=2, overwrite=True)

    # convert to surface orientation for better visualization
    fwd = mne.convert_forward_solution(fwd, surf_ori=True)
    leadfield = fwd['sol']['data']

    logger.info("Leadfield size : %d x %d" % leadfield.shape)

    grad_map = mne.sensitivity_map(fwd, ch_type='grad', mode='fixed')
    mag_map = mne.sensitivity_map(fwd, ch_type='mag', mode='fixed')

    # SAVE
    mne.write_forward_solution(paths('forward', subject, log=True), fwd,
                               overwrite=True)

    # PLOT
    picks = mne.pick_types(fwd['info'], meg=True, eeg=False)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle('Lead field matrix (500 dipoles only)', fontsize=14)
    im = ax.imshow(leadfield[picks, :500], origin='lower', aspect='auto',
                   cmap='RdBu_r')
    ax.set_xlabel('sources')
    ax.set_ylabel('sensors')
    plt.colorbar(im, ax=ax, cmap='RdBu_r')
    report.add_figs_to_section(fig, '{}: Lead field matrix'.format(subject),
                               subject)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.hist([grad_map.data.ravel(), mag_map.data.ravel()], bins=20,
            label=['Gradiometers', 'Magnetometers'], color=['c', 'b'])
    ax.legend()
    ax.set_title('Normal orientation sensitivity')
    ax.set_xlabel('sensitivity')
    ax.set_ylabel('count')
    report.add_figs_to_section(
        fig, '{}: Normal orientation sensitivity'.format(subject), subject)

logger.info('Finished with no error')
report.save(open_browser=open_browser)
