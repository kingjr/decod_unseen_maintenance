import sys
sys.path.insert(0, './')
import matplotlib.pyplot as plt

from meeg_preprocessing.utils import setup_provenance

import mne
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator)

from scripts.config import (
    paths,
    subjects,
    open_browser,
    missing_mri,
)

mne.set_log_level('INFO')
report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir=paths('report'))


subjects = [s for s in subjects if s not in missing_mri]

for subject in subjects:
    logger.info(subject)
    # setup paths
    epo_fname = paths('epoch', subject, epoch='stim_lock')
    fwd_fname = paths('forward', subject)
    cov_fname = paths('covariance', subject)
    src_fname = paths('source', subject)

    # XXX loop across channel types
    # Load data
    epochs = mne.read_epochs(epo_fname)
    evoked = epochs.average()  # XXX evoked?
    del epochs
    noise_cov = mne.read_cov(cov_fname)

    forward_meg = mne.read_forward_solution(fwd_fname, surf_ori=True)

    # Restrict forward solution as necessary for MEG
    forward_meg = mne.pick_types_forward(forward_meg, meg=True, eeg=False)

    # make inverse operators
    info = evoked.info
    inverse_operator_meg = make_inverse_operator(info, forward_meg, noise_cov,
                                                 loose=0.2, depth=0.8)
    # Save
    inv_fname = paths('inverse', subject, log=True)
    write_inverse_operator(inv_fname, inverse_operator_meg)

    # Plot report
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stc = apply_inverse(evoked, inverse_operator_meg, lambda2, "dSPM",
                        pick_ori=None)

    # View activation time-series
    fig = plt.figure(figsize=(8, 6))
    plt.plot(1e3 * stc.times, stc.data[::150, :].T)
    plt.ylabel('MEGdSPM value')
    plt.xlabel('Time (ms)')
    report.add_figs_to_section(fig, '{}: Evoked sources'.format(subject),
                               subject)

logger.log('Finished with no error')
report.save(open_browser=open_browser)
