import sys
sys.path.insert(0, './')

import matplotlib.pyplot as plt

import mne
from meeg_preprocessing.utils import setup_provenance
from orientations.utils import load_epochs_events

from scripts.config import (
    paths,
    subjects,
    cov_method,
    cov_reject,
    open_browser,
    missing_mri,
    chan_types
)

# Prepare logs
mne.set_log_level('INFO')
report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir=paths('report'))

subjects = [s for s in subjects if s not in missing_mri]

fig_all, ax_all = plt.subplots(1, 2)
for subject in subjects:
    logger.info(subject)
    epochs, events = load_epochs_events(subject, paths, data_type='erf')

    # epochs.drop_bad_epochs(cov_reject) # XXX uncomment once re preprocessed
    evoked = epochs.average()  # keep for whitening plot
    epochs.pick_types(meg=True)

    # Compute the covariance on baseline
    covs = mne.compute_covariance(
        epochs, tmin=None, tmax=0, method=cov_method,
        return_estimators=True)

    for c, cov in enumerate(covs):
        # Plot
        fig_cov, fig_svd = mne.viz.plot_cov(cov, epochs.info, colorbar=True,
                                            proj=True, show=False)

        # plot whitening of evoked response
        fig_white = evoked.plot_white(cov, show=False)

        # Plot across subjects
        times, gfp = fig_white.get_children()[2].get_children()[2].get_data()
        ax_all[c].plot(times, gfp, show=False)

        report.add_figs_to_section(
            [fig_cov, fig_svd, fig_white],
            [string.format(subject, cov['method']) for string in
             ['{}: COV ({})', '{}: SVD ({})', '{}: WHITE ({})']], subject)
    # Save best covariance method
    covs[0].save(paths('covariance', subject, log=True))

report.add_figs_to_section(fig_all, 'WHITE', 'all')


logger.info('Finished with no error')
report.save(open_browser=open_browser)
