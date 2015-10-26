"""Run control analyses related to the present vs absent decoding: does it
covary with/is modulated by target contrast and target visibility?"""
import pickle
import numpy as np
from jr.gat import get_diagonal_ypred, subscore
from jr.stats import repeated_spearman
from jr.utils import align_on_diag
from scripts.config import paths, subjects, tois
from scripts.base import stats

# Gather data
n_times = 154  # XXX the number of time samples in an epoch should be automatic
contrast_list = [.5, .75, 1.]
pas_list = np.arange(4.)

# initialize results
results = dict(
    data=np.nan * np.zeros((len(subjects), n_times, 4, 3)),
    R_vis=np.nan * np.zeros((len(subjects), n_times)),
    R_contrast=np.nan * np.zeros((len(subjects), n_times)),
    AUC_pas=np.nan * np.zeros((4, len(subjects), n_times)),
    AUC_pas_duration=np.nan * np.zeros((len(tois), len(pas_list),
                                        len(subjects), n_times))
)

for s, subject in enumerate(subjects):
    print s

    # Load data
    fname = paths('decod', subject=subject, analysis='target_present')
    with open(fname, 'rb') as f:
        gat, _, events_sel, events = pickle.load(f)
    times = gat.train_times_['times']
    y_pred = np.transpose(np.squeeze(get_diagonal_ypred(gat)))
    y_error = y_pred - np.tile(gat.y_true_, [n_times, 1]).T
    subevents = events.iloc[events_sel].reset_index()

    # contrast effect: is there a correlation between the contrast of the
    # target and our ability to decode its presence?
    r = list()
    for ii, pas in enumerate(pas_list):
        key = 'detect_button == %s and target_present == True' % pas
        subsel = subevents.query(key).index
        if len(subsel) > 0:
            r.append(repeated_spearman(y_error[subsel, :],
                     np.array(subevents.target_contrast)[subsel]))
    results['R_contrast'][s, :] = np.nanmean(r, axis=0)

    # visibility effect: is there a correlation between the visibility of the
    # target and our ability to decode its presence
    r = list()
    for ii, contrast in enumerate(contrast_list):
        key = 'target_contrast == %s' % contrast
        subsel = subevents.query(key).index
        if len(subsel) > 0:
            r.append(repeated_spearman(y_error[subsel, :],
                     np.array(subevents.detect_button)[subsel]))
    results['R_vis'][s, :] = np.nanmean(r, axis=0)

    # mean decoding seen unseen: Can we decode the presence of seen and unseen
    # target?. We'll get the AUC from target present (seen or unseen subset)
    # versus absent target (all vis)
    for ii, pas in enumerate(pas_list):
        key = 'detect_button == %s or target_present == False' % pas
        subsel = subevents.query(key).index
        if len(subsel) == 0:
            continue
        score = subscore(gat, subsel)
        results['AUC_pas'][ii, s, :] = np.diagonal(score)
        # duration effect: are early and late estimators differentially able
        # to generalizae the decoding of the target's presence over time?
        score_align = align_on_diag(score)
        for jj, toi in enumerate(tois):
            results_pas = list()
            toi_ = np.where((times >= toi[0]) & (times <= toi[1]))[0]
            results['AUC_pas_duration'][jj, ii, s, :] = np.mean(
                score_align[toi_, :], axis=0)

results['times'] = times
results['p_vis'] = stats(results['R_vis'][:, :, None])
results['p_contrast'] = stats(results['R_contrast'][:, :, None])

fname = paths('score', analysis='present_anova')
with open(fname, 'wb') as f:
    pickle.dump(results, f)
