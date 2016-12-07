"""Plot target-presence AUC time course of each region of interest.

Used to generate Figures 2.a
"""

import numpy as np
import os.path as op
from config import missing_mri, bad_mri, paths
import shutil

from scipy import sparse
from pandas import DataFrame
import matplotlib.pyplot as plt

import mne

from jr.plot import pretty_decod

from base import stats
from conditions import analyses
from config import load, subjects, subjects_id, report

# A priori defined ROI
rois = ['lingual', 'inferiortemporal', 'superiorparietal',
        'rostralmiddlefrontal', 'precentral']

# Copy fsaverage labels for subjects without an mri
for subject in missing_mri:
    to = op.join(paths('freesurfer'), subject, 'label')
    if not op.exists(to):
        shutil.copytree(op.join(paths('freesurfer'), 'fsaverage', 'label'), to)

sel_analyses = ['target_present', 'target_circAngle', 'detect_button_pst']
analyses = [ana for ana in analyses if ana['name'] in sel_analyses]
for analysis in analyses:
    section = ['sources_%s' % analysis['name']]
    chance = analysis['chance']
    # Read data
    evokeds = dict()
    for subject, subject_id in zip(subjects, subjects_id):
        if subject_id in bad_mri:
            continue
        print(subjects)
        labels = mne.read_labels_from_annot(subject_id, parc='aparc',
                                            subjects_dir=paths('freesurfer'))
        inv = load('inv', subject=subject)
        stc, _, _ = load('evoked_source', subject=subject,
                         analysis=analysis['name'])

        # Extract time course
        for label in labels:
            if 'unknown' in label.name:
                continue
            # Init
            if label.name not in evokeds.keys():
                evokeds[label.name] = list()
            evoked = np.squeeze(stc.extract_label_time_course(
                label, inv['src'], mode='mean', verbose=False))
            evokeds[label.name].append(evoked)
    labels = np.array(evokeds.keys())
    data = np.transpose([evokeds.values()], [0, 2, 1, 3])[0]
    times = stc.times

    # Stats on region of interest across subjects
    scores = list()
    for roi in rois:
        idx = np.where([roi in this_roi for this_roi in labels])[0]
        scores.append(np.mean(data[:, idx, :], axis=1))
    p_values = stats(np.transpose(scores, [1, 2, 0]) - chance, n_jobs=-1)

    fig, axes = plt.subplots(len(rois), 1, sharex=True, sharey=True,
                             figsize=[10, 40])
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0., 1., len(rois)))
    ylim = [2, -2]

    # Plot
    details_toi, details_roi = list(), list()
    for ii, (score, p_val, ax, color, roi) in enumerate(zip(
            scores, p_values.T, axes, colors, rois)):
        pretty_decod(score, sig=p_val < .05, times=times-np.min(times),
                     color=color, ax=ax, fill=True, chance=chance)

        xlim = ax.get_xlim()
        ylim[0] = min([ylim[0], ax.get_ylim()[0]])
        ylim[1] = max([ylim[1], ax.get_ylim()[1]])
        ax.set_ylim(ylim)
        ax.set_yticks(ylim)
        ax.set_yticklabels(['%.3f' % y for y in ylim])
        ax.set_xticklabels([int(x) if x in np.linspace(0., 1200., 11) else ''
                            for x in np.round(1e3 * ax.get_xticks())])
        if ax != axes[-1]:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        elif ax == axes[0]:
            ax.text(0, ylim[1], 'Target',  backgroundcolor='w', ha='center',
                    va='top')
            ax.text(.800, ylim[1], 'Probe', backgroundcolor='w', ha='center',
                    va='top')
            ax.set_yticks([chance, ylim[1]])
            ax.set_yticklabels(['', '%.2f' % ylim[1]])
            if analysis['typ'] == 'regress':
                ax.set_ylabel('R', labelpad=-15)
            elif analysis['typ'] == 'categorize':
                ax.set_ylabel('AUC', labelpad=-15)
            else:
                ax.set_ylabel('rad.', labelpad=-15)
            txt = ax.text(xlim[0] + .5 * np.ptp(xlim),
                          ylim[0] + .75 * np.ptp(ylim),
                          analysis['title'], color=[.2, .2, .2],
                          ha='center', weight='bold')

        # Print some mean and SEM on a priori regions to report in the
        # manuscript
        tois = [.109, .169, .500, 1.300]
        for toi in tois:
            toi_ = np.where(times > toi)[0][0]
            details_toi.append(dict(
                roi=roi,
                toi=toi,
                mean=np.mean(score[:, toi_]),  # FIXME contrast is inverted
                inv_mean=1.-np.mean(score[:, toi_]),  # XXX FIXME
                sem=np.std(score[:, toi_]) / np.sqrt(len(score))))

        # Print some sig cluster details
        sig = p_val < .05
        diff = np.diff(1. * sig)
        starts, stops = np.where(diff > 0)[0], np.where(diff < 0)[0]
        for start, stop in zip(starts, stops):
            score_toi = np.mean(score[:, start:stop], axis=1)
            details_roi.append(dict(
                roi=roi,
                p_val=p_val[stop],
                start=times[start],
                stop=times[stop],
                mean=np.mean(score_toi),  # FIXME contrast is inverted
                inv_mean=1.-np.mean(score_toi),  # XXX FIXME
                sem=np.std(score_toi) / np.sqrt(len(score))))

    report.add_figs_to_section([fig], section, 'source')
    report.add_htmls_to_section(DataFrame(details_roi).to_html(),
                                section, 'roi')
    report.add_htmls_to_section(DataFrame(details_toi).to_html(),
                                section, 'toi')

    # Print significant clusters on all regions
    connectivity = sparse.csr_matrix(np.eye((data.shape[1])))
    p_vals = stats(data.transpose(0, 2, 1) - chance, None, n_jobs=-1)
    sig_labels = DataFrame(labels[np.where(np.sum(p_vals < .05, axis=0))[0]])
    report.add_htmls_to_section(sig_labels.to_html(), section, 'sig_cluster')

    sig_labels = DataFrame(labels[np.where(np.sum(p_vals < .05, axis=0))[0]])
    report.add_htmls_to_section(sig_labels.to_html(), section, 'sig_cluster')

report.save()
