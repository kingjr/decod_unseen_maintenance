Brain mechanisms underlying the brief maintenance of seen and unseen sensory information
========================================================================================

Jean-Rémi King, Niccolo Pescetelli & Stanislas Dehaene, Neuron 2016


Abstract
========

Recent evidence of unconscious working memory challenges the notion that only visible stimuli can be actively maintained over time. In the present study, we investigated the neural dynamics underlying the maintenance of variably visible stimuli using magnetoencephalography. Subjects had to detect and mentally maintain the orientation of a masked grating. We show that the stimulus is fully encoded in early brain activity independently of visibility reports. However, the presence and orientation of the target are actively maintained throughout the brief retention period, even when the stimulus is reported as unseen. Source and decoding analyses revealed that perceptual maintenance recruits a hierarchical network spanning the early visual, temporal, parietal and frontal cortices. Importantly, the representations coded in the late processing stages of this network specifically predict visibility reports. These unexpected results challenge several theories of consciousness and suggest that invisible information can be briefly maintained within the higher processing stages of visual perception.


Data
====

The data will soon be made publicly accessible. Stay tuned.

Notebooks
=========

A series of tutorials have been made available in `notebook/` to clarify the methods, and strip them from the data handling.

- `notebook/method_decoding.ipynb` explains the general procedure used to perform decoding with MEG data.
- `notebook/method_model_types.ipynb` explains how categorical, ordinal and circular models can be fitted and scored.
- `notebook/method_statistics.ipynb` explains how the statistics are performed in the manuscript.
- `notebook/results_summary.ipynb` gives a preview of some of the results to allow user to replicate our analyses, or go further by looking at individual subjects, test other statistical methods etc.

Also consider looking at [MNE's examples and tutorials](http://mne-tools.github.io/stable/auto_tutorials/plot_sensors_decoding.html) showing how the `TimeDecoding` and `GeneralizationAcrossTime` classes can be used on your own data.


Scripts
=======

Overall, the current scripts remain designed for research purposes, and could therefore be improved and clarified. If you judge that some codes would benefit from specific clarifications do not hesitate to contact us.

The scripts are generally decomposed in terms of general functions (base), actual analyses (decoding, cluster analyses), and report (plotting, tables, quick stats).

#### Config files
- 'scripts/base.py' # where all generic functions are defined
- 'scripts/conditions.py' # where the analyses, multivariate estimators, scorers etc are defined
- 'scripts/config.py'  # where the paths and filenames are setup

#### Prepare data
- 'scripts/run_plot_behavior.py'  # plot visibility, accuracy and d-prime
- 'scripts/run_preprocessing.py'  # filter and epochs raw signals

#### Mass univariate sensor analyses
- 'scripts/run_sensor_analysis.py'  # analyze sensor ERF within each subject
- 'scripts/run_stats_sensors.py'  # run second-level (between subjects) stats
- 'scripts/plot_stats_sensors.py'  # plot average topographies across subjects

#### Mass univariate source analyses
- 'scripts/run_preprocess_source.py'
- 'scripts/run_source_analysis.py'
- 'scripts/run_stats_source.py'
- 'scripts/plot_anatomy_roi.py'
- 'scripts/plot_source_time_course.py'  # time course of source of regions of interest
- 'scripts/plot_source_analysis.py'  # all-brain sources

#### Decoding
- 'scripts/run_decoding.py'  # decoding evoked related fields
- 'scripts/run_decoding_timefreq.py'  # decoding of induced related fields
- 'scripts/run_stats_decodings.py'  # second-level statistics across subjects
- 'scripts/plot_stats_decoding.py'
- 'scripts/plot_time_freqs.py'
- 'scripts/run_plot_subscore_gat.py'  # score each visibility condition, correlates decoding scores with experimental conditions etc

#### Models
- 'scripts/run_plot_simulations.py'

#### Additional control analyses
- 'scripts/run_decod_angles_bias.py'  # control analyses to test independence between target and probe codes
- 'scripts/plot_decod_angles_bias.py'
- 'scripts/run_decod_phase_probe.py' # demonstrate that estimators fitted on the probe phase can significantly predict the target phase
- 'scripts/run_plot_veryhighpass.py' # show that high pass filtering removes late metastability


Dependencies
============

- MNE: 0.13.dev0
- scikit-learn: 0.18.dev0
- pandas: 0.18.1
- matplotlib: 1.5.1
- scipy: 0.17


Acknowledgements
================

This project is powered by

![logos](docs/logo_computing.png)

and JRK received fundings from

![logos](docs/logo_funding.png)
