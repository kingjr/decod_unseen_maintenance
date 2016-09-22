Brain mechanisms underlying the brief maintenance of seen and unseen sensory information
=======================================================================================

This repository stores all scripts to analyze MEG data from the eponymous manuscript, by Jean-Remi King, Niccolo Pescetelli & Stanislas Dehaene.

The corresponding manuscript has been submitted and **is currently not peer-reviewed**. The preprint pdf can be downloaded from [BioArXiv](http://biorxiv.org/content/early/2016/02/18/040030), and will be updated once the paper is accepted for publication.

Abstract
========

Recent studies of "unconscious working memory" have challenged the notion that only visible stimuli can be actively maintained over time. In the present study, we investigated the neural dynamics underlying the processing and brief maintenance of subjectively visible and invisible stimuli, using machine learning techniques applied to magnetoencephalography recordings (MEG). Subjects were presented with a masked Gabor patch whose angle had to be briefly memorized. We show that targets are encoded in early brain activity independently of their visibility, and that the maintenance of target presence and orientation continues to be decodable above chance level throughout the retention period, even in the lowest visibility condition. Source and temporal generalization analyses revealed that perceptual maintenance depends on a deep hierarchical network ranging from early visual cortex to temporal, parietal and frontal cortices. Importantly, the representations coded in the late processing stages of this network specifically predict subjective reports. Together, these results challenge several predictions of current neuronal theories of visual awareness and suggest that unseen sensory information can be briefly maintained within the higher processing stages of visual perception.


Scripts
=======

Overall, the scripts remain designed for research purposes, and could therefore be improved and clarified. If you judge that some codes would benefit from specific clarifications do not hesitate to raise an issue.

The scripts are generally decomposed in terms of general functions (base), actual analyses (decoding, cluster analyses), and report (plotting, tables, quick stats).

#### Config files
- 'base.py' # where all generic functions are defined
- 'conditions.py' # where the analyses, multivariate estimators, scorers etc are defined
- 'config.py'  # where the paths and filenames are setup

#### Prepare data
- 'run_download_all.py'  # downloads data from aws
- 'run_behavior.py'  # plot visibility, accuracy and d-prime
- 'run_preprocessing.py'  # filter and epochs raw signals

#### Mass univariate sensor analyses
- 'run_sensor_analysis.py'  # analyze sensor ERF within each subject
- 'run_stats_sensors.py'  # run second-level (between subjects) stats
- 'plot_stats_sensors.py'  # plot average topographies across subjects

#### Mass univariate source analyses
- 'run_preprocess_source.py'
- 'run_source_analysis.py'
- 'run_stats_source.py'
- 'plot_anatomy_roi.py'
- 'plot_source_analysis.py'  # non-thresholded sources
plot_source_time_course  # time course of source ROI

#### Decoding
- 'run_decoding.py'  # evoked
- 'run_decoding_timefreq.py'  # induced
- 'run_stats_decodings.py'
- 'plot_stats_decoding.py'
- 'plot_time_freqs.py'
- 'run_subscore_gat.py'  # score each visibility condition, correlates score with factors etc

#### Models
- 'run_simulations.py'

#### Additional control analyses
- 'run_decod_angles_bias.py'  # control analyses to test independence between target and probe codes
- 'plot_decod_angles_bias.py'
- 'run_decod_phase_probe.py' # demonstrate that estimators fitted on the probe phase can significantly predict the target phase
- 'run_veryhighpass.py' # show that high pass filtering removes late metastability


Data
====

The available data is currently partial. The complete data and non truncated results will be made publicly available once the paper is accepted for publication.

Online Tutorials and Results
============================

I wrote a series of tutorials available in notebook/ to clarify the methods, and strip them from the data handling.

* `method_decoding.ipynb` explains the general procedure used to perform decoding with MEG data.
* `method_model_types.ipynb` explains how categorical, ordinal and circular models can be fitted and scored.
* `method_statistics.ipynb` explains how the statistics are performed in the manuscript.
* `results_summary.ipynb` gives a preview of some of the results to allow user to replicate our analyses, or go further by looking at individual subjects, test other statistical methods etc.

Also consider looking at the [MNE-Python gallery](http://martinos.org/mne/dev/auto_examples/). You will find several examples, showing how the `TimeDecoding` and `GeneralizationAcrossTime` can be used.


Dependencies
============

- mne-python: 0.13.dev0
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
