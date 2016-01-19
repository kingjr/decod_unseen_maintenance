Selective maintenance of seen and unseen sensory features in the human brain
============================================================================

This repository stores all scripts to analyze MEG data from the unconscious orientations decoding project, led by Jean-Remi King and Niccolo Pescetelli at Neurospin, CEA, Gif / Yvette, France. The details of the experiment will soon be available on arXiv once the corresponding paper is under review. This repository is still work-in-progress.

Overall, the scripts remain designed for research purposes, and could clearly be improved and better documented in many places. If you judge that some codes would benefit from specific clarifications do not hesitate to contact us. If, however, you are interested in using the tools implemented in the present papers for your own data, most of them have been implemented in [MNE-Python](https://github.com/mne-tools/mne-python).

Experimental protocol
=====================

20 subjects were presented with a Gabor patch that could vary among 6 different orientations and flashed very briefly on screen before being masked by a heavy-contrasted circular mask. After the mask a strong contrast probe Gabor patch was flashed around 800 ms after the initial target. The probe's orientation could be tilted 30° to the right or to the left of the initial target. Subjects evaluated the tilt of the probe compared to the target (clockwise vs counter clockwise) with left button press and after reported the visibility of the target stimulus with the 'perceptual awareness scale' (PAS) ranging from 1 (not seen) to 4 (perfectly seen).


Dependencies
============

- [MNE-Python](https://github.com/mne-tools/mne-python) to analyze MEG data
- [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) to run decoding analyses
- [jr-tools](https://github.com/kingjr/jr-tools), wrapper to extend the functionalities of the `TimeDecoding` and `GeneralizationAcrossTime` MNE-Python classes, and add statistical and plotting functions.


Scripts
=======

The scripts are generally decomposed in terms of general functions, actual analyses (decoding, cluster analyses), and report (plotting, tables, quick stats).

- `conditions.py` defines the analyses at a high level: how are the contrasts defined (e.g. decoding of the presence, phase etc), which conditions to include and exclude, what statistical pipeline to use depending (circular, linear, categorical) etc.

- `config.py` defines the data handling: data paths, downloading scheme (when run on a distant server), subjects to include in the analyses, time regions of interest etc. This is generally the script that one can modify to run the analyses in a more or less fast manner.

- `base.py` contains all generic functions.

- `run_behavior.py` analyses all behavioral data.

- `run_sensor_analysis.py` pipeline to compute univariate analyses for each subject and each analysis.

- `run_stats_sensors.py` pipeline to compute 2nd order stats across subjects for univariate effects.

- `plot_stats_sensors.py` outputs the main topographical effects.

- `run_decoding.py` pipeline to compute decoding and temporal generalization for each subject and each analysis.

- `run_stats_decoding.py` pipeline to compute 2nd order stats across subjects for decoding and temporal generalization effects.

- `plot_stats_decoding.py` outputs the main decoding figures.

- `plot_graphs` transform the temporal generalization matrices in terms of animated graphs.

The other `run_*` as well as `plot_` scripts are generally designed for control analyses, typically aiming at investigating properties of decoding scores as a function of time, visibility etc. These script are under documented. Do not hesitate to ask me for clarifications.

Folders
=======

- `results` is where the scripts outputs their figures and tables.

- `cloud` is a series of script to handle data download when the pipeline is run on a distant server (typically AWS)

- `sandbox` is intended to ultimately disappear.
