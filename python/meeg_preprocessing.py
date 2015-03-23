# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import sys
import os
import os.path as op
import subprocess
import time
import json
from distutils.version import LooseVersion

from mne import pick_types
from mne.utils import logger, set_log_file
from mne.report import Report

from mne.io.constants import FIFF


def get_data_picks(inst, meg_combined=False):
    """Get data channel indices as separate list of tuples

    Parameters
    ----------
    inst : instance of mne.measuerment_info.Info
        The info
    meg_combined : bool
        Whether to return combined picks for grad and mag.

    Returns
    -------
    picks_list : list of tuples
        The list of tuples of picks and the type string.
    """
    info = inst.info
    picks_list = []
    has_mag, has_grad, has_eeg = [k in inst for k in ('mag', 'grad', 'eeg')]
    if has_mag and (meg_combined is not True or not has_grad):
        picks_list.append(
            (pick_types(info, meg='mag', eeg=False, stim=False), 'mag')
        )
    if has_grad and (meg_combined is not True or not has_mag):
        picks_list.append(
            (pick_types(info, meg='grad', eeg=False, stim=False), 'grad')
        )
    if has_mag and has_grad and meg_combined is True:
        picks_list.append(
            (pick_types(info, meg=True, eeg=False, stim=False), 'meg')
        )
    if has_eeg:
        picks_list.append(
            (pick_types(info, meg=False, eeg=True, stim=False), 'eeg')
        )
    return picks_list


def fname_to_string(fname):
    """Return given file as sring
    Parameters
    ----------
    fname : str
        absolute path to file.
    """
    with open(fname) as fid:
        string = fid.read()
    return string


def _get_git_head(path):
    """Aux function to read HEAD from git"""
    if not isinstance(path, str):
        raise ValueError('path must be a string, you passed a {}'.format(
            type(path))
        )
    if not op.exists(path):
        raise ValueError('This path does not exist: {}'.format(path))
    command = ('cd {gitpath}; '
               'git rev-parse --verify HEAD').format(gitpath=path)
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               shell=True)
    proc_stdout = process.communicate()[0].strip()
    del process
    return proc_stdout


def get_versions(sys):
    """Import stuff and get versions if module

    Parameters
    ----------
    sys : module
        The sys module object.

    Returns
    -------
    module_versions : dict
        The module names and corresponding versions.
    """
    module_versions = {}
    for name, module in sys.modules.items():
        if '.' in name:
            continue
        module_version = LooseVersion(getattr(module, '__version__', None))
        module_version = getattr(module_version, 'vstring', None)
        if module_version is None:
            module_version = None
        elif 'git' in module_version or '.dev' in module_version:
            git_path = op.dirname(op.realpath(module.__file__))
            head = _get_git_head(git_path)
            module_version += '-HEAD:{}'.format(head)

        module_versions[name] = module_version
    return module_versions


def create_run_id():
    """Get the run hash

    Returns
    -------
    run_hash : str
        A hex hash.
    """
    return str(time.time().hex())


def setup_provenance(script, results_dir, use_agg=True):
    """Setup provenance tracking

    Parameters
    ----------
    script : str
        The script that was executed.
    results_dir : str
        The results directory.
    use_agg : bool
        Whether to use the 'Agg' backend for matplotlib or not.

    Returns
    -------
    report : mne.report.Report
        The mne report.

    Side-effects
    ------------
    - make results dir if it does not exists
    - sets log file for sterr output
    - writes log file with runtime information
    """
    if use_agg is True:
        import matplotlib
        matplotlib.use('Agg')

    if not op.isfile(script):
        raise ValueError('sorry, this is not a script!')

    step = op.splitext(op.split(script)[1])[0]
    results_dir = op.join(results_dir, step)
    if not op.exists(results_dir):
        logger.info('generating results dir')
        os.mkdir(results_dir)

    run_id = create_run_id()
    logger.info('generated run id: %s' % run_id)

    logger.info('preparing logging:')
    logging_dir = op.join(results_dir, run_id)
    logger.info('... making logging directory: %s' % logging_dir)
    os.mkdir(logging_dir)
    modules = get_versions(sys)
    runtime_log = op.join(logging_dir, 'run_time.json')
    with open(runtime_log, 'w') as fid:
        json.dump(modules, fid)
    logger.info('... writing runtime info to: %s' % runtime_log)
    std_logfile = op.join(logging_dir, 'run_output.log')

    script_code = op.join(logging_dir, 'script.py')
    with open(script_code, 'w') as fid:
        with open(script) as script_fid:
            source_code = script_fid.read()
        fid.write(source_code)
    logger.info('... logging source code')

    logger.info('... preparing Report')
    report = Report(title=step)
    report.data_path = logging_dir
    logger.info('... setting logfile: %s' % std_logfile)
    set_log_file(std_logfile)
    return report, run_id, results_dir, logger


def set_eog_ecg_channels(raw, eog_ch='EEG062', ecg_ch='EEG063'):
    """Set the EOG and ECG channels

    Will modify the channel info in place.

    Parameters
    ----------
    raw : instance of Raw
        The raw object.
    eog_ch : list | str
        EOG channel name(s).
    ecg_ch : list | str
        ECG channel name(s).
    """
    if isinstance(eog_ch, basestring):
        eog_ch = [eog_ch]
    if isinstance(ecg_ch, basestring):
        ecg_ch = [ecg_ch]
    for channel in eog_ch:
        raw.info['chs'][raw.ch_names.index(channel)]['kind'] = FIFF.FIFFV_EOG_CH
    for channel in ecg_ch:
        raw.info['chs'][raw.ch_names.index(channel)]['kind'] = FIFF.FIFFV_ECG_CH
