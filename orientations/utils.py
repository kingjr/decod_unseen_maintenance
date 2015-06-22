import numpy as np
import scipy.io as sio
import pandas as pd

from sklearn.svm import LinearSVR, SVC
from sklearn.linear_model import LogisticRegression


def fix_wrong_channel_names(inst):
    from mne.epochs import EpochsArray
    from mne.evoked import Evoked
    inst.info['chs'] = inst.info['chs'][:306]
    inst.info['nchan'] = 306
    if isinstance(inst, Evoked):
        inst.data = inst.data[:306, :]
    elif isinstance(inst, EpochsArray):
        inst._data = inst._data[:, :306, :]
    else:
        raise ValueError('Unknown instance')
    return inst


def load_epochs_events(subject, paths=None, data_type='erf',
                       lock='target'):
    # Get MEG data
    meg_fname = paths('epoch', subject=subject, data_type=data_type, lock=lock)
    epochs = load_FieldTrip_data(meg_fname)
    epochs = fix_wrong_channel_names(epochs)
    # Get behavioral data
    bhv_fname = paths('behavior', subject=subject)
    events = get_events(bhv_fname)
    epochs.crop(-.200, 1.200)
    return epochs, events


def angle2circle(angles):
    """from degree to radians multipled by rm2"""
    return np.deg2rad(2 * (np.array(angles) + 7.5))


def load_FieldTrip_data(meg_fname):
    from mne.io.meas_info import create_info
    from mne.epochs import EpochsArray
    """XXX Here explain what this does"""
    # import information from fieldtrip data to get data shape
    ft_data = sio.loadmat(meg_fname[:-4] + '.mat', squeeze_me=True,
                          struct_as_record=True)['data']
    # import binary MEG data
    bin_data = np.fromfile(meg_fname[:-4] + '.dat', dtype=np.float32)
    Xdim = ft_data['Xdim'].item()
    bin_data = np.reshape(bin_data, Xdim[[2, 1, 0]]).transpose([2, 1, 0])

    # Create an MNE Epoch
    n_trial, n_chans, n_time = bin_data.shape
    sfreq = ft_data['fsample'].item()
    time = ft_data['time'].item()[0]
    tmin = min(time)
    chan_names = [str(label) for label in ft_data['label'].item()]
    chan_types = np.squeeze(np.concatenate(
        (np.tile(['grad', 'grad', 'mag'], (1, 102)),
         np.tile('misc', (1, n_chans - 306))), axis=1))
    info = create_info(chan_names, sfreq, chan_types)
    events = np.c_[np.cumsum(np.ones(n_trial)) * 5 * sfreq,
                   np.zeros(n_trial),
                   ft_data['trialinfo'].item()]
    epochs = EpochsArray(bin_data, info, events=events, tmin=tmin)

    return epochs


def get_events(bhv_fname):
    # Load behavioral file
    trials = sio.loadmat(bhv_fname, squeeze_me=True,
                         struct_as_record=True)["trials"]

    def trial2event(trial):
        event = dict()
        # Change meaningless values with NaNs
        event['target_present'] = trial['present'] == 1
        event['discrim_pressed'] = trial['response_responsed'] == 1
        event['detect_pressed'] = trial['response_vis_responsed'] == 1
        nan_default = lambda check, value: value if check else np.nan
        target_present = lambda v: nan_default(event['target_present'], v)
        # discrim_pressed = lambda v: nan_default(event['discrim_pressed'], v)
        discrim_buttons = lambda v: nan_default(
            v in ['left_green', 'left_yellow'], 1. * (v == 'left_green'))
        detect_pressed = lambda v: nan_default(event['detect_pressed'], v)
        # Target
        event['target_contrast'] = [0, .5, .75, 1][trial['contrast'] - 1]
        event['target_spatialFreq'] = target_present(trial['lambda'] == 1)
        event['target_angle'] = target_present(trial['orientation'] * 30 - 15)
        event['target_circAngle'] = angle2circle(event['target_angle'])
        # Probe
        event['probe_angle'] = (trial['orientation'] * 30 - 15 +
                                trial['tilt'] * 30) % 180
        event['probe_circAngle'] = angle2circle(event['probe_angle'])
        event['probe_tilt'] = target_present(trial['tilt'])
        # Response 1: forced choice discrimination
        event['discrim_button'] = discrim_buttons(trial['response_keyPressed'])

        event['discrim_correct'] = target_present(trial['correct'] == 1)
        # Response 2: detection/visibility
        event['detect_button'] = \
            detect_pressed(trial['response_visibilityCode'] - 1)
        event['detect_seen'] = event['detect_button'] > 0
        return event

    events = list()
    for t, trial in enumerate(trials):
        # define previous trial
        event = trial2event(trial)
        if t > 1:
            previous_event = trial2event(trials[t-1])
            for key in previous_event:
                event['previous_' + key] = previous_event[key]
        events.append(event)
    events = pd.DataFrame(events)
    return events


class clf_2class_proba(LogisticRegression):  # XXX not used?
    """Probabilistic SVC for 2 classes only"""
    def predict(self, x):
        probas = super(clf_2class_proba, self).predict_proba(x)
        return probas[:, 1]


class SVC_2class_proba(SVC):
    """Probabilistic SVC for 2 classes only"""
    def predict(self, x):
        probas = super(clf_2class_proba, self).predict_proba(x)
        return probas[:, 1]


class SVR_angle(LinearSVR):

    def __init__(self):
        from sklearn.svm import LinearSVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        scaler_cos = StandardScaler()
        scaler_sin = StandardScaler()
        svr_cos = LinearSVR(C=1)
        svr_sin = LinearSVR(C=1)
        self.clf_cos = Pipeline([('scaler', scaler_cos), ('svr', svr_cos)])
        self.clf_sin = Pipeline([('scaler', scaler_sin), ('svr', svr_sin)])

    def fit(self, X, y):
        """
        Fit 2 regressors cos and sin of angles y
        Parameters
        ----------
        X : np.array, shape(n_trials, n_chans, n_time)
            MEG data
        y : list | np.array (n_trials)
            angles in degree
        """
        # Go from orientation space (0-180 degrees) to complex space
        # (0 - 2 pi radians)
        self.clf_cos.fit(X, np.cos(y))
        self.clf_sin.fit(X, np.sin(y))

    def predict(self, X):
        """
        Predict orientation from MEG data in radians
        Parameters
        ----------
        X : np.array, shape(n_trials, n_chans, n_time)
            MEG data
        Returns
        -------
        predict_angle : list | np.array, shape(n_trials)
            angle predictions in radian
        """
        predict_cos = self.clf_cos.predict(X)
        predict_sin = self.clf_sin.predict(X)
        predict_angle = np.arctan2(predict_sin, predict_cos)
        return predict_angle
