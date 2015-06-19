import numpy as np
import scipy.io as sio
import pandas as pd

from sklearn.svm import LinearSVR, SVC
from sklearn.linear_model import LogisticRegression


def load_epochs_events(subject, paths=None, data_type='erf',
                       lock='target'):
    # Get MEG data
    meg_fname = paths('epoch', subject=subject, data_type=data_type, lock=lock)
    epochs = load_FieldTrip_data(meg_fname)
    # Get behavioral data
    bhv_fname = paths('behavior', subject=subject)
    events = get_events(bhv_fname)
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

    keys = ['ISI', 'response_keyPressed', 'time_jitter',
            'block', 'response_responsed', 'time_maskIn',
            'break', 'response_tilt', 'time_preparation',
            'contrast', 'response_time', 'time_probe',
            'correct', 'response_vis_RT', 'time_prompt',
            'feedback', 'response_vis_keyPressed', 'time_response',
            'gabors', 'response_vis_responsed', 'time_targetIn',
            'lambda', 'response_vis_time', 'time_targetOut',
            'localizer', 'response_visibilityCode', 'trialid',
            'orientation', 'tilt', 'ttl_value',
            'present', 'time_delay', 'response_RT', 'time_feedback_on']

    contrasts = [0, .5, .75, 1]
    events = list()
    for t, trial in enumerate(trials):
        event = dict()
        # define previous trial
        prevtrial = trials[t-1]
        for key in keys:
                event[key] = trial[key]
        # manual new keys
        event['targetContrast'] = contrasts[event['contrast']-1]
        event['seen_unseen'] = event['response_visibilityCode'] > 1

        event['orientation_target'] = event['orientation']*30-15
        event['orientation_probe'] = (event['orientation']*30-15 +
                                      event['tilt'] * 30) % 180
        event['orientation_target_rad'] = angle2circle(
            event['orientation_target'])
        event['orientation_probe_rad'] = angle2circle(
            event['orientation_probe'])
        event['targetContrast'] = contrasts[event['contrast']-1]
        event['seen_unseen'] = event['response_visibilityCode'] > 1
        event['previous_trial_visibility'] = prevtrial[
            'response_visibilityCode']
        event['previous_orientation_target'] = prevtrial[
            'orientation']*30-15
        event['previous_orientation_probe'] = (prevtrial[
            'orientation']*30-15 +
            trial['tilt'] * 30) % 180

        # append to all events
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
