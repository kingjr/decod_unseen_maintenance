import matplotlib.pyplot as plt
import pickle

from mne import read_epochs
from mne.decoding import TimeDecoding, GeneralizationAcrossTime

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVR

from jr.gat import subscore
from jr.gat import AngularRegression, scorer_angle
from jr.stats import corr_linear_circular

###############################################################################
epochs = read_epochs('data/subject_01-epo.fif')
with open('data/subject_01-eve.pkl', 'rb') as f:
    events = pickle.load(f)

# Let's have a look at the average response
evoked = epochs.average()
evoked.plot_joint(title='all')

###############################################################################
# Let's see whether we can isolate the target evoke response.
# For this we can subtract the absent trials (mask only) from the present:
evo_present = epochs[events['target_present']].average()
evo_absent = epochs[-events['target_present']].average()

evoked.data = evo_present.data - evo_absent.data
evoked.plot_joint(title='Present - Absent')

###############################################################################
# The presence of the target evoke some activity. We can 'decode' this
# presence at each time point by fitting a classifier with all sensors
# at a given time point.

# See `method_decoding.ipynb` to see how this is built.

# Since there's a huge class imbalance, we're using a probabilistic output
# and an ROC scorer.
def scorer(y_true, y_pred):
    return roc_auc_score(y_true, y_pred[:, 1])
td = TimeDecoding(predict_method='predict_proba', scorer=scorer, n_jobs=-1)
td.fit(epochs, y=events['target_present'])
td.score(epochs, y=events['target_present'])
td.plot(chance=.5)

###############################################################################
# The decoding tells very little about the underlying neural substrates.
# Temporal generalization can help clarify the functional architecture that
# underlies the signal.
gat = GeneralizationAcrossTime(predict_method='predict_proba', scorer=scorer,
                               n_jobs=-1)
y = events['target_present']  # in machine learning, y is the regressor.
gat.fit(epochs, y=y)
gat.score(epochs, y=y)
gat.plot()

# These scores reflects both seen and unseen trials. Let's see how these
# two categories differe with one another
subselects = dict(seen=np.where(events['detect_button'] > 0)[0],
                  unseen=np.where(events['detect_button'] == 0)[0],)
y = events['target_present']
for name, idx in subselects.iteritems():
    gat.scores_ = subscore(gat, y=y[idx], sel=idx)
    gat.plot(title=name)

###############################################################################
# Detecting the presence of the target is easy. Let's try to detect the
# orientation of the probe angle

# Univariate analyses: do the amplitude of the meg channels correlate with the
# angle of the stimulus. Note that for practical purposes, we consider that the
# angle is two times the orientations (so that we can easily use trigonometry).

# define regressor
y = np.array(events['probe_circAngle'].values)
X = epochs._data

# We need a 2D X (trials x dimension)
n_trial, n_chan, n_time = epochs._data.shape
X = X.reshape([-1, n_chan * n_time])

# linear circular correlation between MEG and stim angle
_, R2, _ = corr_linear_circular(X, y)

# plot back using MNE
R2 = R2.reshape([n_chan, n_time])
evoked.data = R2
evoked.plot_joint()

###############################################################################
# The same can be done with decoding. One of the advantages is that we
# are not restricted to R squared estimates (which are only one-tailed), but we
# can estimate the angle error between the true and the predicted angle (which)
# is two-tailed. This will simplify the stats across subjects, because we know
# the chance level can be analytically inferred.

# Let's use a double regressor to estimate the angle from the sine and cosine
# See `method_model_types.ipynb` to see understand how this is built.
clf_angle = make_pipeline(StandardScaler(), AngularRegression(clf=LinearSVR()))
td = TimeDecoding(clf=clf_angle, scorer=scorer_angle, n_jobs=-1)

# define regressor
y = np.array(events['probe_circAngle'].values)

# This can take a while, so let's only decode around probe onset
epochs_probe = epochs.crop(.700, copy=True)
td.fit(epochs_probe[present], y=y[present])
td.score(epochs_probe[present], y=y[present])
td.plot(chance=0.)
