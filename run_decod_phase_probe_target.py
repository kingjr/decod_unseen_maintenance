"""Additional analysis consisting in taking the estimators trained on the probe
phase to predict the phase of the target. The results show that the phase of
the target is decodable slightly above chance, although this effect is
extremely tiny"""
import numpy as np
from jr.gat import subscore
from jr.plot import pretty_decod
from base import stats
from config import load, save, subjects
from conditions import analyses

analysis = [ana for ana in analyses if ana['title'] == 'Target Phase'][0]
toi = [-.100, .305]

scores = list()
for s, subject in enumerate(subjects):  # Loop across each subject
    print(subject)
    epochs = load('epochs', subject=subject, preload=True)
    events = load('behavior', subject=subject)

    # select trials
    query, condition = analysis['query'], analysis['condition']
    sel = range(len(events)) if query is None \
        else events.query(query).index
    sel = [ii for ii in sel if ~np.isnan(events[condition][sel][ii])]
    y = np.array(events[condition], dtype=np.float32)

    # Load classifier
    gat, _, sel_gat, _ = load('decod', subject=subject, analysis='probe_phase')
    # only keep estimators after probe onset
    times = gat.train_times_['times']
    toi_ = np.where((times >= (.800 + toi[0])) & (times < (.800 + toi[1])))[0]
    gat.estimators_ = [gat.estimators_[t] for t in toi_]
    gat.train_times_['times'] = [gat.train_times_['times'][t] for t in toi_]
    gat.train_times_['slices'] = [gat.train_times_['slices'][t] for t in toi_]
    # predict all trials, including absent to keep cv scheme
    gat.predict(epochs[sel_gat])

    # subscore on present only
    gat.scores_ = subscore(gat, sel, y[sel])
    scores.append(gat.scores_)

    # Save classifier results
    save([gat, analysis, sel_gat, events], 'decod', subject=subject,
         analysis=analysis['name'] + '_probe_to_target')

# FIXME probe starts at 816!
times = epochs.times[:-5]
score_diag = np.array([np.diag(np.array(score)[4:, :]) for score in scores])
p_val = stats(score_diag)
pretty_decod(score_diag, times=times, sig=p_val < .05, color=analysis['color'],
             fill=True)
# toi_ = np.where((times >= toi[0]) & (times < toi[1]))[0]
toi_ = np.where(p_val < .05)[0]
score = np.mean(score_diag[:, toi_], axis=1)
print(np.mean(score, axis=0),
      np.std(score, axis=0) / np.sqrt(len(score_diag)),
      np.min(p_val[toi_]),
      times[toi_])
