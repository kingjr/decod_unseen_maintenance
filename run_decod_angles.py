"""Sub analyses related to the decoding of probe and target orientations"""
import numpy as np
from jr.stats import circ_tuning, circ_mean, repeated_spearman
from jr.gat import get_diagonal_ypred, scorer_angle
from config import subjects, tois, load, save
from conditions import analyses, subscores
from base import get_predict, get_predict_error, angle_acc
analyses = [analysis for analysis in analyses if analysis['name'] in
            ['target_circAngle', 'probe_circAngle']]


# These analyses are applied both for target and probe related estimators
for analysis in ['target_circAngle', 'probe_circAngle']:
    # Initialize results
    results = dict(diagonal=list(), angle_pred=list(), toi=list(),
                   subscore=list(), corr_contrast=list(), corr_pas=list(),
                   R_contrast=list(), R_vis=list(), align_on_diag=list(),
                   early_maintain=list(), R_vis_duration=list(),
                   R_contrast_toi=list(), R_vis_toi=list())
    for s, subject in enumerate(subjects):
        # Load data
        print s
        gat, _, events_sel, events = load('decod', subject=subject,
                                          analysis=analysis)
        y_true = gat.y_train_
        times = gat.train_times_['times']
        subevents = events.iloc[events_sel].reset_index()
        n_bins = 24

        # Mean error across trial on the diagonal
        y_pred = np.transpose(get_diagonal_ypred(gat), [1, 0, 2])
        score = scorer_angle(y_true, y_pred)
        results['diagonal'].append(score)

        # Mean prediction for each angle at peak time
        # This is to be able to plot the tuning functions (histogram) for each
        # stimulus orientation separately.
        toi = tois[1] if analysis == 'target_circAngle' else tois[3]
        results_ = list()
        for angle in np.unique(y_true):
            y_pred = get_predict(gat, sel=np.where(y_true == angle)[0],
                                 toi=toi, mean=True)[..., 0]
            probas, bins = circ_tuning(y_pred, n=n_bins)
            results_.append(probas)
        results['angle_pred'].append(results_)

        # Mean tuning error per Time Region of Interest (TOI):
        # Does the decoding performance of stimulus orientation varies with
        # time?
        results_ = list()
        for toi in tois:
            y_error = get_predict_error(gat, toi=toi, mean=True)
            probas, bins = circ_tuning(y_error, n=n_bins)
            results_.append(probas)
        results['toi'].append(results_)

        # Mean y_error per toi per visibility rating (0-3):
        # Is decoding performance significant when subjects reported not seen
        # the target (0), guessing it (1) ...
        results_ = dict()
        y_pred = np.transpose(get_diagonal_ypred(gat), [1, 0, 2])
        n_train = len(gat.y_pred_)
        for subanalysis in subscores:
            results_[subanalysis[0] + '_toi'] = list()
            # subselect events (e.g. seen vs unseen)
            subsel = subevents.query(subanalysis[1]).index
            # add nan if no trial matches subconditions
            if len(subsel) < 5:
                results_[subanalysis[0]] = np.nan * np.zeros(n_train)
                for toi in tois:
                    results_[subanalysis[0] + '_toi'].append(np.nan)
                continue
            # dynamics of mean error
            results_[subanalysis[0]] = scorer_angle(
                y_true[subsel], y_pred[subsel, :, 0])
            # mean error per toi
            for toi in tois:
                # integrate over time points to get single estimate
                toi = np.where((times >= toi[0]) &
                               (times <= toi[1]))[0]
                angle, radius = y_pred[..., 0], y_pred[..., 1]
                cmplx = radius * (np.cos(angle) + 1j * np.sin(angle))
                y_pred_toi = np.angle(np.median(cmplx[:, toi], 1))
                # score
                score = scorer_angle(y_true[subsel], y_pred_toi[subsel])
                # store
                results_[subanalysis[0] + '_toi'].append(score)
        results['subscore'].append(results_)

        # Contrast and visibility modulatory effects:
        # how is decoding performance of the target/probe angles affected by/
        # covaries with the target contrast and visibility?
        y_error = get_predict_error(gat, mean=False)
        R_contrast = list()
        R_vis = list()
        subsel = subevents.query('target_present==True and ' +
                                 'detect_pressed==True').index
        for ii, toi in enumerate(tois):
            toi_ = np.where((times >= toi[0]) & (times < toi[1]))[0]
            y_error_ = np.abs(circ_mean(y_error[subsel, :][:, toi_], axis=1))
            R_vis_ = repeated_spearman(y_error_,
                                       events['detect_button'][subsel])
            R_contrast_ = repeated_spearman(y_error_,
                                            events['target_contrast'][subsel])
            R_vis.append(R_vis_)
            R_contrast.append(R_contrast_)
        results['R_vis_toi'].append(R_vis)
        results['R_contrast_toi'].append(R_contrast)

        # Duration analyses
        # are early a late estimators equally able to generalize over time?
        # Do these durations vary as a function of visibility?
        results_ = list()
        R_vis_duration = list()
        for toi in tois:
            # for each toi: 1. realign pred on diag, 2. compute error
            y_error = get_predict_error(gat, toi=toi, typ='align_on_diag',
                                        mean=False)
            results_pas = list()
            for pas in range(4):
                # select condition
                pas_cond = 'detect_button == %i' % pas
                present_cond = ' and target_present == True'
                sel = subevents.query(pas_cond + present_cond).index
                # if no trial matches seleted conditions, add NaNs
                if len(sel) == 0:
                    results_pas.append(np.nan * np.zeros(y_error.shape[2]))
                    continue
                # 3. mean error across trials
                y_error_ = angle_acc(y_error[sel, ...], axis=0)
                # 4. mean across classifier (toi)
                y_error_ = np.mean(y_error_, axis=0)
                results_pas.append(y_error_)
            results_.append(results_pas)
            # Finally, test if duration if affected by visibility :
            sel = subevents.query('target_present == True').index
            # mean clf R in TOI
            X = angle_acc(y_error[sel, :, :], axis=1)
            R = repeated_spearman(X.reshape([len(sel), -1]),
                                  np.array(subevents['detect_button'][sel]))
            R_vis_duration.append(R.reshape(y_error.shape[2]))
        # shape(subjects, tois, pas, times)
        results['align_on_diag'].append(results_)
        results['R_vis_duration'].append(R_vis_duration)

        # Maintenance of early classifiers
        results_ = list()
        for toi in [[.100, .150], [.170, .220]]:
            y_error = get_predict_error(gat, toi=toi, typ='gat', mean=True)
            results_.append(angle_acc(y_error, axis=0))
        results['early_maintain'].append(results_)

    # Save results
    results['times'] = times
    results['bins'] = bins
    save(results, 'score', subject='fsaverage', analysis=analysis + '-tuning',
         overwrite=True, upload=True)
