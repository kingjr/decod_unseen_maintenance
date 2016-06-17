"""Behavioral analyses and plot"""
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import pandas
from itertools import product
from jr.plot import pretty_plot, plot_sem
from scipy.stats import wilcoxon
from jr.stats import dPrime, repeated_spearman
from config import load, report, subjects
mpl.rcParams['legend.fontsize'] = 10


# Gather all beahvioral data
subjects_events = list()
for subject in subjects:
    events = load('behavior', subject=subject)
    events['subject'] = subject
    subjects_events.append(events)
data = pandas.concat(subjects_events)

contrasts = [0., .5, .75, 1.]
visibilities = [0., 1., 2., 3.]

# ##############################################################################
# Plotting functions
cmap = mpl.colors.LinearSegmentedColormap.from_list('RdPuBu', ['b', 'r'])


def make_3d_plot(ax, xs, y, zs, colors):
    # make areas
    verts = []
    verts_sem = []
    for ii in range(y.shape[2]):
        ys = np.mean(y[:, :, ii], axis=0)
        verts.append(list(zip(np.hstack((0., xs, 1.)),
                              np.hstack((0., ys, 0.)))))
        ys += np.std(y[:, :, ii], axis=0) / np.sqrt(len(subjects))
        verts_sem.append(list(zip(np.hstack((0., xs, 1.)),
                                  np.hstack((0., ys, 0.)))))
    poly = PolyCollection(verts, facecolors=colors, edgecolor='k',
                          linewidth=1)
    poly.set_alpha(.75)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    poly = PolyCollection(verts_sem, facecolors=colors, edgecolor='none')
    poly.set_alpha(0.5)
    ax.add_collection3d(poly, zs=zs, zdir='y')
    return pretty_plot(ax)


def fill_between_gradient(xx, yy, clim=None, cmap='RdBu_r', alpha=1., ax=None,
                          zorder=-1):
    """/!\ should not change ylim as it vary the color"""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    if ax is None:
        fig, ax = plt.subplots(1)
        xlim, ylim = (np.min(xx), np.max(xx)), (np.min(yy), np.max(yy))
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if clim is None:
        clim = [0, 1]
    path = Path(np.array([xx, yy]).transpose())
    patch = PathPatch(path, facecolor='none', edgecolor='none')
    ax.add_patch(patch)
    ax.imshow(xx.reshape(np.size(yy), 1), vmin=clim[0], vmax=clim[1],
              origin='lower', cmap=cmap,
              alpha=alpha, aspect='auto', clip_path=patch, clip_on=True,
              extent=xlim + ylim, zorder=zorder)
    return ax

# #############################################################################
# 1. report of visibility as a function of contrast
x_vis = np.zeros((len(subjects), len(contrasts), len(visibilities)))
for (s, subject), (c, contrast), (v, visibility) in product(
        enumerate(subjects), enumerate(contrasts), enumerate(visibilities)):
    query = 'subject==%s and target_contrast==%s and detect_button==%s' % (
        subject, contrast, visibility)
    x_vis[s, c, v] = len(data.query(query))
# normalize per subject per contrast
for (s, subject), (c, contrast) in product(
        enumerate(subjects), enumerate(contrasts)):
    x_vis[s, c, :] /= np.sum(x_vis[s, c, :])


def print_stats(X):
    X = np.array(X)
    dims = np.shape(X)
    dims = np.hstack((dims, 1)) if len(dims) == 1 else dims
    X = np.reshape(X, [len(X), -1])
    stats = np.empty(np.prod(dims[1:]), dtype=object)
    for ii, x in enumerate(X.T):
        m = np.nanmean(x)
        sem = np.nanstd(x) / np.sqrt(sum(~np.isnan(x)))
        stats[ii] = '%.2f +/- %.2f' % (m, sem)
    stats = np.reshape(stats, dims[1:]) if len(dims) > 1 else stats
    return stats

print('unseen absent: %s' % print_stats(x_vis[:, 0, 0]))
print('seen present: %s' % print_stats(np.mean(1. - x_vis[:, 1:, 0], axis=1)))

# compute detection d'
x_vis_dprime = np.nan * np.zeros(len(subjects))
for s, subject in enumerate(subjects):
    def count(pst, seen):
        query = ('subject==%s and target_present==%s and '
                 'detect_seen==%s' % (subject, pst, seen))
        return len(data.query(query))
    hit = count(True, True)
    fa = count(False, True)
    miss = count(True, False)
    cr = count(False, False)
    if 0 not in [hit + miss, fa + cr]:
        x_vis_dprime[s] = dPrime(hit, miss, fa, cr)['d']
print('detection dprime: %s' % print_stats(x_vis_dprime))

fig = plt.figure(figsize=[6, 4])
cmap_contrast = plt.get_cmap('hot_r')
ax = fig.gca(projection='3d')
ax = make_3d_plot(ax, np.linspace(0, 1., 4.), x_vis, np.linspace(.25, 1., 4.),
                  [cmap_contrast(i) for i in contrasts])
ax.set_xlabel('Visibility Rating')
ax.set_ylabel('Contrast')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('Response %', rotation=90)
ax.set_xticks(np.linspace(0, 1, 2))
ax.set_yticks([])
ax.set_zticks(np.linspace(0, .7, 2))
ax.set_ylim(.25, 1.)
ax.set_zlim(0, .75)
tmp_planes = ax.zaxis._PLANES
ax.zaxis.grid('off')
ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                    tmp_planes[0], tmp_planes[1],
                    tmp_planes[4], tmp_planes[5])
# fig.show()
report.add_figs_to_section(fig, 'visibility 3d', 'visibility')


# 2D
fig, ax = plt.subplots(1, figsize=[6, 4])
abs_m = x_vis[:, 0, :].mean(0)
abs_sem = abs_m + x_vis[:, 0, :].mean(0).std(0) / x_vis.shape[0]
pst_m = x_vis[:, 1:, :].mean(1).mean(0)
pst_sem = pst_m + x_vis[:, 1:, :].mean(1).std(0).mean(0) / x_vis.shape[0]
fill = lambda z, c: ax.fill_between(np.linspace(0, 1., 4.), z, alpha=.5,
                                    color=cmap(c),  edgecolor='none')
fill(abs_sem, 0.)
fill(pst_sem, 1.)
fill(abs_m, 0.)
fill(pst_m, 1.)
ax.text(0.1, np.mean(x_vis[:, 0, 0], axis=0) - .05, 'Absent', color=cmap(0.),
        ha='left', va='bottom')
ax.text(1., np.mean(x_vis[:, -1, -1], axis=0) + .05, 'Present',
        color=cmap(1.), ha='right', va='bottom')
ax.set_xlabel('Visibility Rating')
ax.set_ylabel('Response %')
ax.set_xticks(np.linspace(0, 1, 2))
ax.set_yticks(np.linspace(0, .7, 2))
ax.set_ylim(0, .75)
ax = pretty_plot(ax)
report.add_figs_to_section(fig, 'visibility 2d', 'visibility')


# #############################################################################
# 2.0 Overall discrimination performance
x = dict(Accuracy=list(), Dprime=list())
for subject in subjects:
    query = 'subject==%s and target_present==True' % subject
    x['Accuracy'].append(np.nanmean(data.query(query)['discrim_correct']))

    def count(tilt, acc):
        query2 = ' and probe_tilt==%s and discrim_correct==%s' % (tilt, acc)
        return len(data.query(query + query2))

    hits = count(1, True)
    misses = count(1, False)
    fas = count(-1, False)
    crs = count(-1, True)
    d = np.nan
    if (hits + misses) > 0 and (fas + crs) > 0:
        d = dPrime(hits, misses, fas, crs)['d']
    x['Dprime'].append(d)
print('Overall discrimination acc: %s' % print_stats(x['Accuracy']))
print('Overall discrimination dprime: %s' % print_stats(x['Dprime']))

# 2.1 Discrimination performance as a function of visibility
x = dict(Accuracy=np.zeros((len(subjects), 4)),
         Dprime=np.zeros((len(subjects), 4)))
for (s, subject), vis in product(enumerate(subjects), range(4)):
    query = ('subject==%s and target_present==True' % subject +
             ' and detect_button==%f' % vis)
    x['Accuracy'][s, vis] = np.nanmean(data.query(query)['discrim_correct'])

    def count(tilt, acc):
        query2 = ' and probe_tilt==%s and discrim_correct==%s' % (tilt, acc)
        return len(data.query(query + query2))

    hits = count(1, True)
    misses = count(1, False)
    fas = count(-1, False)
    crs = count(-1, True)
    d = np.nan
    if (hits + misses) > 0 and (fas + crs) > 0:
        d = dPrime(hits, misses, fas, crs)['d']
    x['Dprime'][s, vis] = d


# 2.1 discrimination performance as a function of visibility and contrast
x = dict(Accuracy=np.nan * np.zeros((len(subjects), 3, 4)),
         Dprime=np.nan * np.zeros((len(subjects), 3, 4)))

for (s, subject), (c, contrast), (v, visibility) in product(
        enumerate(subjects), enumerate(contrasts[1:]),
        enumerate(visibilities)):
    query = ('subject==%s and target_contrast==%s and detect_button==%s'
             ' and target_present') % (subject, contrast, visibility)
    x['Accuracy'][s, c, v] = np.nanmean(data.query(query)['discrim_correct'])
    count = lambda tilt, acc: len(data.query(
        query + ' and probe_tilt==%s and discrim_correct==%s' % (tilt, acc)))
    hits = count(1, True)
    misses = count(1, False)
    fas = count(-1, False)
    crs = count(-1, True)
    if (hits + misses) > 0 and (fas + crs) > 0:
        x['Dprime'][s, c, v] = dPrime(hits, misses, fas, crs)['d']
    else:
        x['Dprime'][s, c, v] = np.nan

print('Overall discrimination acc: %s' % print_stats(x['Accuracy']))
print('Overall discrimination dprime: %s' % print_stats(x['Dprime']))
x['R_vis'] = repeated_spearman(np.nanmean(x['Accuracy'], axis=1).T,
                               np.arange(4))
print('discrimination varies as a function vis: %s, p=%.5f' % (
    print_stats(x['R_vis']), wilcoxon(x['R_vis'])[1]))
x['R_contrast'] = repeated_spearman(np.nanmean(x['Accuracy'], axis=2).T,
                                    np.arange(3))
print('discrimination varies as a function contrast: %s, p=%.5f' % (
    print_stats(x['R_contrast']), wilcoxon(x['R_contrast'])[1]))


for metric, ylim, in zip(['Accuracy', 'Dprime'], ((.5, 1.), (0., 3.))):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[5, 3])
    # Visibility
    accuracy = np.squeeze(np.nanmean(x[metric], axis=1))
    plot_sem(visibilities, accuracy, color='k', ax=axes[0])
    for vis, acc in zip(visibilities, accuracy.T):
        if metric == 'Dprime':
            p_val = wilcoxon(acc)[1]
        else:
            p_val = wilcoxon(acc - .5)[1]
        axes[0].text(vis, np.nanmean(acc, axis=0) + np.ptp(ylim) / 10,
                     ' *'[p_val < .05], size=15, ha='center')
    axes[0].set_xlabel('Visibility Rating')
    axes[0].set_ylabel(metric)
    axes[0].set_xlim(-.3, 3.3)
    axes[0].set_xticks(visibilities)
    axes[0].set_yticks(ylim)
    axes[0].set_yticklabels(ylim)

    # Contrast
    accuracy = np.squeeze(np.nanmean(x[metric], axis=2))
    plot_sem(contrasts[1:], accuracy, color='k', ax=axes[1])
    for contrast, acc in zip(contrasts[1:], accuracy.T):
        if metric == 'Dprime':
            p_val = wilcoxon(acc)[1]
        else:
            p_val = wilcoxon(acc - .5)[1]
        axes[1].text(contrast, np.nanmean(acc, axis=0) + np.ptp(ylim) / 10,
                     ' *'[p_val < .05], size=15, ha='center')
    axes[1].set_xlabel('Contrast')
    axes[1].set_xticks(contrasts[1:])
    axes[1].set_xlim(.45, 1.05)
    axes[1].set_yticks(ylim)
    axes[1].set_yticklabels([])

    for ax in axes:
        ax = pretty_plot(ax)
        ax.axhline(ylim[0], linestyle='--', color='k')
        ax.set_ylim(ylim[0] - np.ptp(ylim) / 20, ylim[1])
    report.add_figs_to_section(fig, metric, 'discrimination')
# plt.show()
# XXX /!\ Dprime is sig for unseen but this is only if nan are counted as 0,
# and not removed.

# 2.2 discrimination performance of unseen trials
x = dict(Accuracy=list(), Dprime=list())
for subject in subjects:
    query = ('subject==%s' % subject + ' and target_present==True and '
             'detect_button==0. and (discrim_button==-1 or discrim_button==1)')
    x['Accuracy'].append(np.nanmean(data.query(query)['discrim_correct']))

    def count(tilt, acc):
        query2 = ' and probe_tilt==%s and discrim_correct==%s' % (tilt, acc)
        return len(data.query(query + query2))

    hits = count(1, True)
    misses = count(1, False)
    fas = count(-1, False)
    crs = count(-1, True)
    d = np.nan
    if (hits + misses) > 0 and (fas + crs) > 0:
        d = dPrime(hits, misses, fas, crs)['d']
    x['Dprime'].append(d)
print('Overall discrimination acc: %s' % print_stats(x['Accuracy']))
print('Overall discrimination dprime: %s' % print_stats(x['Dprime']))
print('discrimination of unseen trials (Acc): %s, p=%.5f' % (
    print_stats(x['Accuracy']),
    wilcoxon(np.array(x['Accuracy']) - .5)[1]))
print('discrimination of unseen trials (Dprime): %s, p=%.5f' % (
      print_stats(x['Dprime']), wilcoxon(x['Dprime'])[1]))

report.save()
