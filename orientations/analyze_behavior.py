import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
mpl.rcParams['legend.fontsize'] = 10

import pandas
from orientations.utils import get_events
from itertools import product
from scripts.config import paths  # , subjects
from base import plot_sem
subjects = [
    'ak130184', 'el130086', 'ga130053', 'gm130176', 'hn120493',
    'ia130315', 'jd110235', 'jm120476', 'ma130185', 'mc130295',
    'mj130216', 'mr080072', 'oa130317', 'rg110386', 'sb120316',
    'tc120199', 'ts130283', 'yp130276', 'av130322', 'ps120458']

# Gather all beahvioral data
subjects_events = list()
for subject in subjects:
    bhv_fname = paths('behavior', subject=subject)
    events = get_events(bhv_fname)
    events['subject'] = subject
    subjects_events.append(events)
data = pandas.concat(subjects_events)

contrasts = [0., .5, .75, 1.]
visibilities = [0., 1., 2., 3.]

# ##############################################################################
# Plotting functions

# set color map
# cmap = mcol.LinearSegmentedColormap('black_green',
#     {'red':   ([0.] * 3, (1.0, 0.0, 0.0)),
#      'green': ([0.] * 3, [1.] * 3),
#      'blue':  ([0.] * 3, (1.0, 0.0, 0.0))}, 256)
# cmap = mcol.LinearSegmentedColormap.from_list('RdPuBu', ['b', 'r'])
cmap = plt.get_cmap('coolwarm')


def pretty_ax(ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.tick_params(colors='dimgray')
    ax.xaxis.label.set_color('dimgray')
    ax.yaxis.label.set_color('dimgray')
    try:
        ax.zaxis.label.set_color('dimgray')
    except AttributeError:
        pass
    try:
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    except ValueError:
        pass
    ax.spines['left'].set_color('dimgray')
    ax.spines['bottom'].set_color('dimgray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


def make_3d_plot(ax, xs, y, zs, colors):
    # make areas
    verts = []
    verts_sem = []
    for ii in range(y.shape[2]):
        ys = np.mean(y[:, :, ii], axis=0)
        verts.append(list(zip(np.hstack((0., xs, 1.)),
                              np.hstack((0., ys, 0.)))))
        ys += np.std(x[:, :, ii], axis=0) / np.sqrt(len(subjects))
        verts_sem.append(list(zip(np.hstack((0., xs, 1.)),
                                  np.hstack((0., ys, 0.)))))
    poly = PolyCollection(verts, facecolors=colors, edgecolor=colors,
                          linewidth=2)
    poly.set_alpha(1.)
    poly = PolyCollection(verts_sem, facecolors=colors, edgecolor='none')
    ax.add_collection3d(poly, zs=zs, zdir='y')
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')
    return pretty_ax(ax)


# #############################################################################
# 1. report of visibility as a function of contrast
x = np.zeros((len(subjects), len(contrasts), len(visibilities)))
for (s, subject), (c, contrast), (v, visibility) in product(
        enumerate(subjects), enumerate(contrasts), enumerate(visibilities)):
    query = 'subject=="%s" and target_contrast==%s and detect_button==%s' % (
        subject, contrast, visibility)
    x[s, c, v] = len(data.query(query))
# normalize per subject per contrast
for (s, subject), (c, contrast) in product(
        enumerate(subjects), enumerate(contrasts)):
    x[s, c, :] /= np.sum(x[s, c, :])

fig = plt.figure(figsize=[6, 4])
ax = fig.gca(projection='3d')
ax = make_3d_plot(ax, np.linspace(0, 1., 4.), x, np.linspace(.25, 1., 4.),
                  [cmap(i) for i in contrasts])
ax.text(1., .25, np.mean(x[:, -1, 0], axis=0) + .05, 'Absent', 'x',
        color=cmap(0.), ha='right', va='bottom')
ax.text(1., 1, np.mean(x[:, -1, -1], axis=0) + .05, 'Full contrast', 'x',
        color=cmap(1.), ha='right', va='bottom')
ax.set_xlabel('Visibility Rating')
ax.set_ylabel('Contrast')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('Response %', rotation=90)
ax.set_xticks(np.linspace(0, 1, 2))
ax.set_yticks([])
ax.set_zticks(np.linspace(0, .7, 2))
ax.set_ylim(.25, 1.)
ax.set_zlim(0, .7)
tmp_planes = ax.zaxis._PLANES
ax.zaxis.grid('off')
ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                    tmp_planes[0], tmp_planes[1],
                    tmp_planes[4], tmp_planes[5])
fig.show()


# #############################################################################
# 2. discrimination performance as a function of visibility and contrast
from scipy.stats import wilcoxon
from base import dPrime
measure = dict()
measure['Accuracy'] = np.zeros((len(subjects), len(contrasts) - 1,
                                len(visibilities)))
measure['D prime'] = np.zeros_like(measure['Accuracy'])

for (s, subject), (c, contrast), (v, visibility) in product(
        enumerate(subjects), enumerate(contrasts[1:]), enumerate(visibilities)):
    query = ('subject=="%s" and target_contrast==%s and detect_button==%s'
             ' and target_present') % (subject, contrast, visibility)
    x['Accuracy'][s, c, v] = np.nanmean(data.query(query)['discrim_correct'])
    count = lambda tilt, acc: len(data.query(
        query + ' and probe_tilt==%s and discrim_correct==%s' % (tilt, acc)))
    hits = count(1, True)
    misses = count(1, False)
    fas = count(-1, False)
    crs = count(-1, True)
    if (hits + misses) > 0 and (fas + crs) > 0:
        x['D prime'][s, c, v] = dPrime(hits, misses, fas, crs)['d']
    else:
        x['D prime'][s, c, v] = np.nan

for metric, ylim, in zip(['Accuracy', 'D prime'], ((.5, 1.), (0., 3.))):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[5, 3])
    # Visibility
    accuracy = np.squeeze(np.nanmean(x[metric], axis=1))
    plot_sem(visibilities, accuracy, color='k', ax=axes[0])
    for vis, acc in zip(visibilities, accuracy.T):
        axes[0].text(vis, np.nanmean(acc, axis=0) + .05,
                     ' *'[wilcoxon(acc)[1] < .05], size=15, ha='center')
    axes[0].set_xlabel('Visibility Rating')
    axes[0].set_ylabel(metric)
    axes[0].set_xlim(-.3, 3.3)
    axes[0].set_xticks(visibilities)
    # Contrast
    accuracy = np.squeeze(np.nanmean(x[metric], axis=2))
    plot_sem(contrasts[1:], accuracy, color='k', ax=axes[1])
    for contrast, acc in zip(contrasts[1:], accuracy.T):
        axes[1].text(contrast, np.nanmean(acc, axis=0) + .05,
                     ' *'[wilcoxon(acc)[1] < .05], size=15, ha='center')
    axes[1].set_xlabel('Contrast')
    axes[1].set_xticks(contrasts[1:])
    axes[1].set_xlim(.45, 1.05)
    axes[1].set_yticklabels([])

    for ax in axes:
        ax = pretty_ax(ax)
        ax.axhline(.5, linestyle='--', color='k')
        ax.set_yticks(ylim)
        ax.set_ylim(ylim[0] - .05, ylim[1])
plt.show()
# XXX /!\ Dprime is sig for unseen but this is only if nan are counted as 0,
# and not removed.
