import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
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
cmap = mpl.colors.LinearSegmentedColormap.from_list('RdPuBu', ['b', 'r'])
# cmap = plt.get_cmap('coolwarm')


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
    # shift = lambda z: np.ptp(z) * .525 * np.array([-1, 1]) + np.mean(z)
    # ax.set_xlim(shift(ax.get_xlim()))
    # ax.set_ylim(shift(ax.get_ylim()))
    # try:
    #     ax.set_zlim(shift(ax.get_zlim()))
    # except AttributeError:
    #     pass
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
    poly.set_alpha(.75)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    poly = PolyCollection(verts_sem, facecolors=colors, edgecolor='none')
    poly.set_alpha(0.5)
    ax.add_collection3d(poly, zs=zs, zdir='y')
    return pretty_ax(ax)


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
ax.text(1., .25, np.mean(x[:, -1, 0], axis=0) + .02, 'Absent', 'x',
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
ax.set_zlim(0, .75)
tmp_planes = ax.zaxis._PLANES
ax.zaxis.grid('off')
ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                    tmp_planes[0], tmp_planes[1],
                    tmp_planes[4], tmp_planes[5])
# fig.show()
fig.savefig('visibility3D.png', dpi=300)


# 2D
fig, ax = plt.subplots(1, figsize=[6, 4])
abs_m = x[:, 0, :].mean(0)
abs_sem = abs_m + x[:, 0, :].mean(0).std(0) / x.shape[0]
pst_m = x[:, 1:, :].mean(1).mean(0)
pst_sem = pst_m + x[:, 1:, :].mean(1).std(0).mean(0) / x.shape[0]
fill = lambda z, c: ax.fill_between(np.linspace(0, 1., 4.), z, alpha=.5,
                                    color=cmap(c),  edgecolor='none')
fill(abs_sem, 0.)
fill(pst_sem, 1.)
fill(abs_m, 0.)
fill(pst_m, 1.)
ax.text(0.1, np.mean(x[:, 0, 0], axis=0) - .05, 'Absent', color=cmap(0.),
        ha='left', va='bottom')
ax.text(1., np.mean(x[:, -1, -1], axis=0) + .05, 'Present',
        color=cmap(1.), ha='right', va='bottom')
ax.set_xlabel('Visibility Rating')
ax.set_ylabel('Response %')
ax.set_xticks(np.linspace(0, 1, 2))
ax.set_yticks(np.linspace(0, .7, 2))
ax.set_ylim(0, .75)
ax = pretty_ax(ax)
fig.savefig('visibility2D.png', dpi=300)
# #############################################################################
# 2. discrimination performance as a function of visibility and contrast
from scipy.stats import wilcoxon
from base import dPrime
x = dict()
x['Accuracy'] = np.zeros((len(subjects), len(contrasts) - 1,
                          len(visibilities)))
x['D prime'] = np.zeros_like(x['Accuracy'])

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
        axes[0].text(vis, np.nanmean(acc, axis=0) + np.ptp(ylim) / 10,
                     ' *'[wilcoxon(acc)[1] < .05], size=15, ha='center')
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
        axes[1].text(contrast, np.nanmean(acc, axis=0) + np.ptp(ylim) / 10,
                     ' *'[wilcoxon(acc)[1] < .05], size=15, ha='center')
    axes[1].set_xlabel('Contrast')
    axes[1].set_xticks(contrasts[1:])
    axes[1].set_xlim(.45, 1.05)
    axes[1].set_yticks(ylim)
    axes[1].set_yticklabels([])

    for ax in axes:
        ax = pretty_ax(ax)
        ax.axhline(ylim[0], linestyle='--', color='k')
        ax.set_ylim(ylim[0] - np.ptp(ylim) / 20, ylim[1])
        fig.savefig('%s.png' % metric, dpi=300)
# plt.show()
# XXX /!\ Dprime is sig for unseen but this is only if nan are counted as 0,
# and not removed.

# #############################################################################
# Effect of previous trial on current visibility
x = np.zeros((len(subjects), len(contrasts), len(visibilities), 2))
for (s, subject), (c, contrast), (v, visibility) in product(
        enumerate(subjects), enumerate(contrasts), enumerate(visibilities)):
    query = 'subject=="%s" and target_contrast==%s and detect_button==%s' % (
        subject, contrast, visibility)
    x[s, c, v, 0] = len(data.query(query + ' and previous_detect_seen==False'))
    x[s, c, v, 1] = len(data.query(query + ' and previous_detect_seen==True'))
# normalize per subject per contrast
for (s, subject), (c, contrast) in product(
        enumerate(subjects), enumerate(contrasts)):
    x[s, c, :, 0] /= np.sum(x[s, c, :, 0])
    x[s, c, :, 1] /= np.sum(x[s, c, :, 1])


fig, axes = plt.subplots(2, 1, figsize=[4, 7])
ax = axes[0]
plot_sem(np.linspace(0, 1, 4), x[:, :, :, 1].mean(1), color='r', ax=ax)
plot_sem(np.linspace(0, 1, 4), x[:, :, :, 0].mean(1), color='b', ax=axes[0])
ax.text(0.05, .35, 'Previously unseen', color='b')
ax.text(0.05, .12, 'Previously seen', color='r')
ax = pretty_ax(axes[0])
ax.set_ylabel('Response %')
ax.set_ylim(.1, .4)
ax.set_yticks([.1, .4])
ax.set_xticks([0, 1])

ax = axes[1]
contrast = np.squeeze(np.mean(x[:, :, :, 1] - x[:, :, :, 0], axis=1))
ax.set_ylim([-.15, .15])
plot_sem(np.linspace(0, 1, 4), contrast, color='k', ax=ax)
fill_between_gradient(np.hstack((0, np.linspace(0, 1, 4), 1)),
                      np.hstack((0, contrast.mean(0), 0)),
                      ax=ax, cmap='seismic', clim=[-.5, 1.5])
ax = pretty_ax(ax)
ax.set_xticks([0, 1])
ylim = ax.get_ylim()
ax.axhline(0., linestyle='--', color='k')
ax.set_yticks(ylim)
ax.set_xlabel('Visibilities')
ax.set_ylabel('P. Seen - P. Unseen')
# plt.show()

fig.savefig('visibility_prior.png', dpi=300)
# #############################################################################
# Effect of previous trial on current orientation XXX for later
