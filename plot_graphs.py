import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from jr.gif import writeGif, Figtodat
from jr.gat.graphs import plot_graph, animate_graph
from config import report, load
from conditions import analyses
from base import stats

analyses = [a for a in analyses if a['name'] in
            ['target_present', 'target_circAngle']]

all_scores = list()
for analysis in analyses:
    # load data
    out = load('score', analysis='stats_' + analysis['name'])
    scores = np.array(out['scores'])
    times = out['times']
    toi = np.where(times < 1.200)[0]

    # normalize to combine different analyses
    scores = np.mean(scores, axis=0)
    scores -= analysis['chance']
    scores[out['p_values'] > .05] = 0.
    scores /= np.max(np.diag(scores))
    scores = scores[toi, :][:, toi]
    # all_scores.append(scores)

    # weights = np.mean(all_scores, axis=0)
    weights = scores

    # MDS
    mds = MDS(dissimilarity='precomputed',  random_state=0)
    abs_weights = (weights + weights.T) / 2.
    abs_weights = np.abs(abs_weights)
    pos = mds.fit_transform(abs_weights.max() - abs_weights)

    # plot
    weights /= np.percentile(np.abs(weights), 95)
    node_size = np.abs(np.diagonal(weights)) * 100
    weights *= 2  # to see the weights
    clim = [-1., 1.]
    nx_params = dict(edge_curve=False, node_alpha=1., negative_weights=True,
                     edge_alpha=.05, weights_scale=1, clim=clim,
                     final_pos='horizontal',
                     edge_color=plt.get_cmap('coolwarm'))

    fig, ax = plt.subplots(1, figsize=[10, 10])
    G, nodes, edges = plot_graph(weights, ax=ax, iterations=0,
                                 node_color=plt.cm.plasma,
                                 node_size=node_size, init_pos=pos.T,
                                 **nx_params)
    nodes._alpha = 0.
    nodes.set_edgecolors((1., 1., 1., 0.))
    ax.patch.set_visible(False)
    report.add_figs_to_section(fig, analysis['name'], 'network')

# Dynamics seen unseen --------------------------------------------------------
#
# fig.set_facecolor(None)
# anim = animate_graph(weights, G, nodes, times=times * 1e3,
#                      clim=clim, cmap='RdBu_r')
# gif_name = report.report.data_path + '/' + analysis['name'] + '_dyn.gif'
# anim.save(gif_name, writer='imagemagick', dpi=75)
# anim = animate_graph(weights[::2, :], G, nodes,
#                      times=times[::2] * 1e3, clim=clim,
#                      cmap='RdBu_r')
# report.add_images_to_section(gif_name, analysis['name'], 'gif')
# gif_name = gif_name[:-4] + '_fast.gif'
# anim.save(gif_name, writer='imagemagick', dpi=50)

# report.add_images_to_section(gif_name, analysis['name'] + 'fast', 'gif')

# def snapshot(time, title, clim=None):
#     dynamics = np.mean(scores, axis=0)
#     if clim is None:
#         cmax = np.percentile(np.abs(dynamics), 99)
#         clim = -cmax, cmax
#     cmap = plt.get_cmap('bwr')
#     nframe = np.where(times > time)[0][0]
#     dynamic = dynamics[nframe, :]
#     colors = list()
#     zorders = list()
#     sizes = list()
#     for ii in G.nodes():
#         color = (dynamic[ii] - clim[0]) / np.ptp(clim)
#         color = color if color < 1. else 1.
#         color = color if color > 0. else 0.
#         colors.append(cmap(color))
#         zorders.append(dynamic[ii])
#         sizes.append(np.abs(dynamic[ii]) * 500)
#     nodes.set_facecolors(colors)
#     nodes.set_sizes(sizes)
#     zorders = np.argsort(zorders) + 10000
#     # nodes.set_zorder(zorders)
#     nodes.set_edgecolors(colors)
#     ax.set_title('%s ms' % (int(times[nframe] * 100) * 10),
#                  fontdict=dict(horizontalalignment='left'))
#     fig.canvas.draw()
#     fig.tight_layout()
#     fname = report.report.data_path + '/' + analysis['name']
#     fig.savefig(fname + '_dyn%i.png' % (time * 1e3),
#                 dpi=100, transparent=True,)
#     report.add_figs_to_section(fig, analysis['name'] + str((time * 1e3)),
#                                'snapshot')
#
# # tois = [150, 190, 310, 410, 610, 930, 990]
# tois = np.arange(100, 1000, 100)
# for t in tois:
#     snapshot(t/1e3, '%i ms' % (1e3 * t))
report.save()
