import matplotlib.pyplot as plt
import numpy as np
import pickle
from scripts.config import paths
from jr.gat.graphs import plot_graph, animate_graph
stats_fname = paths('score', subject='fsaverage', data_type='erf',
                    analysis=('stats_target_circAngle'))
with open(stats_fname, 'rb') as f:
    out = pickle.load(f)
    scores = out['scores']
    p_values = out['p_values']
    times = out['times'] / 1e3  # FIXME

connectivity = np.mean(scores, axis=0)
fig, ax = plt.subplots(1, figsize=[5, 5])
connectivity *= p_values < .05
clim = np.percentile(np.abs(connectivity), 90) * np.array([-1, 1])
G, nodes, = plot_graph(connectivity, node_alpha=1.,
                       negative_weights=True, weights_scale=50, ax=ax,
                       edge_color=plt.get_cmap('coolwarm'), clim=clim,
                       final_pos='horizontal')
fig.set_facecolor(None)
anim = animate_graph(np.mean(scores, axis=0), G, nodes, times=times * 1e3,
                     clim=[-.08, .08], cmap='RdBu_r')
anim.save('results/graphs/target_circAngle.gif', writer='imagemagick', dpi=75)
anim = animate_graph(np.mean(scores, axis=0)[::2, :], G, nodes,
                     times=times[::2] * 1e3, clim=[-.08, .08], cmap='RdBu_r')
anim.save('results/graphs/target_circAngle_fast.gif', writer='imagemagick',
          dpi=50)


def snapshot(time, title, clim=None):
    dynamics = np.mean(scores, axis=0)
    if clim is None:
        cmax = np.percentile(np.abs(dynamics), 99)
        clim = -cmax, cmax
    cmap = plt.get_cmap('RdBu_r')
    nframe = np.where(times > time)[0][0]
    dynamic = dynamics[nframe, :]
    colors = list()
    for ii in G.nodes():
        color = (dynamic[ii] - clim[0]) / np.ptp(clim)
        color = color if color < 1. else 1.
        color = color if color > 0. else 0.
        colors.append(cmap(color))
    nodes.set_facecolors(colors)
    ax.set_title('%s ms' % (int(times[nframe] * 100) * 10),
                 fontdict=dict(horizontalalignment='left'))
    fig.canvas.draw()
    fig.tight_layout()
    fig.savefig('results/graphs/target_circAngle_%i.png' % (time * 1e3),
                dpi=100, transparent=True,)

tois = [150, 190, 310, 410, 610, 930, 990]
for t in tois:
    snapshot(t/1e3, '%i ms' % (1e3 * t))
