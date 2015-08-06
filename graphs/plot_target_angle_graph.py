import matplotlib.pyplot as plt
import numpy as np
import pickle
from scripts.config import paths
from model.graphs import plot_graph, animate_graph
stats_fname = paths('score', subject='fsaverage', data_type='erf',
                    analysis=('stats_target_circAngle'))
with open(stats_fname, 'rb') as f:
    out = pickle.load(f)
    scores = out['scores']
    p_values = out['p_values']
    times = out['times']


connectivity = np.mean(scores, axis=0)
fig, ax = plt.subplots(1, figsize=[10, 10])
connectivity *= p_values < .05
G, nodes, = plot_graph(connectivity, prune=p_values>.10, negative_weights=True,
                       weights_scale=50, ax=ax)

anim = animate_graph(connectivity, G, nodes, times=times, clim=[-.08, .08], cmap='RdBu_r')
anim.save('target_circAngle.gif', writer='imagemagick', dpi=75)
anim = animate_graph(connectivity[::2, :], G, nodes, times=times[::2], clim=[-.08, .08], cmap='RdBu_r')
anim.save('target_circAngle_fast.gif', writer='imagemagick', dpi=50)

def snapshot(time, title):
    dynamics = connectivity
    cmap = plt.get_cmap('RdBu_r')
    clim = np.min(dynamics), np.max(dynamics)
    nframe = np.where(times > time)[0][0]
    dynamic = dynamics[nframe, :]
    colors = list()
    for ii in G.nodes():
        color = (dynamic[ii] - clim[0]) / np.ptp(clim)
        color = color if color < 1. else 1.
        color = color if color > 0. else 0.
        colors.append(cmap(color))
    nodes.set_facecolors(colors)
    fig.canvas.draw()
    fig.show()
    ax.set_title('%s ms' % (int(times[nframe])),
                 fontdict=dict(horizontalalignment='left'))
    fig.savefig('target_circAngle_%s.png' % time, dpi=100)

snapshot(.150, '150 ms')
snapshot(.250, '250 ms')
snapshot(.400, '400 ms')
snapshot(.600, '600 ms')
snapshot(.800, 'Probe onset: 800 ms')
snapshot(1.000, 'Probe onset: +200 ms')
