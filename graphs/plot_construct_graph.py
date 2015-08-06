import matplotlib.pyplot as plt
import numpy as np
import pickle
from scripts.config import paths
from gat.graphs import plot_graph
stats_fname = paths('score', subject='fsaverage', data_type='erf',
                    analysis=('stats_target_circAngle'))
with open(stats_fname, 'rb') as f:
    out = pickle.load(f)
    scores = out['scores']
    p_values = out['p_values']
    times = out['times']


connectivity = np.mean(scores, axis=0)
connectivity *= p_values < .05

fig, ax = plt.subplots(1, figsize=[10, 10], facecolor='w')
G, nodes, = plot_graph(connectivity,
                       negative_weights=True, edge_alpha=.2,
                       weights_scale=50, ax=ax, iterations=0, node_size=10,
                       edge_color=plt.get_cmap('RdBu_r'),
                       clim=[-.10, .10])
nodes.set_edgecolors((1., 1., 1., 0.))



def animate_graph_construction(connectivity, iterations=1000, step=100,
                               **kwargs):
    from matplotlib import animation

    def animate(nframe):
        for ii in range(iterations / step):
            G, nodes, = plot_graph(connectivity, iterations=ii*step, **kwargs)
    anim = animation.FuncAnimation(fig, animate)
    return anim

animate_graph_construction(connectivity, prune=p_values > .10,
                           negative_weights=True, weights_scale=50, ax=ax)


G, nodes, = plot_graph(connectivity, prune=p_values>.10, negative_weights=True,
                       weights_scale=50, ax=ax, iterations=0, edge_color=plt.get_cmap('coolwarm'), clim=[.4, .6])
