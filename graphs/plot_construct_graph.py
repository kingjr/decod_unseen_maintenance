import matplotlib.pyplot as plt
import numpy as np
import pickle
from scripts.config import paths
from jr.gif import writeGif, Figtodat
from jr.gat.graphs import plot_graph
from scripts.config import report, analyses
from base import stats

analyses = [a for a in analyses if a['name'] in
            ['target_present', 'target_circAngle']]

for analysis in analyses:
    # load data
    with open(paths('score', analysis='stats_' + analysis['name']), 'rb') as f:
        out = pickle.load(f)
        scores = np.array(out['scores']) - analysis['chance']
        p_values = out['p_values']
        times = out['times']

    # compute stats
    p_values = stats(scores)

    # normalize
    weights = np.mean(scores, axis=0)
    weights /= np.percentile(np.abs(weights), 95)
    # fix bug
    node_size = np.abs(np.diagonal(weights)) * 100
    # threshold with significance
    weights *= p_values < .05
    # vizulization parameters
    nx_params = dict(edge_curve=False, node_alpha=1., negative_weights=True,
                     edge_alpha=.05, weights_scale=1, clim=[-.5, .5],
                     final_pos='horizontal', prune=p_values > .05,
                     edge_color=plt.get_cmap('coolwarm'))
    # final construction ------------------------------------------------------
    fig, ax = plt.subplots(1, figsize=[10, 10])
    G, nodes, edges = plot_graph(weights, ax=ax, iterations=100,
                                 node_size=node_size, **nx_params)
    nodes._alpha = 0.
    nodes.set_edgecolors((1., 1., 1., 0.))
    ax.patch.set_visible(False)
    report.add_figs_to_section(fig, analysis['name'], 'network')

    # snapshots construction --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=[12, 3])
    fig.subplots_adjust(wspace=0, hspace=0)
    for iteration, ax in zip([0, 3, 10], axes):
        G, nodes, edges = plot_graph(weights, ax=ax, iterations=int(iteration),
                                     node_size=node_size / 3., **nx_params)
        nodes._alpha = 0.
        nodes.set_edgecolors((1., 1., 1., 0.))
        ax.patch.set_visible(False)
    report.add_figs_to_section(fig, analysis['name'], 'build')

    # animation construction --------------------------------------------------
    images = list()
    for iteration in np.logspace(0, 2., 25):
        print iteration
        fig, ax = plt.subplots(1, figsize=[10, 10], facecolor='w')
        G, nodes, edges = plot_graph(weights, ax=ax, iterations=int(iteration),
                                     node_size=node_size / 3., **nx_params)
        nodes._alpha = 0.
        nodes.set_edgecolors((1., 1., 1., 0.))
        im = Figtodat.fig2img(fig)
        images.append(im)
    name = "/network_construction%s.gif" % analysis['name']
    writeGif(report.report.data_path + name, images)
    report.add_images_to_section(report.report.data_path + name,
                                 analysis['name'], 'build')
    plt.close('all')
report.save()
