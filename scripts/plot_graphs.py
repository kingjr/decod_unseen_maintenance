import matplotlib.pyplot as plt
import numpy as np
import pickle
# from sklearn.manifold import MDS
from scripts.config import paths
from jr.gif import writeGif, Figtodat
from jr.gat.graphs import plot_graph, animate_graph
from scripts.config import report, analyses
from scripts.base import stats

analyses = [a for a in analyses if a['name'] in
            ['target_present', 'target_circAngle', 'detect_button_pst']]

for analysis in analyses:
    # load data
    with open(paths('score', analysis='stats_' + analysis['name']), 'rb') as f:
        out = pickle.load(f)
        scores = np.array(out['scores']) - analysis['chance']
        p_values = out['p_values']
        times = out['times'] / 1e3  # FIXME

    # compute stats
    p_values = stats(scores)

    # normalize
    weights = np.mean(scores, axis=0)
    weights /= np.percentile(np.abs(weights), 95)
    # fix bug
    weights[0, :] = 0
    weights[:, 0] = 0
    node_size = np.abs(np.diagonal(weights)) * 100
    weights *= p_values < .05
    weights *= 10  # to see the weights
    clim = [-1., 1.]
    nx_params = dict(edge_curve=False, node_alpha=1., negative_weights=True,
                     edge_alpha=.05, weights_scale=1, clim=clim,
                     final_pos='horizontal', prune=p_values > .05,
                     edge_color=plt.get_cmap('coolwarm'))

    # snapshots construction --------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=[12, 3])
    fig.subplots_adjust(wspace=0, hspace=0)
    for iteration, ax in zip([0, 3, 6, 9], axes):
        G, nodes, edges = plot_graph(weights, ax=ax, iterations=int(iteration),
                                     node_size=node_size / 3., **nx_params)
        nodes._alpha = 0.
        nodes.set_edgecolors((1., 1., 1., 0.))
        ax.patch.set_visible(False)
    report.add_figs_to_section(fig, analysis['name'], 'build')

    # animation construction --------------------------------------------------
    images = list()
    for iteration in np.logspace(0, .9, 50):
        print iteration
        fig, ax = plt.subplots(1, figsize=[5, 5], facecolor='w')
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

    # final construction ------------------------------------------------------
    fig, ax = plt.subplots(1, figsize=[10, 10])
    G, nodes, edges = plot_graph(weights, ax=ax, iterations=100,
                                 node_size=node_size, **nx_params)
    nodes._alpha = 0.
    nodes.set_edgecolors((1., 1., 1., 0.))
    ax.patch.set_visible(False)
    report.add_figs_to_section(fig, analysis['name'], 'network')

    # Dynamics ---------------------------------------------------------------
    fig, ax = plt.subplots(1, figsize=[5, 5])
    G, nodes, edges = plot_graph(weights, ax=ax, iterations=100,
                                 node_size=node_size, **nx_params)
    fig.set_facecolor(None)
    anim = animate_graph(weights, G, nodes, times=times * 1e3,
                         clim=clim, cmap='RdBu_r')
    gif_name = report.report.data_path + '/' + analysis['name'] + '_dyn.gif'
    anim.save(gif_name, writer='imagemagick', dpi=75)
    anim = animate_graph(weights[::2, :], G, nodes,
                         times=times[::2] * 1e3, clim=clim,
                         cmap='RdBu_r')
    report.add_images_to_section(gif_name, analysis['name'], 'gif')
    gif_name = gif_name[:-4] + '_fast.gif'
    anim.save(gif_name, writer='imagemagick', dpi=50)
    report.add_images_to_section(gif_name, analysis['name'] + 'fast', 'gif')

    def snapshot(time, title, clim=None):
        dynamics = np.mean(scores, axis=0)
        if clim is None:
            cmax = np.percentile(np.abs(dynamics), 99)
            clim = -cmax, cmax
        cmap = plt.get_cmap('bwr')
        nframe = np.where(times > time)[0][0]
        dynamic = dynamics[nframe, :]
        colors = list()
        zorders = list()
        sizes = list()
        for ii in G.nodes():
            color = (dynamic[ii] - clim[0]) / np.ptp(clim)
            color = color if color < 1. else 1.
            color = color if color > 0. else 0.
            colors.append(cmap(color))
            zorders.append(dynamic[ii])
            sizes.append(np.abs(dynamic[ii]) * 500)
        nodes.set_facecolors(colors)
        nodes.set_sizes(sizes)
        zorders = np.argsort(zorders) + 10000
        # nodes.set_zorder(zorders)
        nodes.set_edgecolors(colors)
        ax.set_title('%s ms' % (int(times[nframe] * 100) * 10),
                     fontdict=dict(horizontalalignment='left'))
        fig.canvas.draw()
        fig.tight_layout()
        fname = report.report.data_path + '/' + analysis['name']
        fig.savefig(fname + '_dyn%i.png' % (time * 1e3),
                    dpi=100, transparent=True,)
        report.add_figs_to_section(fig, analysis['name'] + str((time * 1e3)),
                                   'snapshot')

    # tois = [150, 190, 310, 410, 610, 930, 990]
    tois = np.arange(100, 1000, 100)
    for t in tois:
        snapshot(t/1e3, '%i ms' % (1e3 * t))
report.save()
