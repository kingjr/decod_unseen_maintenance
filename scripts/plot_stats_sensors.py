import pickle
import numpy as np
import matplotlib.pyplot as plt
from jr.plot import plot_butterfly, plot_gfp
from scripts.config import (report, paths, analyses)

cmap = plt.get_cmap('gist_rainbow')
colors = cmap(np.linspace(0, 1., len(analyses) + 1))

# Apply contrast on each type of epoch
for analysis, color in zip(analyses, colors):

    # load stats
    pkl_fname = paths('evoked', analysis=('stats_' + analysis['name']))
    with open(pkl_fname, 'rb') as f:
        evoked, data, p_values, sig, analysis = pickle.load(f)
    sig = np.zeros_like(evoked.data)
    sig[::3, :] = sig[1::3, :] = sig[2::3, :] = p_values < .05
    sig_times = np.array(np.sum(sig, axis=0) > 0., dtype=int)

    # Plot topo
    tois = np.linspace(0, .500, 6)
    sig = sig[:, np.where((evoked.times > -.100) & (evoked.times <= .600))[0]]
    evoked.crop(-.100, .600)

    def topomap_clim(data, factor=10, grad=False):
        if grad:
            # m = np.median(data)
            # s = np.median(np.abs(data - m))
            # vmin, vmax = m - factor * s, m + factor * s
            vmin, vmax = np.percentile(data, [factor, 100-factor])
            vmin = 0 if vmin < 0 else vmin
        else:
            # s = np.median(np.abs(data))
            # vmin, vmax = -factor * s, factor * s
            vmin, vmax = np.percentile(data, [factor, 100-factor])
            vmin, vmax = (-np.max([np.abs(vmin), vmax]),
                          np.max([np.abs(vmin), vmax]))
        smin, smax = '%.2f' % vmin, '%.2f' % vmax
        if vmax < 5e-3:
            smin = '0' if vmin == 0 else '%.0e' % vmin
            smax = '%.0e' % vmax
        return vmin, vmax, smin, smax

    vmin, vmax, smin, smax = topomap_clim(evoked.data[::3, :], grad=True)
    opts = dict(sensors=False, scale=1, contours=False,
                times=tois, average=.025, colorbar=True, show=False)
    fig_grad = evoked.plot_topomap(cmap='afmhot_r', ch_type='grad',
                                   vmin=vmin, vmax=vmax, **opts)
    cax = fig_grad.get_children()[-1]
    cax.set_yticks([vmin, vmax])
    cax.set_yticklabels([smin, smax])
    cax.set_title('')
    report.add_figs_to_section(fig_grad, 'topo_grad', analysis['name'])

    vmin, vmax, smin, smax = topomap_clim(evoked.data[2::3, :])
    fig_mag = evoked.plot_topomap(ch_type='mag', vmin=vmin, vmax=vmax, **opts)
    cax = fig_mag.get_children()[-1]
    cax.set_yticks([vmin, vmax])
    cax.set_yticklabels([smin, '', smax])
    cax.set_title('')
    fig_mag.tight_layout()
    report.add_figs_to_section(fig_mag, 'topo_mag', analysis['name'])

    # Plot butterfly
    fig_butt_m, ax = plt.subplots(1, figsize=fig_grad.get_size_inches())
    plot_butterfly(evoked, ax=ax, sig=sig, color=color, ch_type='mag')
    # ax.axvline(800, color='k')
    ax.set_xlim([-100, 600])
    ax.set_xlabel('Times (ms)', labelpad=-15)
    fig_butt_m.tight_layout()
    report.add_figs_to_section(fig_butt_m, 'butterfly_mag', analysis['name'])

    # plot GFP
    fig_butt_gfp, ax = plt.subplots(1, figsize=fig_grad.get_size_inches())
    plot_gfp(evoked, ax=ax, sig=sig, color=color)
    # ax.axvline(800, color='k')
    ax.set_xlim([-100, 600])
    ax.set_xlabel('Times (ms)', labelpad=-15)
    fig_butt_gfp.tight_layout()
    report.add_figs_to_section(fig_butt_gfp, 'butterfly_gfp', analysis['name'])

report.save()
