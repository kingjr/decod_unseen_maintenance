"""Plot the topographical effects obtained in each analysis"""
import numpy as np
import matplotlib.pyplot as plt
from jr.plot import plot_butterfly, plot_gfp, pretty_colorbar
from config import report, load
from conditions import analyses

cmap = plt.get_cmap('gist_rainbow')
colors = cmap(np.linspace(0, 1., len(analyses) + 1))
tois = [(-.100, 0.050), (.100, .250), (.300, .800), (.900, 1.050)]

analyses = list(analyses)
colors = np.vstack((colors, [0., 0., 0., 0.]))

# Apply contrast on each type of epoch
for analysis, color in zip(analyses, colors):
    # load stats
    evoked, data, p_values, sig, analysis = load(
        'evoked', analysis=('stats_' + analysis['name']))
    evoked_full = evoked.copy()

    sig = np.zeros_like(evoked.data)
    sig[::3, :] = sig[1::3, :] = sig[2::3, :] = p_values < .05
    sig_times = np.array(np.sum(sig, axis=0) > 0., dtype=int)

    # Plot topo
    continuous_tois = np.linspace(0, .500, 6)
    sig = sig[:, np.where((evoked.times >= -.100) & (evoked.times <= .600))[0]]
    evoked.crop(-.100, .600)

    def topomap_clim(data, factor=10, grad=False, baseline=0.):
        if grad:
            vmin, vmax = np.percentile(data, [factor, 100-factor])
            vmin = 0 if vmin < 0 else vmin
        else:
            vmin, vmax = np.percentile(data, [factor, 100-factor])
            vmin, vmax = (-np.max([np.abs(vmin), vmax]),
                          np.max([np.abs(vmin), vmax]))
        # make sure that baseline is around 0, to avoid plotting noise when
        # nothing is sig.
        if baseline is not None:
            baseline = np.where(evoked.times <= baseline)[0][-1]
            bsl_mM = np.percentile(np.abs(data[:, :baseline]), [10, 90])
            while np.ptp(bsl_mM) > (np.ptp([vmin, vmax]) / 2.):
                bsl_mM = np.percentile(np.abs(data[:, :baseline]), [10, 90])
                vmax *= 1.1
                vmin = vmin if grad else vmin * 1.1

        smin, smax = '%.2f' % vmin, '%.2f' % vmax
        if vmax < 5e-3:
            smin = '0' if vmin == 0 else '%.0e' % vmin
            smax = '%.0e' % vmax
        return vmin, vmax, smin, smax

    vmin, vmax, smin, smax = topomap_clim(evoked.data[::3, :], grad=True)
    opts = dict(sensors=False, scale=1, contours=False,
                times=continuous_tois, average=.025, colorbar=True, show=False)
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
    # fig_mag.tight_layout()
    report.add_figs_to_section(fig_mag, 'topo_mag', analysis['name'])

    # Plot butterfly
    fig_butt_m, ax = plt.subplots(1, figsize=fig_grad.get_size_inches())
    plot_butterfly(evoked, ax=ax, sig=sig, color=color, ch_type='mag')
    # ax.axvline(800, color='k')
    ax.set_xlim([-100, 600])
    ax.set_xlabel('Times (ms)', labelpad=-15)
    # fig_butt_m.tight_layout()
    report.add_figs_to_section(fig_butt_m, 'butterfly_mag', analysis['name'])

    # plot GFP
    fig_butt_gfp, ax = plt.subplots(1, figsize=fig_grad.get_size_inches())
    plot_gfp(evoked, ax=ax, sig=sig, color=color)
    # ax.axvline(800, color='k')
    ax.set_xlim([-100, 600])
    ax.set_xlabel('Times (ms)', labelpad=-15)
    # fig_butt_gfp.tight_layout()
    report.add_figs_to_section(fig_butt_gfp, 'butterfly_gfp', analysis['name'])

    # Plot topo of mean |effect| on TOI
    evoked_full.data = np.abs(evoked_full.data)
    fig, axes = plt.subplots(1, len(tois), figsize=[9, 2.5])
    fig.subplots_adjust(wspace=0.01, left=0.)
    vmin, vmax, smin, smax = topomap_clim(evoked_full.data[::3, :], grad=True)
    for ax, toi in zip(axes, tois):
        evoked_full.plot_topomap(times=[np.mean(toi)], average=np.ptp(toi),
                                 cmap='afmhot_r', ch_type='grad', show=False,
                                 vmin=vmin, vmax=vmax, contours=False, scale=1,
                                 colorbar=False, sensors=False, axes=ax)
    from matplotlib.image import AxesImage
    objects = axes[-2].get_children()
    im = objects[np.where([isinstance(ii, AxesImage) for ii in objects])[0][0]]
    pretty_colorbar(cax=fig.add_axes([.91, 0.15, .03, .6]),
                    im=im, ticklabels=[smin, '', smax])
    report.add_figs_to_section(fig, 'topo_mean', analysis['name'])
report.save()
