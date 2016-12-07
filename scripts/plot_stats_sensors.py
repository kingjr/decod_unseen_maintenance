"""Plot the topographical effects obtained in each analysis"""
import numpy as np
import matplotlib.pyplot as plt
from jr.plot import plot_butterfly, plot_gfp, pretty_colorbar
from config import report, load
from conditions import analyses, tois

# Apply contrast on each type of epoch
for analysis in analyses:
    # load stats
    evoked, _, p_values, sig, _ = load(
        'evoked', analysis=('stats_' + analysis['name']))
    evoked.data -= analysis['chance']  # to avoid interpolation bug
    evoked_full = evoked.copy()  # used for tois

    # set sig
    sig = np.zeros_like(evoked.data)
    sig[::3, :] = sig[1::3, :] = sig[2::3, :] = p_values < .05
    sig_times = np.array(np.sum(sig, axis=0) > 0., dtype=int)
    sig = sig[:, np.where((evoked.times >= -.100) & (evoked.times <= .600))[0]]

    # define color limit
    toi = np.where((evoked_full.times >= tois[0][0]) &
                   (evoked_full.times <= tois[0][1]))[0]
    vmin = np.percentile(np.mean(evoked_full.data[:, toi], axis=1), 99)
    max_toi = 3
    if 'target' in analysis['name']:
        max_toi = 1
    toi = np.where((evoked_full.times >= tois[max_toi][0]) &
                   (evoked_full.times <= tois[max_toi][1]))[0]
    vmax = np.percentile(np.mean(evoked_full.data[:, toi], axis=1), 99)
    vmin_mag = analysis['chance'] - vmax
    if vmax - vmin < .005:
        vmax += .005

    # clim text
    smin = '%.2f' % (vmin + analysis['chance'])
    smin_mag = '%.2f' % (vmin_mag + analysis['chance'])
    smax = '%.2f' % (vmax + analysis['chance'])
    if vmax < 5e-3:
        smin = '0' if vmin == 0 else '%.0e' % vmin
        smax = '%.0e' % vmax

    # Plot topo snapshots
    evoked.crop(-.100, .600)
    opts = dict(sensors=False, scale=1, contours=False, show=False,
                times=np.linspace(0, .500, 6), average=.025, colorbar=True)
    fig_grad = evoked.plot_topomap(cmap=analysis['cmap'], ch_type='grad',
                                   vmin=vmin, vmax=vmax, **opts)
    cax = fig_grad.get_children()[-1]
    cax.set_yticks([vmin, vmax])
    cax.set_yticklabels([smin, smax])
    cax.set_title('')
    report.add_figs_to_section(fig_grad, 'topo_grad', analysis['name'])

    fig_mag = evoked.plot_topomap(ch_type='mag', vmin=vmin_mag, vmax=vmax,
                                  **opts)
    cax = fig_mag.get_children()[-1]
    cax.set_yticks([vmin, vmax])
    cax.set_yticklabels([smin_mag, '', smax])
    cax.set_title('')
    # fig_mag.tight_layout()
    report.add_figs_to_section(fig_mag, 'topo_mag', analysis['name'])

    # Plot butterfly
    fig_butt_m, ax = plt.subplots(1, figsize=fig_grad.get_size_inches())
    plot_butterfly(evoked, ax=ax, sig=sig, color=analysis['color'],
                   ch_type='mag')
    # ax.axvline(800, color='k')
    ax.set_xlim([-100, 600])
    ax.set_xlabel('Times (ms)', labelpad=-15)
    # fig_butt_m.tight_layout()
    report.add_figs_to_section(fig_butt_m, 'butterfly_mag', analysis['name'])

    # plot GFP
    fig_butt_gfp, ax = plt.subplots(1, figsize=fig_grad.get_size_inches())
    plot_gfp(evoked, ax=ax, sig=sig, color=analysis['color'])
    # ax.axvline(800, color='k')
    ax.set_xlim([-100, 600])
    ax.set_xlabel('Times (ms)', labelpad=-15)
    # fig_butt_gfp.tight_layout()
    report.add_figs_to_section(fig_butt_gfp, 'butterfly_gfp', analysis['name'])

    # Plot topo of mean |effect| on TOI
    evoked_full.data = np.abs(evoked_full.data)
    fig, axes = plt.subplots(1, len(tois), figsize=[9, 2.5])
    fig.subplots_adjust(wspace=0.01, left=0.)
    for ax, toi in zip(axes, tois):
        evoked_full.plot_topomap(times=[np.mean(toi)], average=np.ptp(toi),
                                 cmap=analysis['cmap'], ch_type='grad',
                                 show=False,
                                 vmin=vmin, vmax=vmax, contours=False, scale=1,
                                 colorbar=False, sensors=False, axes=ax)
    from matplotlib.image import AxesImage
    objects = axes[-2].get_children()
    im = objects[np.where([isinstance(ii, AxesImage) for ii in objects])[0][0]]
    pretty_colorbar(cax=fig.add_axes([.91, 0.15, .03, .6]),
                    im=im, ticklabels=[smin, '', smax])
    report.add_figs_to_section(fig, 'topo_mean', analysis['name'])
report.save()
