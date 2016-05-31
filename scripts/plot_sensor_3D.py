import os.path as op
import mne
from mne.forward import make_field_map
from mne.viz import plot_evoked_field
from .config import load
from .conditions import analyses

analyses = list(analyses)
analyses.append(dict(name='angle_bias', title='Angle Bias'))

# evoked_full.plot_joint()
data_path = mne.datasets.sample.data_path()
subjects_dir = data_path + '/subjects'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
evoked_ = mne.read_evokeds(fname, baseline=(None, 0), proj=True)[0]
evoked_.pick_types(meg='mag')

maps = make_field_map(evoked_, trans=trans_fname, subjects_dir=subjects_dir,
                      subject='sample', n_jobs=-1, meg_surf='head')
head = plot_evoked_field(evoked_, maps, time=.1)

analysis = analyses[0]
_evoked_, data, p_values, sig, analysis = load(
    'evoked', analysis=('stats_' + analysis['name']))
_evoked_.pick_types(meg='mag')
evoked_.data = _evoked_.data
evoked_.times = _evoked_.times
evoked_.data -= .5
evoked_.data *= 1e-12
evoked_.pick_types(meg='mag')
head = plot_evoked_field(evoked_, maps, time=.1)
