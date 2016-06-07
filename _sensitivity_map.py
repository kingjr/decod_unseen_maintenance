from mne import sensitivity_map, read_forward_solution
from os.path import join
meg_path = '/media/jrking/harddrive/Niccolo/data/s1/'
fs_path = '/media/jrking/harddrive/Niccolo/subjects/'
fwd = read_forward_solution(join(meg_path, 'sss', 'ak130184-meg-fwd.fif'))
grad_map = sensitivity_map(fwd, ch_type='grad', mode='fixed')
mag_map = sensitivity_map(fwd, ch_type='mag', mode='fixed')

mag_map.plot(time_label='Gradiometer sensitivity', subjects_dir=fs_path,
             clim=dict(lims=[0, 50, 100]), subject='ak130184')
