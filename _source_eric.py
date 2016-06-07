import os.path as op
import mne
from mne.tests.test_bem import _compare_bem_solutions
recalc = False
subj = 'ak130184'
if recalc:
    bem = mne.make_bem_model(subj, conductivity=(0.3,), verbose=True)
    sol = mne.make_bem_solution(bem, verbose=True)
    mne.write_bem_solution('test-py-homog-bem-sol.fif', sol)
sol = mne.read_bem_solution('test-py-homog-bem-sol.fif')
sol_c = mne.read_bem_solution('ak130184-5120-bem-sol.fif')
_compare_bem_solutions(sol, sol_c)
trans = subj + '-trans.fif'
src = 'ak130184-oct-6-src.fif'
epo = mne.read_epochs(subj + '-epo.fif', preload=True).pick_types(exclude=())
epo.apply_baseline((None, 0))
cov = mne.compute_covariance(epo)
ave = epo.average()
ave.plot_joint()
fwd = mne.make_forward_solution(ave.info, trans, src, sol)
inv = mne.minimum_norm.make_inverse_operator(ave.info, fwd, cov)
stc = mne.minimum_norm.apply_inverse(ave, inv)
brain = stc.plot(time_viewer=True)
