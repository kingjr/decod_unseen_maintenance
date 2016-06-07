
#     for present in [False, True]:
#         sel = events['target_present'] == present
#         evoked = epochs[sel].average()
#         evo_cond.append(np.median(epochs[sel], axis=0))
#         stc = apply_inverse(evoked, inv,  method='dSPM')
#         stc_morph = morph_data(subject, 'fsaverage', stc,
#                                [np.arange(10242)] * 2, n_jobs=-1)
#         stc_cond.append(stc_morph.data)
#     evo_conds.append(evo_cond)
#     stc_conds.append(stc_cond)
#
# evo_conds = np.array(evo_conds)
# evoked.data = np.mean(evo_conds[1] - evo_conds[0], axis=0)
#
# stc_conds = np.array(stc_conds)
# stc_morph._data = np.mean(stc_conds[1] - stc_conds[0], axis=0)
# this_stc_morph = stc_morph.copy()
# this_stc_morph.crop(-.100, .800)
# brain = this_stc_morph.plot(hemi='split', subject='fsaverage',
#                             views=['lat', 'med'])
# brain.set_time(180)
#
#
# # import pickle
# # out = [stc_morph, stc_conds, evoked, evo_conds]
# # with open('source.pkl', 'wb') as f:
# #     pickle.dump(out, f)
#
# # morph = compute_morph_matrix(subject, 'fsaverage', stc.vertices,
# #                              vertices_to=[np.arange(10242)] * 2,
# #                              subjects_dir=paths('freesurfer'))
# # from mne import compute_morph_matrix
