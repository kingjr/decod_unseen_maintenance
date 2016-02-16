import os
import os.path as op
from mne.bem import make_watershed_bem
from mne.commands.mne_make_scalp_surfaces import _run as make_scalp_surfaces
from mne.surface import read_morph_map
from scripts.config import paths, subjects


# export FREESURFER_HOME=/home/jrking/freesurfer
# source $FREESURFER_HOME/SetUpFreeSurfer.sh
# export MNE_ROOT=/home/jrking/MNE-2.7.4-3452-Linux-x86_64
# source $MNE_ROOT/bin/mne_setup_sh
# export LD_LIBRARY_PATH=/home/jrking/anaconda/lib/


subjects_dir = paths('freesurfer')
for subject in subjects:
    # Check freesurfer finished without any errors
    fname = op.join(paths('freesurfer'), subject, 'scripts', 'recon-all.log')
    if op.isfile(fname):
        with open(fname, 'rb') as fh:
            fh.seek(-1024, 2)
            last = fh.readlines()[-1].decode()
        print('{}: ok'.format(subject))
    else:
        print('{}: missing'.format(subject))
        continue

    # Create BEM surfaces
    make_watershed_bem(subject=subject, subjects_dir=subjects_dir,
                       overwrite=True, volume='T1', atlas=False,
                       gcaatlas=False, preflood=None)

    # Make symbolic links
    for surface in ['inner_skull', 'outer_skull', 'outer_skin']:
        from_file = op.join(subjects_dir, subject, 'bem',
                            'watershed/%s_%s_surface' % (subject, surface))
        to_file = op.join(subjects_dir, subject, 'bem',
                          '%s-%s.surf' % (subject, surface))
        if op.exists(to_file):
            os.remove(to_file)
        os.symlink(from_file, to_file)

    # Make scalp surfaces
    make_scalp_surfaces(subjects_dir, subject, force='store_true',
                        overwrite='store_true', verbose=None)

    # Make morphs to fsaverage
    read_morph_map(subject, 'fsaverage', subjects_dir=subjects_dir)
