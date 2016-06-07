cd ${SUBJECTS_DIR}/${SUBJECT}/bem
mne_watershed_bem --overwrite

ln -sf watershed/${SUBJECT}_inner_skull_surface ${SUBJECT}-inner_skull.surf
ln -sf watershed/${SUBJECT}_outer_skin_surface ${SUBJECT}-outer_skin.surf
ln -sf watershed/${SUBJECT}_outer_skull_surface ${SUBJECT}-outer_skull.surf
