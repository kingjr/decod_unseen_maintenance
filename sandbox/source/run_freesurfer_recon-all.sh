#!/usr/bin/env bash
# We need
# - 1. 'mri' folder, containing one folder per subject in which a
#      'anat_SUBJECTID.nii' can be found
# - 2. an empty 'subjects' folder (that will get freesurfer's outputs)

export MAIN_DIR=/media/harddrive/2013_meg_ambiguity/python/data/ambiguity
export MRI_DIR=$MAIN_DIR/mri
export SUBJECTS_DIR=$MAIN_DIR/subjects

# Go to main directory
cd $MAIN_DIR

# List subjects' folder
subjects=$(ls $MRI_DIR)

# Count number of subjects
n=$(ls $MRI_DIR -1 | wc -l)

# Convert .nii files into .mgz
for subject in $subjects; do
    # check whether mgz file hasn't been created yet
    if [ ! -e $subject/T1.mgz ]; then
      mri_convert $MRI_DIR/$subject/anat_$subject.nii $MRI_DIR/$subject/T1.mgz;
    fi;
    error_log=$SUBJECTS_DIR/$subject/scripts/IsRunning.lh+rh;
    if [ -e $error_log ]; then
        echo  removing $error_log;
        rm $error_log;
    fi;
done

# XXX put the -j{digit} flag to set n jobs
# need parallel: sudo apt-get install parallel
# you might need to remove the --tolef option from /etc/parallel/config
parallel -j$n 'export SUBJECT={}; echo processing: $SUBJECT; recon-all -all -s $SUBJECT -i $MRI_DIR/$SUBJECT/T1.mgz > $MRI_DIR/$SUBJECT/my-recon-all.txt' ::: $subjects
