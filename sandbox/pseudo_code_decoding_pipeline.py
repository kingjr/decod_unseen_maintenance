"""
This script serves as a general pipeline that loops across all possible
contrasts, data types and subjects to create plots and stats, both at the GAT
level and at the diagonal level

created by Niccolo Pescetelli niccolo.pescetelli@psy.ox.ac.uk
"""

# PSEUDO CODE

"# 1)
allContrasts = ['presentAbsent','seenUnseen', .....]
for contrast in allContrasts:
    for s, subject in enumerate(subjects):
        # retrieve individual data
        load_data
        # define subdivision
        define_subdivision
        # concatenate
        diag_contrast_aggregate
        gat_contrast_aggregate

    # plot AUC on the same graph (overlap or subplot) and do
    # cluster test for significance of each contrast (main effects) on diagonal
    # ie: different_from_chance?
    cluster_test_main(diag_contrast_aggregate)

    # plot GAT presence and GAT seenUnseen and do
    # cluster test for significance of each contrast (main effects) on GAT
    # ie: different_from_chance?
    cluster_test_main(gat_contrast_aggregate)

    # plot presentSeen vs presentUnseen and do
    # cluster test for significance of this difference (interactions) on diagonal
    # ie: different_from_each_other?
    cluster_test_interaction(diag_contrastSeen, diag_contrastUnseen)

    # plot different between presentSeen and presentUnseen GATs and do
    # cluster test for significance of their difference (interactions) on GAT
    # ie: different_from_each_other?
    cluster_test_interaction(gat_contrastSeen, gat_contrastUnseen)
