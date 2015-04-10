"""
Final Decoding Pipeline

This script serves as a general pipeline that loops across all possible
contrasts, data types and subjects to create plots and stats, both at the GAT
level and at the diagonal level

created by Niccolo Pescetelli niccolo.pescetelli@psy.ox.ac.uk
"""

# PSEUDO CODE

"# 0) SUMMARY (diagonals only)

for s, subject in enumerate(subjects):
    for contrast in contrasts:
        # load individual data

        # concatenate across subjects

        # plot on the same figure (subplots?)


"# 1) CLASSIC VIEW: presence vs visibility decoding
contrasts = ['presentAbsent','seenUnseen']
for contrast in contrasts:
    for s, subject in enumerate(subjects):
        # retrieve individual data
        # and concatenate
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
cluster_test_interaction(diag_presentSeen, diag_presentUnseen)

# plot different between presentSeen and presentUnseen GATs and do
# cluster test for significance of their difference (interactions) on GAT
# ie: different_from_each_other?
cluster_test_interaction(gat_presentSeen, gat_presentUnseen)

"# 2.1) ADDED VALUE (orientation): we can decode content
for inputType:
    for s, subject in enumerate(subjects):
        # retrieve individual data
        # concatenate
        diag_aggregate

    # plot orientation decoding performance (angle errors) in aggregate form
    cluster_test_main(diag_aggregate)

    # plot tuning curve for aggregate form
    plot_tuning_curve(diag_aggregate)

"# 2.2) ADDED VALUE (orientation): Is it different for different visibilities?
for inputType:
    for s, subject in enumerate(subjects):
        # retrieve individual data
        # and concatenate
        diag_orientation_aggregate
        gat_orientation_aggregate

    # plot orientation decoding performance (angle errors)
    # for each visibility level
    for vis in range(1,5):
        # plot diagonal for each visibility and do cluster test on main effect
        # ie Is it difference from chance?
        cluster_test_main(diag_orientation_vis)

        # plot tuning curve for each visibility level
        plot_tuning_curve(diag_orientation_vis)

    # cluster test on interaction visibility by orientation performance
    # compute for each subject the spearman correlation between visibility
    # rating and orientation performance
    cluster_test_interaction(diag_vis0,diag_vis1, diag_vis2, diag_vis3)

"# 2.3) ADDED VALUE (orientation): are representations stable or dynamic?
for inputType:
    for s, subject in enumerate(subjects):
        # retrieve individual data
        # and concatenate
        gat_aggregate

    # test main effects
    cluster_test_main(gat_aggregate)

    # for each visibility
    for vis in range(1,5):
        cluster_test_main(gat_vis)

    # interaction (regression or spearman correlation)
    cluster_test_interaction(gat_vis0, gat_vis1, gat_vis2, gat_vis3)

"# 3) EXPECTATIONS: is probe decoded earlier if target is seen?
# does the visibility of a present target influence the decoding of the probe?
contrasts = ['presentAbsent','seenUnseen']
for contrast in contrasts:
    for s, subject in enumerate(subjects):
        # retrieve individual data
        # and concatenate
        gat_contrast

    # test main effects
    cluster_test_main(gat_contrast)

# plot difference
plot_probeDecodAUC_presentSeen_vs_presentUnseen
