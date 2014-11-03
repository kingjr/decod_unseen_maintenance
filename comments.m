% _test_decode4visibilities: In this script we aimed at separating the
% effect specific to visibility form those generated by the contrast. For
% this purpose we trained a series of classifier on visibilities, and
% analyzed their prediction with an ANOVA, to see wether the visibility
% decoding was predicted by contrast and/or by visibility. And it
% works,yeah

%_check_responses_from_meg:  this revealed that the protocol is a
%two-successive response one. You can actually use this code to easily
%identify the response code that will be required for response lock analyses

%_plot_decoding_acc: did not work. We may need to check whether there s a
%bug or not. Indeed, we are skeptical: we can decode
%visibilities;visibility is correalted with accuracy. why can t we decode
%accuracy??

%decod_reg2angle: in some sense, computes prediction-truth by providing the
%predicted angle realigned according to its truth. 1) you should modify its
%name to be more explicit, 2) you should modify it in a similar philosophy
%to decode_reg2angle_predict so that you pass predictions rather than
%results structures.

%decode_defineContrast: check that target angle and probe angle have
%similar length, I suspect that one removed the
%absent trials while the other has NaN.

%_decodOrientations:after one day of struggling, this is not the script to
%look at: we tried correlations, v stats, r stats etc but on the diagonal
%of generalization across time, which had a window width=1. When taking the
%diagonal (classic) decoding, with a window of 4, the difference between
%seen and unseen is super clear. 1) Make a figure of this difference, 2)
%try the ANOVA : contrast x visibility effects on orientation decoding 3)
%compute time generalization with window width=4.

%preproc_timeFrequency: this scripts computes all (high res) time frequency
%for all subjects, but does not keep the trials, and thus prevents any
%condition comparison. 1) make figure of mean tf across all channels that
%will be used to motivate the selection of our FOI 2) Find TF parameters
%that other people use to show gamma effects. 3)Potentially, try to compute
%visibility, present/absent, and response lock contrasts so as to give an
%idea to the reader of what frequency band these classic effects tap onto.
%This requires quite a lot of coding, that differs from the rest.
%Alternatively, you can keep all trials if you have a massive computer.

%preproc_TFoi: preproc single trial for frequencies of interest (typically,
%not more than 7). This if what is going to be fed to the topographies,
%univariate analyses, decoding etc. Please save them separately.

%setup_paths: careful, we need the novel version of fieldtrip because of
%the TF baseline (db).

%tf_basic_contrast: shows topogrqphy of some basic contrast (present versus
%absent), plus some decoding (orientation) and GAT. This script needs to be
%systematize and integrated with the rest of the pipeline to provide
%topogrqphies of each contrast of each FOI, and decoding GAT of each
%contrast too.

%_tf_basic_contrast_allSbj: same as above.

%decode_defineContrast: ProbeAngle was wrong. AND transform this type of
%scripts into functions.


% Overall
% 1. clean scripts & comment as much as possible. Normally, if a code is
% copied and paste, it means that it should be transformed into a function.
% 2. Main story: decod visibility effect (Neural Correlate of Consciousness 
% classic story): but how is representational content modified? 
%  - decoding of orientation decreases massively when unseen
%  - chain of processing characterizes visibility cases
% 3. Other important stories: we identify a large number of parallel
% content and processing: Probe+Target, lambda etc
% 4. If you want add on it github. 
% 5. We have a document where all the analyses we do are specified, and put
% some images of the figure and the reference to the script that generated
% them. This will help you structure the systematic analyses. Tree like
% structure.


%% 2014 11 03
% I added scripts on GitHub and made first modifications to the scripts

% I ran time generalization tAll with wsize =4 for the target and probe
% decoder (SVR) on all subjects. Time points of interest were however reduced
% every 4 time points and from -.2 to 1.2 secs. Plotting results a clearer
% distinction between seen and unseen trials exist for target orientations
% for several measures like angle error, pvalue, rtests and so on. However
% I was concerned that results are due to the lack of trials in the unseen
% present bin compared to the seen present ones. However at second thought
% I realized that the SVR is trained on all present trials. What we see is
% simply the generalization to a general model to a specific case (unseen).
% In other words the SVR was not trained on a smaller number of trials

% I ran time generalization tAll with wsize=4 for the 4 visibilities
% regressor (SVR). I plotted some graphs but I encountered a problem of
% baseline and in general graphs were highly significant also before target
% presentation. However some signal seems to be present given that a clean diagonal
% emerged from the visibile one.

% I transformed decode_defineContrast into a function taking as inputs cfg
% and trials. cfg structure, used by the jr_classify was added a .contrast
% field that specify the kind of contrast under study.