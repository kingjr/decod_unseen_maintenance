cd('/home/niccolo/Dropbox/DOCUP/scripts');
clear all
close all
clc

%% libraries & toolboxes
setup_paths

%% All subjects
setup_subjects;

%% useful variables
stim_contrasts   = [0 .5 .75 1];
angles      = [15 45 75 105 135 165];

%% Preprocessing
if 0 % silence this part if preprocessing already done
    for sbj_number = 1:length(SubjectsList)
        % Select subject
        sbj_initials = SubjectsList{sbj_number};
        data_path = [path 'data/' sbj_initials '/'] ;
        
        preproc_erf;
        % XXX preproc_erf_response_lock : change triggers to ttl>25
        %preproc_gamma;
        
        preproc_fixTriggers;
        %preproc_rejectTrials;  % to be done using fieldtrip art summary tutorial
    end
    % to be finished, and subsequently put in loop
    preproc_timeFrequency; % mean time frequency across all conditions
    
    % preproc_TFoi, it saves each FOI separately keeping each trial
    preproc_TFoi; % single trial TF but on selected frequency bands of interest
end

%% within subject univariate stats
if 0 %switch on/off
for sbj_number = 1:length(SubjectsList)
    tic
     % Select subject
    subject = SubjectsList{sbj_number};
    data_path = [path 'data/' subject '/'] ;
    
    % Average in each condition
    univariate_erf;  % (mean amplitude)
    
    % Within subject contrasts of interest
    univariate_erf_stats; % (first order)
    
    % average in gamma
    % average in time freq
    % stats in gamma
    % stats in time freq
    toc
end
end
% tic
% univariate_concatenate; %RAM saturates at the moment
% toc

%% across subjects univariate stats
% XXX Arrange this so that it can take response lock, time freq etc
univariate_erf_statsAcross;
% univariate_erf_statsSecondOrder; % did not manage yet => should be done 

%% decoding
for s = 1:length(SubjectsList)
    s
    % select subject and details
    subject = SubjectsList{s};
    
    % decoding on event related field data
    decode_erf;
    %XXX decode_gamma
    %XXX decode_timeFreq etc
end

%% plotting
plot_univariateWithin % for each subject plot topos etc
%plot_univariateSecondOrder
plot_univariateAcross;

%% Across subjects plots
plot_acrossSubj_topos;

%-- getting the mean target angle decoding performance across time
%-- getting the mean target angle decoding performance across time SORTED BY SEEN OR UNSEEN
%-- getting the mean target angle decoding performance generalized across time (squared one)
%-- getting the mean target angle decoding performance generalized across time (squared one) SORTED BY SEEN UNSEEN
%-- (getting the mean probe angle decoding performance across time)
%-- getting the mean probe angle decoding performance generalized across time (squared one)
%-- getting the mean probe angle decoding performance generalized across time realigned on target (squared one)
%-- getting the mean visibility (present trials) decoding performance across time
plot_acrossSubj_decoding;

plot_across_decode_circstats
plot_across_decode_SVC

%% important controls
control_decodeEOG;

%% in progress
in_progess

% change name of postproc_reg2angle into decode_reg2angle

