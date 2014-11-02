%% identify files
files = dir([data_path 'fif/*.fif']);
files = files(SelFileOrder{sbj_number}); % put files in order

%% preprocessing of each block
for f = 1:length(files)
    %% Trial definition
    cfg = [];
    cfg.dataset             = fullfile([data_path 'fif'],files(f).name);
    cfg.headerformat        = 'neuromag_mne';
    cfg.dataformat          = 'neuromag_mne';
    %cfg.trialfun           = 'ft_trialfun_general';
    cfg.trialdef.eventtype  = 'STI101';
    cfg.trialdef.eventvalue = 1:25;
    cfg.trialdef.prestim    = .800; % number, latency in seconds (optional);
    cfg.trialdef.poststim   = 2.200; % number, latency in seconds (optional);
    cfg                     = ft_definetrial(cfg);
    
    %% Read in the data
    %cfg.channel     = {'MEG','MEGREF'};
    cfg.continuous  = 'yes';
    cfg.feedback    = 'gui';
    cfg.lpfilter    = 'yes'; % low pass filter (previous preprocessing)
    cfg.lpfreq      = 16;
    cfg.hpfilter    = 'yes'; % low pass filter (previous preprocessing)
    cfg.hpfreq      = .5;
    cfg.hpfiltord   = 3;
    data            = ft_preprocessing(cfg);
    
    %% Resampling
    cfg             = [];
    cfg.resamplefs  = 128;
    cfg.detrend     = 'no';
    cfg.feedback    = 'gui';
    data            = ft_resampledata(cfg,data);
    
    %% baseline correction
    cfg             = [];
    cfg.demean      = 'yes';
    cfg.baselinewindow  = [-.300 -.050];
    data            = ft_preprocessing(cfg,data);
    
    %% concatenates
    datas{f} = data;
    clear data;
end


%% concatenate within one file
data = ft_appenddata([],datas{:});
data.grad=datas{1}.grad; % fix fieldtrip bug
clear datas;

%% save data
%---- save in a binary file
%---- extract meg data from fieldtirp structure to transform it into a matrix
save([data_path 'preprocessed/' sbj_initials '_preprocessed.mat'], 'data'); % save details

ft2mat = @(data) permute(reshape(cell2mat(data.trial),[size(data.trial{1}), length(data.trial)]),[3 1 2]);
X = ft2mat(data);
binsave([data_path 'preprocessed/', sbj_initials '_preprocessed.dat'], X);
data.Xdim = size(X); % size of the data matrix to be used by the decoder.
data.trial = []; % remove data from mat file
save([data_path 'preprocessed/' sbj_initials '_preprocessed.mat'], 'data'); % save details
clear X;


if 0
    % example of how to retrieve data in classic format use:
    data = loadBin2mat([data_path 'preprocessed/' sbj_initials '_preprocessed.mat']);
end


